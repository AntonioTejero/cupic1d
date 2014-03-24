/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/

/****************************** HEADERS ******************************/

#include "init.h"

/************************ FUNCTION DEFINITIONS ***********************/

void init_dev(void)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  int dev;
  int devcnt;
  cudaDeviceProp devProp;
  cudaError_t cuError;
  
  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  // check for devices instaled in the host
  cuError = cudaGetDeviceCount(&devcnt);
  if (0 != cuError)
  {
    printf("Cuda error (%d) detected in 'init_dev(void)'\n", cuError);
    cout << "exiting simulation..." << endl;
    exit(1);
  }
  cout << devcnt << " devices present in the host:" << endl;
  for (dev = 0; dev < devcnt; dev++) 
  {
    cudaGetDeviceProperties(&devProp, dev);
    cout << "  - Device " << dev << ":" << endl;
    cout << "    # " << devProp.name << endl;
    cout << "    # Compute capability " << devProp.major << "." << devProp.minor << endl;
  }

  // ask wich device to use
  cout << "Select in wich device simulation must be run: 0" << endl;
  dev = 0;  //cin >> dev;
  
  // set device to be used and reset it
  cudaSetDevice(dev);
  cudaDeviceReset();
  
  return;
}

void init_sim(double **d_rho, double **d_phi, double **d_E, particle **d_e, particle **d_i, 
              int **d_e_bm, int **d_i_bm, double *t, curandStatePhilox4_32_10_t **state)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const double dt = init_dt();
  const int n_ini = init_n_ini();

  // device memory
  
  /*----------------------------- function body -------------------------*/

  // check if simulation start from initial condition or saved state
  if (n_ini == 0) {
    // adjust initial time
    *t = 0.;

    // create particles
    create_particles(d_i, d_i_bm, d_e, d_e_bm, state);

    // initialize mesh variables
    initialize_mesh(d_rho, d_phi, d_E, *d_i, *d_i_bm, *d_e, *d_e_bm);

    // adjust velocities for leap-frog scheme
    adjust_leap_frog(*d_i, *d_i_bm, *d_e, *d_e_bm, *d_E);
    
    cout << "Simulation initialized with " << number_of_particles(*d_e_bm)*2 << " particles." << endl << endl;
  } else if (n_ini > 0) {
    // adjust initial time
    *t = n_ini*dt;

    // read particle from file
    load_particles(d_i, d_i_bm, d_e, d_e_bm, state);
    
    // initialize mesh variables
    initialize_mesh(d_rho, d_phi, d_E, *d_i, *d_i_bm, *d_e, *d_e_bm);

    cout << "Simulation state loaded from time t = " << *t << endl;
  } else {
    cout << "Wrong input parameter (n_ini<0)" << endl;
    cout << "Stoppin simulation" << endl;
    exit(1);
  }
  
  return;
}

/**********************************************************/

void create_particles(particle **d_i, int **d_i_bm, particle **d_e, int **d_e_bm, 
                      curandStatePhilox4_32_10_t **state)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const double n = init_n();        // plasma density
  const double mi = init_mi();      // ion's mass
  const double me = init_me();      // electron's mass
  const double kti = init_kti();    // ion's thermal energy
  const double kte = init_kte();    // electron's thermal energy
  const double ds = init_ds();      // spatial step size
  const int nc = init_nc();         // number of cells   
  int N;                            // initial number of particles of each especie

  cudaError_t cuError;              // cuda error variable
  
  // device memory

  /*----------------------------- function body -------------------------*/
 
  // initialize curand philox states
  cuError = cudaMalloc ((void **) state, CURAND_BLOCK_DIM*sizeof(curandStatePhilox4_32_10_t));
  cu_check(cuError, __FILE__, __LINE__);
  cudaGetLastError();
  init_philox_state<<<1, CURAND_BLOCK_DIM>>>(*state);
  cu_sync_check(__FILE__, __LINE__);

  // calculate initial number of particles
  N = int(n*ds*ds*ds)*nc;
  
  // allocate device memory for particle vectors
  cuError = cudaMalloc ((void **) d_i, N*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_e, N*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);
  
  // allocate device memory for bookmark vectors
  cuError = cudaMalloc ((void **) d_e_bm, 2*ncy*sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_i_bm, 2*ncy*sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);

  // create particles (electrons)
  cudaGetLastError();
  create_particles_kernel<<<1, CURAND_BLOCK_DIM>>>(*d_e, *d_e_bm, kte, me, N, nc, ds, *state);
  cu_sync_check(__FILE__, __LINE__);

  // create particles (ions)
  cudaGetLastError();
  create_particles_kernel<<<1, CURAND_BLOCK_DIM>>>(*d_i, *d_i_bm, kti, mi, N, nc, ds, *state);
  cu_sync_check(__FILE__, __LINE__);

  return;
}

/**********************************************************/

void initialize_mesh(double **d_rho, double **d_phi, double **d_E, particle *d_i, int *d_i_bm, 
                     particle *d_e, int *d_e_bm)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const double phi_p = init_phi_p();    // probe's potential
  const int nn = init_nn();             // number of nodes 
  const int nc = init_nc();             // number of cells 
  
  double *h_phi;                        // host vector for potentials
  
  cudaError_t cuError;                  // cuda error variable
  
  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  // allocate host memory for potential
  h_phi = (double*) malloc(nn*sizeof(double));
  
  // allocate device memory for mesh variables
  cuError = cudaMalloc ((void **) d_rho, nn*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_phi, nn*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_E, nn*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  
  //initialize potential (host memory)
  for (int i = 0; i < nn; i++)
  {
    h_phi[i] = (1.0 - double(i)/double(nc))*phi_p;
  }
  
  // copy potential from host to device memory
  cuError = cudaMemcpy (*d_phi, h_phi, nn*sizeof(double), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  
  // free host memory
  free(h_phi);
  
  // deposit charge into the mesh nodes
  charge_deposition(*d_rho, d_e, d_e_bm, d_i, d_i_bm);
  
  // solve poisson equation
  poisson_solver(1.0e-4, *d_rho, *d_phi);
  
  // derive electric fields from potential
  field_solver(*d_phi, *d_E);
  
  return;
}

/**********************************************************/

void adjust_leap_frog(particle *d_i, int *d_i_bm, particle *d_e, int *d_e_bm, double *d_E)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const double mi = init_mi();          // ion's mass
  const double me = init_me();          // electron's mass
  const double ds = init_ds();          // spatial step size
  const double dt = init_dt();          // temporal step size
  const int nc = init_nc();             // number of cells 
  const int nn = init_nn();             // number of nodes
  
  int N = number_of_particles(d_i_bm);  // number of particles of each especie (same for both)

  dim3 griddim, blockdim;               // kernel execution configurations
  size_t sh_mem_size;                   // shared memory size
  cudaError_t cuError;                  // cuda error variable
  
  // device memory
  double *d_F;                          // vectors for store the force that suffer each particle
  
  /*----------------------------- function body -------------------------*/

  // allocate device memory for particle forces
  cuError = cudaMalloc((void **) &d_F, N*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  
  // call kernels to calculate particle forces and fix their velocities
  griddim = nc;
  blockdim = PAR_MOV_BLOCK_DIM;
  sh_mem_size = 2*2*nnx*sizeof(double)+2*sizeof(int); //adjust when fast_grid_to_particle kernel is done
  
  // electrons (evaluate forces and fix velocities)
  cudaGetLastError();
  fast_grid_to_particle<<<griddim, blockdim, sh_mem_size>>>(nn, -1, ds, d_e, d_e_bm, d_E, d_F);
  cu_sync_check(__FILE__, __LINE__);
  
  cudaGetLastError();
  fix_velocity<<<griddim, blockdim>>>(dt, me, d_e, d_e_bm, d_F);
  cu_sync_check(__FILE__, __LINE__);
  
  // ions (evaluate forces and fix velocities)
  cudaGetLastError();
  fast_grid_to_particle<<<griddim, blockdim, sh_mem_size>>>(nn, +1, ds, d_i, d_i_bm, d_E, d_F);
  cu_sync_check(__FILE__, __LINE__);
  
  cudaGetLastError();
  fix_velocity<<<griddim, blockdim>>>(dt, mi, d_i, d_i_bm, d_F);
  cu_sync_check(__FILE__, __LINE__);
  
  // free device and host memory
  cuError = cudaFree(d_F);
  cu_check(cuError, __FILE__, __LINE__);
  
  return;
}

/**********************************************************/

void load_particles(particle **d_i, int **d_i_bm, particle **d_e, int **d_e_bm, 
                    curandStatePhilox4_32_10_t **state)
{
  /*--------------------------- function variables -----------------------*/

  // host memory
  char filename[50];

  cudaError_t cuError;              // cuda error variable

  // device memory

  /*----------------------------- function body -------------------------*/

  // initialize curand philox states
  cuError = cudaMalloc ((void **) state, CURAND_BLOCK_DIM*sizeof(curandStatePhilox4_32_10_t));
  cu_check(cuError, __FILE__, __LINE__);
  cudaGetLastError();
  init_philox_state<<<1, CURAND_BLOCK_DIM>>>(*state);
  cu_sync_check(__FILE__, __LINE__);
  sprintf(filename, "./ions.dat");
  read_particle_file(filename, d_i, d_i_bm);
  sprintf(filename, "./electrons.dat");
  read_particle_file(filename, d_e, d_e_bm);
  
  return;
}

/**********************************************************/

void read_particle_file(string filename, particle **d_p, int **d_bm)
{
  /*--------------------------- function variables -----------------------*/

  // host memory
  const int ncy = init_ncy();   // number of cells in the y dimension
  const double ds = init_ds();  // space step
  particle *h_p;                // host vector for particles
  int *h_bm;                    // host vector for bookmarks
  int n = 0;                    // number of particles
  int bin;                      // bin
  
  ifstream myfile;              // file variables
  char line[150];

  cudaError_t cuError;          // cuda error variable
  
  // device memory

  /*----------------------------- function body -------------------------*/

  // get number of particles
  myfile.open(filename.c_str());
  if (myfile.is_open()) {
    myfile.getline(line, 150);
    while (!myfile.eof()) {
      myfile.getline(line, 150);
      n++;
    }
    n--;
  } else {
    cout << "Error. Can't open " << filename << " file" << endl;
  }
  myfile.close();

  // allocate host and device memory for particles and bookmarks
  h_p = (particle*) malloc(n*sizeof(particle));
  h_bm = (int*) malloc(2*ncy*sizeof(int));
  cuError = cudaMalloc ((void **) d_p, n*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_bm, 2*ncy*sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  
  // read particles from file and store in host memory
  myfile.open(filename.c_str());
  if (myfile.is_open()) {
    myfile.getline(line, 150);
    for (int i = 0; i<n; i++) {
      myfile.getline(line, 150);
      sscanf (line, " %le %le \n", &h_p[i].r, &h_p[i].v);
    }
  } else {
    cout << "Error. Can't open " << filename << " file" << endl;
  }
  myfile.close();

  // calculate bookmarks and store in host memory
  for (int i = 0; i < 2*ncy; i++) h_bm[i]=-1;
  for (int i = 0; i < n; ) {
    bin = int(h_p[i].y/ds);
    h_bm[bin*2] = i;
    while (bin == int(h_p[i].y/ds) && i < n) i++;
    h_bm[bin*2+1] = i-1;
  }

  // copy particle and bookmark vector from host to device memory
  cuError = cudaMemcpy (*d_p, h_p, n*sizeof(particle), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy (*d_bm, h_bm, 2*ncy*sizeof(int), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);

  // free host memory
  free(h_p);
  free(h_bm);
  
  return;
}

/**********************************************************/

void read_input_file(void *data, int data_size, int n)
{
  // function variables
  ifstream myfile;
  char line[80];

  // function body
  myfile.open("../input/input_data");
  if (myfile.is_open()) {
    myfile.getline(line, 80);
    for (int i = 0; i < n; i++) myfile.getline(line, 80);
    if (data_size == sizeof(int)) {
      sscanf (line, "%*s = %d;\n", (int*) data);
    } else if (data_size == sizeof(double)) {
      sscanf (line, "%*s = %lf;\n", (double*) data);
    }
  } else {
    cout << "Error. Input data file could not be opened" << endl;
    exit(1);
  }
  myfile.close();
  
  return;
}

/**********************************************************/

double init_qi(void) 
{
  // function variables
  
  // function body
  
  return 1.0;
}

/**********************************************************/

double init_qe(void) 
{
  // function variables
  
  // function body
  
  return -1.0;
}

/**********************************************************/

double init_mi(void) 
{
  // function variables
  static double gamma = 0.0;

  // function body
  
  if (gamma == 0.0) read_input_file((void*) &gamma, sizeof(gamma), 8);
  
  return gamma;
}

/**********************************************************/

double init_me(void) 
{
  // function variables
  
  // function body
  
  return 1.0;
}

/**********************************************************/

double init_kti(void) 
{ 
  // function variables
  static double beta = 0.0;
  
  // function body
  
  if (beta == 0.0) read_input_file((void*) &beta, sizeof(beta), 7);
  
  return beta;
}

/**********************************************************/

double init_kte(void) 
{
  // function variables
  
  // function body
  
  return 1.0;
}

/**********************************************************/

double init_phi_p(void) 
{
  // function variables
  double Te;
  static double phi_p = 0.0;
  
  // function body
  
  if (phi_p == 0.0) {
    read_input_file((void*) &Te, sizeof(Te), 6);
    read_input_file((void*) &phi_p, sizeof(phi_p), 9);
    phi_p *= CST_E/(CST_KB*Te);
  }
  
  return phi_p;
}

/**********************************************************/

double init_n(void) 
{
  // function variables
  const double Dl = init_Dl();
  static double n = 0.0;
  
  // function body
  
  if (n == 0.0) {
    read_input_file((void*) &n, sizeof(n), 5);
    n *= Dl*Dl*Dl;
  }
  
  return n;
}

/**********************************************************/

double init_L(void) 
{
  // function variables
  static double L = init_ds() * (double) init_nc();

  // function body
  
  return L;
}

/**********************************************************/

double init_ds(void) 
{
  // function variables
  static double ds = 0.0;
  
  // function body
  
  if (ds == 0.0) read_input_file((void*) &ds, sizeof(double), 11);
  
  return ds;
}

/**********************************************************/

double init_dt(void) 
{
  // function variables
  static double dt = 0.0;
  
  // function body
  
  if (dt == 0.0) read_input_file((void*) &dt, sizeof(double), 12);
  
  return dt;
}

/**********************************************************/

double init_epsilon0(void) 
{
  // function variables
  double Te;
  const double Dl = init_Dl();
  static double epsilon0 = 0.0;
  // function body
  
  if (epsilon0 == 0.0) {
    read_input_file((void*) &Te, sizeof(Te), 6);
    epsilon0 = CST_EPSILON;                         // SI units
    epsilon0 /= pow(Dl*sqrt(CST_ME/(CST_KB*Te)),2); // time units
    epsilon0 /= CST_E*CST_E;                        // charge units
    epsilon0 *= Dl*Dl*Dl;                           // length units
    epsilon0 *= CST_ME;                             // mass units
  }
  
  return epsilon0;
}

/**********************************************************/

int init_nc(void) 
{
  // function variables
  static int nc = 0;
  
  // function body
  
  if (nc == 0) read_input_file((void*) &nc, sizeof(nc), 10);
  
  return nc;
}

/**********************************************************/

int init_nn(void) 
{
  // function variables
  static int nn = init_nc()+1;
  
  // function body
  
  return nn;
}

/**********************************************************/

double init_dtin_i(void)
{
  // function variables
  const double mi = init_mi();
  const double kti = init_kti();
  const double n = init_n();
  const double ds = init_ds();
  static double dtin_i = sqrt(2.0*PI*mi/kti)/(n*ds*ds);
  
  // function body
  
  return dtin_i;
}

/**********************************************************/

double init_dtin_e(void)
{
  // function variables
  const double n = init_n();
  const double ds = init_ds();
  static double dtin_e = sqrt(2.0*PI)/(n*ds*ds);
  
  // function body
  
  return dtin_e;
}

/**********************************************************/

double init_Dl(void)
{
  // function variables
  double ne, Te;
  static double Dl = 0.0;
  
  // function body
  
  if (Dl == 0.0) {
    read_input_file((void*) &ne, sizeof(ne), 5);
    read_input_file((void*) &Te, sizeof(Te), 6);
    Dl = sqrt(CST_EPSILON*CST_KB*Te/(ne*CST_E*CST_E));
  }
  
  return Dl;
}

/**********************************************************/

int init_n_ini(void)
{
  // function variables
  static int n_ini = -1;
  
  // function body
  
  if (n_ini < 0) read_input_file((void*) &n_ini, sizeof(n_ini), 1);
  
  return n_ini;
}

/**********************************************************/

int init_n_prev(void)
{
  // function variables
  static int n_prev = -1;
  
  // function body
  
  if (n_prev < 0) read_input_file((void*) &n_prev, sizeof(n_prev), 2);
  
  return n_prev;
}

/**********************************************************/

int init_n_save(void)
{
  // function variables
  static int n_save = -1;
  
  // function body
  
  if (n_save < 0) read_input_file((void*) &n_save, sizeof(n_save), 3);
  
  return n_save;
}

/**********************************************************/

int init_n_fin(void)
{
  // function variables
  static int n_fin = -1;
  
  // function body
  
  if (n_fin < 0) read_input_file((void*) &n_fin, sizeof(n_fin), 4);
  
  return n_fin;
}

/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void init_philox_state(curandStatePhilox4_32_10_t *state)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  
  // kernel registers
  int tid = (int) threadIdx.x + (int) blockIdx.x * (int) blockDim.x;
  curandStatePhilox4_32_10_t local_state;
  
  /*--------------------------- kernel body ----------------------------*/
  
  // load states in local memory 
  local_state = state[tid];

  // initialize each thread state (seed, second seed, offset, pointer to state)
  curand_init (0, tid, 0, &local_state);

  // store initialized states in global memory
  state[tid] = local_state;

  return;
} 

/**********************************************************/
__global__ void create_particles_kernel(particle *g_p, int *g_p_bm, double kt, double m, int N, 
                                        int nc, double ds,  curandStatePhilox4_32_10_t *state)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ int sh_p_bm[2];
  
  // kernel registers
  particle reg_p;
  int ppc = (int) (N/nc);
  double sigma = sqrt(kt/m);
  int tid = (int) threadIdx.x + (int) blockIdx.x * (int) blockDim.x;
  int bdim = (int) blockDim.x;
  curandStatePhilox4_32_10_t local_state;
  double rnd;
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- load philox states from global memory
  local_state = state[tid];
  
  //---- initialize each bin 
  for (int i = 0; i < nc; i++) {
    // set bookmarks for the bin
    if (tid < 2) sh_p_bm[tid] = ((i+tid)*ppc)-tid;
    __syncthreads();
    // create particles of the bin
    for (int j = tid; j < ppc; j+=bdim)
    {
      rnd = curand_uniform_double(&local_state);
      reg_p.r = (double(i)+rnd)*ds;
      rnd = curand_normal_double(&local_state);
      reg_p.v = rnd*sigma;
      // store particles in global memory
      g_p[sh_p_bm[0]+j] = reg_p;
    }
    if (tid < 2) g_p_bm[i*2+tid] = sh_p_bm[tid];
    __syncthreads();
  }

  return;
}

/**********************************************************/

__global__ void fix_velocity(double dt, double m, particle *g_p, int *g_bm, double *g_F) 
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ int sh_bm[2];   // manually set up shared memory variables inside whole shared memory
  
  // kernel registers
  particle p;
  double F;
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory variables
  
  // load bin bookmarks from global memory
  if (threadIdx.x < 2)
  {
    sh_bm[threadIdx.x] = g_bm[blockIdx.x*2+threadIdx.x];
  }
  __syncthreads();
  
  //---- Process batches of particles
  
  for (int i = sh_bm[0]+threadIdx.x; i <= sh_bm[1]; i += blockDim.x)
  {
    // load particle data in registers
    p = g_p[i];
    F = g_F[i];
    
    // fix particle's velocity
    p.v -= 0.5*dt*F/m;
    
    // store particle data in global memory
    g_p[i] = p;
  }
  
  return;
}

/**********************************************************/


