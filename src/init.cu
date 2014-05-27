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

void init_sim(double **d_rho, double **d_phi, double **d_E, double **d_avg_rho, double **d_avg_phi, double **d_avg_E, 
              particle **d_e, int *num_e, particle **d_i, int *num_i, double *t, curandStatePhilox4_32_10_t **state)
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
    create_particles(d_i, num_i, d_e, num_e, state);

    // initialize mesh variables and their averaged counterparts
    initialize_mesh(d_rho, d_phi, d_E, *d_i, *num_i, *d_e, *num_e);
    initialize_avg_mesh(d_avg_rho, d_avg_phi, d_avg_E);

    // adjust velocities for leap-frog scheme
    adjust_leap_frog(*d_i, *num_i, *d_e, *num_e, *d_E);
    
    cout << "Simulation initialized with " << *num_e*2 << " particles." << endl << endl;
  } else if (n_ini > 0) {
    // adjust initial time
    *t = n_ini*dt;

    // read particle from file
    load_particles(d_i, num_i, d_e, num_e, state);
    
    // initialize mesh variables
    initialize_mesh(d_rho, d_phi, d_E, *d_i, *num_i, *d_e, *num_e);
    initialize_avg_mesh(d_avg_rho, d_avg_phi, d_avg_E);

    cout << "Simulation state loaded from time t = " << *t << endl;
  } else {
    cout << "Wrong input parameter (n_ini<0)" << endl;
    cout << "Stoppin simulation" << endl;
    exit(1);
  }
  
  return;
}

/**********************************************************/

void create_particles(particle **d_i, int *num_i, particle **d_e, int *num_e, curandStatePhilox4_32_10_t **state)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const double n = init_n();        // plasma density
  const double mi = init_mi();      // ion's mass
  const double me = init_me();      // electron's mass
  const double kti = init_kti();    // ion's thermal energy
  const double kte = init_kte();    // electron's thermal energy
  const double L = init_L();        // size of simulation
  const double ds = init_ds();      // spacial step

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
  *num_i = int(n*ds*ds*L);
  *num_e = *num_i;
  
  // allocate device memory for particle vectors
  cuError = cudaMalloc ((void **) d_i, (*num_i)*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_e, (*num_e)*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);
  
  // create particles (electrons)
  cudaGetLastError();
  create_particles_kernel<<<1, CURAND_BLOCK_DIM>>>(*d_e, *num_e, kte, me, L, *state);
  cu_sync_check(__FILE__, __LINE__);

  // create particles (ions)
  cudaGetLastError();
  create_particles_kernel<<<1, CURAND_BLOCK_DIM>>>(*d_i, *num_i, kti, mi, L, *state);
  cu_sync_check(__FILE__, __LINE__);

  return;
}

/**********************************************************/

void initialize_mesh(double **d_rho, double **d_phi, double **d_E, particle *d_i, int num_i, 
                     particle *d_e, int num_e)
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
  charge_deposition(*d_rho, d_e, num_e, d_i, num_i);
  
  // solve poisson equation
  poisson_solver(1.0e-4, *d_rho, *d_phi);
  
  // derive electric fields from potential
  field_solver(*d_phi, *d_E);
  
  return;
}

/**********************************************************/

void initialize_avg_mesh(double **d_avg_rho, double **d_avg_phi, double **d_avg_E)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const int nn = init_nn();   // number of nodes
  
  cudaError_t cuError;        // cuda error variable

  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  // allocate device memory for averaged mesh variables
  cuError = cudaMalloc ((void **) d_avg_rho, nn*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_avg_phi, nn*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_avg_E, nn*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  
  // initialize to zero averaged variables
  cuError = cudaMemset ((void *) *d_avg_rho, 0, nn*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemset ((void *) *d_avg_phi, 0, nn*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemset ((void *) *d_avg_E, 0, nn*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);

  return;
}

/**********************************************************/

void adjust_leap_frog(particle *d_i, int num_i, particle *d_e, int num_e, double *d_E)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const double mi = init_mi();          // ion's mass
  const double me = init_me();          // electron's mass
  const double ds = init_ds();          // spatial step size
  const double dt = init_dt();          // temporal step size
  const int nn = init_nn();             // number of nodes
  
  dim3 griddim, blockdim;               // kernel execution configurations
  size_t sh_mem_size;                   // shared memory size
  
  // device memory
  
  /*----------------------------- function body -------------------------*/

  // set grid and block dimensions for fix_velocity kernel
  griddim = 1;
  blockdim = PAR_MOV_BLOCK_DIM;

  // set shared memory size for fix_velocity kernel
  sh_mem_size = nn*sizeof(double);

  // fix velocities (electrons)
  cudaGetLastError();
  fix_velocity<<<griddim, blockdim, sh_mem_size>>>(-1.0, me, num_e, d_e, dt, ds, nn, d_E);
  cu_sync_check(__FILE__, __LINE__);
  
  // fix velocities (ions)
  cudaGetLastError();
  fix_velocity<<<griddim, blockdim, sh_mem_size>>>(1.0, mi, num_i, d_i, dt, ds, nn, d_E);
  cu_sync_check(__FILE__, __LINE__);
  
  return;
}

/**********************************************************/

void load_particles(particle **d_i, int *num_i, particle **d_e, int *num_e, curandStatePhilox4_32_10_t **state)
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

  // load particles
  sprintf(filename, "./ions.dat");
  read_particle_file(filename, d_i, num_i);
  sprintf(filename, "./electrons.dat");
  read_particle_file(filename, d_e, num_e);
  
  return;
}

/**********************************************************/

void read_particle_file(string filename, particle **d_p, int *num_p)
{
  /*--------------------------- function variables -----------------------*/

  // host memory
  particle *h_p;                // host vector for particles
  
  ifstream myfile;              // file variables
  char line[150];

  cudaError_t cuError;          // cuda error variable
  
  // device memory

  /*----------------------------- function body -------------------------*/

  // get number of particles (test if n is correctly evaluated)
  *num_p = 0;
  myfile.open(filename.c_str());
  if (myfile.is_open()) {
    myfile.getline(line, 150);
    while (!myfile.eof()) {
      myfile.getline(line, 150);
      *num_p += 1;
    }
  } else {
    cout << "Error. Can't open " << filename << " file" << endl;
  }
  myfile.close();

  // allocate host and device memory for particles
  h_p = (particle*) malloc(*num_p*sizeof(particle));
  cuError = cudaMalloc ((void **) d_p, *num_p*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);
  
  // read particles from file and store in host memory
  myfile.open(filename.c_str());
  if (myfile.is_open()) {
    myfile.getline(line, 150);
    for (int i = 0; i<*num_p; i++) {
      myfile.getline(line, 150);
      sscanf (line, " %le %le \n", &h_p[i].r, &h_p[i].v);
    }
  } else {
    cout << "Error. Can't open " << filename << " file" << endl;
  }
  myfile.close();

  // copy particle vector from host to device memory
  cuError = cudaMemcpy (*d_p, h_p, *num_p*sizeof(particle), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);

  // free host memory
  free(h_p);
  
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
  static double phi_p = 0.0;
  
  // function body
  
  if (phi_p == 0.0) read_input_file((void*) &phi_p, sizeof(phi_p), 9);
  
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
__global__ void create_particles_kernel(particle *g_p, int num_p, double kt, double m, double L, 
                                        curandStatePhilox4_32_10_t *state)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  
  // kernel registers
  particle reg_p;
  double sigma = sqrt(kt/m);
  int tid = (int) threadIdx.x + (int) blockIdx.x * (int) blockDim.x;
  int bdim = (int) blockDim.x;
  curandStatePhilox4_32_10_t local_state;
  double rnd;
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- load philox states from global memory
  local_state = state[tid];
  
  //---- create particles 
  for (int i = tid; i < num_p; i+=bdim) {
    rnd = curand_uniform_double(&local_state);
    reg_p.r = rnd*L;
    rnd = curand_normal_double(&local_state);
    reg_p.v = rnd*sigma;
    // store particles in global memory
    g_p[i] = reg_p;
  }
  __syncthreads();

  //---- store philox states in global memory
  state[tid] = local_state;

  return;
}

/**********************************************************/

__global__ void fix_velocity(double q, double m, int num_p, particle *g_p, double dt, double ds, int nn, double *g_E)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  double *sh_E = (double *) sh_mem;
  
  // kernel registers
  int tid = (int) threadIdx.x;  // thread Id
  int bdim = (int) blockDim.x;  // block dimension
  particle reg_p;               // register particles
  int ic;                       // cell index
  double dist;                  // distance from particle to nearest down vertex (normalized to ds)
  double F;                     // force suffered for each register particle
  
  /*--------------------------- kernel body ----------------------------*/
 
  //---- load electric field in shared memory
  for (int i = tid; i<nn; i+=bdim) {
    sh_E[i] = g_E[i];
  }
  __syncthreads();

  //---- load and analize and fix particles
  for (int i = tid; i<num_p; i += bdim) {
    // load particles from global to shared memory
    reg_p = g_p[i];

    // analize particles
    ic = __double2int_rd(reg_p.r/ds);

    // evaluate particle forces
    dist = fabs(reg_p.r-ic*ds)/ds;
    F = q*(sh_E[ic]*(1-dist)+sh_E[ic+1]*dist);

    // fix particle velocities
    reg_p.v -= 0.5*dt*F/m;

    // store back particles in global memory
    g_p[i] = reg_p;
  }

  return;
}

/**********************************************************/


