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
              particle **d_e, int *num_e, particle **d_i, int *num_i, double **d_avg_ddf_e, double **d_avg_vdf_e, 
              double **d_avg_ddf_i, double **d_avg_vdf_i, double *t, curandStatePhilox4_32_10_t **state)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const double dt = init_dt();
  const int n_ini = init_n_ini();

  // device memory
  
  /*----------------------------- function body -------------------------*/

  cout << "n = " << init_n() << endl;
  // check if simulation start from initial condition or saved state
  if (n_ini == 0) {
    // adjust initial time
    *t = 0.;

    // create particles
    create_particles(d_i, num_i, d_e, num_e, state);

    // initialize mesh variables and their averaged counterparts
    initialize_mesh(d_rho, d_phi, d_E, *d_i, *num_i, *d_e, *num_e);

    // adjust velocities for leap-frog scheme
    adjust_leap_frog(*d_i, *num_i, *d_e, *num_e, *d_E);

    //initialize diagnostic variables
    initialize_avg_mesh(d_avg_rho, d_avg_phi, d_avg_E);
    initialize_avg_df(d_avg_ddf_e, d_avg_vdf_e, d_avg_ddf_i, d_avg_vdf_i);
    
    cout << "Simulation initialized with " << *num_e*2 << " particles." << endl << endl;
  } else if (n_ini > 0) {
    // adjust initial time
    *t = n_ini*dt;

    // read particle from file
    load_particles(d_i, num_i, d_e, num_e, state);
    
    // initialize mesh variables
    initialize_mesh(d_rho, d_phi, d_E, *d_i, *num_i, *d_e, *num_e);
    
    //initialize diagnostic variables
    initialize_avg_mesh(d_avg_rho, d_avg_phi, d_avg_E);
    initialize_avg_df(d_avg_ddf_e, d_avg_vdf_e, d_avg_ddf_i, d_avg_vdf_i);

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
  //*num_i = int(n*ds*ds*L);
  *num_i = 0;
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
  const double phi_s = -0.5*init_mi()*init_vd_i()*init_vd_i();
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
    h_phi[i] = phi_p + double(i)*(phi_s-phi_p)/double(nc);
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

void initialize_avg_df(double **d_avg_ddf_e, double **d_avg_vdf_e, double **d_avg_ddf_i, double **d_avg_vdf_i)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const int n_bin_ddf = init_n_bin_ddf();   // number of bins for density distribution function
  const int n_bin_vdf = init_n_bin_vdf();   // number of bins for velocity distribution function
  const int n_vdf = init_n_vdf();           // number of velocity distribution functions to calculate
  
  cudaError_t cuError;        // cuda error variable

  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  // allocate device memory for averaged distribution functions
  cuError = cudaMalloc ((void **) d_avg_ddf_e, n_bin_ddf*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_avg_ddf_i, n_bin_ddf*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_avg_vdf_e, n_bin_vdf*n_vdf*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_avg_vdf_i, n_bin_vdf*n_vdf*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  
  // initialize to zero averaged distribution functions
  cuError = cudaMemset ((void *) *d_avg_ddf_e, 0, n_bin_ddf*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemset ((void *) *d_avg_ddf_i, 0, n_bin_ddf*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemset ((void *) *d_avg_vdf_e, 0, n_bin_vdf*n_vdf*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemset ((void *) *d_avg_vdf_i, 0, n_bin_vdf*n_vdf*sizeof(double));
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

template <typename type> void read_input_file(type *data, int n)
{
  // function variables
  ifstream myfile;
  char line[80];

  // function body
  myfile.open("../input/input_data");
  if (myfile.is_open()) {
    for (int i = 0; i < n; i++) myfile.getline(line, 80);
    if (sizeof(type) == sizeof(int)) {
      sscanf (line, "%*s = %d;\n", (int*) data);
    } else if (sizeof(type) == sizeof(double)) {
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
  
  if (gamma == 0.0) read_input_file(&gamma, 12);
  
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
  
  if (beta == 0.0) read_input_file(&beta, 9);
  
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

double init_vd_i(void) 
{ 
  // function variables
  static double vd_i = -10.0;
  
  // function body
  
  if (vd_i == -10.0) read_input_file(&vd_i, 11);
  
  return vd_i;
}

/**********************************************************/

double init_vd_e(void) 
{ 
  // function variables
  static double vd_e = -10.0;
  
  // function body
  
  if (vd_e == -10.0) read_input_file(&vd_e, 10);
  
  return vd_e;
}

/**********************************************************/

double init_phi_p(void) 
{
  // function variables
  static double phi_p = 0.0;
  
  // function body
  
  if (phi_p == 0.0) read_input_file(&phi_p, 14);
  
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
    read_input_file(&n, 7);
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
  
  if (ds == 0.0) read_input_file(&ds, 17);
  
  return ds;
}

/**********************************************************/

double init_dt(void) 
{
  // function variables
  static double dt = 0.0;
  
  // function body
  
  if (dt == 0.0) read_input_file(&dt, 18);
  
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
    read_input_file(&Te, 8);
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
  
  if (nc == 0) read_input_file(&nc, 16);
  
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
  const double n = init_n();
  const double ds = init_ds();
  const double mi = init_mi();
  const double kti = init_kti();
  const double vd_i = init_vd_i();
  const double phi_s = -0.5*init_mi()*init_vd_i()*init_vd_i();
  const double phi_p = init_phi_p();
  static double dtin_i = 0.0;
  
  // function body
  
  if (dtin_i == 0.0) {
    dtin_i = n*sqrt(kti/(2.0*PI*mi))*exp(-0.5*mi*vd_i*vd_i/kti);  // thermal component of input flux
    dtin_i += 0.5*n*vd_i*(1.0+erf(sqrt(0.5*mi/kti)*vd_i));        // drift component of input flux
    dtin_i *= exp(phi_s)*0.5*(1.0+erf(sqrt(phi_s-phi_p)));        // correction on density at sheath edge

    dtin_i *= ds*ds;      // number of particles that enter the simulation per unit of time
    dtin_i = 1.0/dtin_i;  // time between consecutive particles injection
  }

  return dtin_i;
}

/**********************************************************/

double init_dtin_e(void)
{
  // function variables
  const double n = init_n();
  const double ds = init_ds();
  const double me = init_me();
  const double kte = init_kte();
  const double vd_e = init_vd_e();
  const double phi_s = -0.5*init_mi()*init_vd_i()*init_vd_i();
  const double phi_p = init_phi_p();
  static double dtin_e = 0.0;
  
  // function body
  
  if (dtin_e == 0.0) {
    dtin_e = n*sqrt(kte/(2.0*PI*me))*exp(-0.5*me*vd_e*vd_e/kte);  // thermal component of input flux
    dtin_e += 0.5*n*vd_e*(1.0+erf(sqrt(0.5*me/kte)*vd_e));        // drift component of input flux
    dtin_e *= exp(phi_s)*0.5*(1.0+erf(sqrt(phi_s-phi_p)));        // correction on density at sheath edge

    dtin_e *= ds*ds;      // number of particles that enter the simulation per unit of time
    dtin_e = 1.0/dtin_e;  // time between consecutive particles injection
  }

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
    read_input_file(&ne, 7);
    read_input_file(&Te, 8);
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
  
  if (n_ini < 0) read_input_file(&n_ini, 2);
  
  return n_ini;
}

/**********************************************************/

int init_n_prev(void)
{
  // function variables
  static int n_prev = -1;
  
  // function body
  
  if (n_prev < 0) read_input_file(&n_prev, 3);
  
  return n_prev;
}

/**********************************************************/

int init_n_save(void)
{
  // function variables
  static int n_save = -1;
  
  // function body
  
  if (n_save < 0) read_input_file(&n_save, 4);
  
  return n_save;
}

/**********************************************************/

int init_n_fin(void)
{
  // function variables
  static int n_fin = -1;
  
  // function body
  
  if (n_fin < 0) read_input_file(&n_fin, 5);
  
  return n_fin;
}

/**********************************************************/

int init_n_bin_ddf(void)
{
  // function variables
  static int n_bin_ddf = -1;
  
  // function body
  
  if (n_bin_ddf < 0) read_input_file(&n_bin_ddf, 20);
  
  return n_bin_ddf;
}

/**********************************************************/

int init_n_bin_vdf(void)
{
  // function variables
  static int n_bin_vdf = -1;
  
  // function body
  
  if (n_bin_vdf < 0) read_input_file(&n_bin_vdf, 22);
  
  return n_bin_vdf;
}

/**********************************************************/

int init_n_vdf(void)
{
  // function variables
  static int n_vdf = -1;
  
  // function body
  
  if (n_vdf < 0) read_input_file(&n_vdf, 21);
  
  return n_vdf;
}

/**********************************************************/

double init_vth_e(void)
{
  // function variables
  static double kte = init_kte();         // thermal energy of electrons
  static double me = init_me();           // electron mass
  static double vth_e = sqrt(2*kte/me);   // thermal velocity of electrons
  
  // function body
  
  return vth_e;
}

/**********************************************************/

double init_vth_i(void)
{
  // function variables
  static double kti = init_kti();         // thermal energy of ions
  static double mi = init_mi();           // ion mass
  static double vth_i = sqrt(2*kti/mi);   // thermal velocity of ions
  
  // function body
  
  return vth_i;
}

/**********************************************************/

double init_v_max_e(void)
{
  // function variables
  static double v_max_e = 0;   // max velocity to consider in velocity histograms
  
  // function body

  if (v_max_e == 0) read_input_file(&v_max_e, 23);
  
  return v_max_e;
}

/**********************************************************/

double init_v_min_e(void)
{
  // function variables
  static double v_min_e = 0;   // min velocity to consider in velocity histograms
  
  // function body

  if (v_min_e == 0) read_input_file(&v_min_e, 24);
  
  return v_min_e;
}

/**********************************************************/

double init_v_max_i(void)
{
  // function variables
  static double v_max_i = 0;   // max velocity to consider in velocity histograms
  
  // function body

  if (v_max_i == 0) read_input_file(&v_max_i, 25);
  
  return v_max_i;
}

/**********************************************************/

double init_v_min_i(void)
{
  // function variables
  static double v_min_i = 0;   // min velocity to consider in velocity histograms
  
  // function body

  if (v_min_i == 0) read_input_file(&v_min_i, 26);
  
  return v_min_i;
}

/**********************************************************/

bool calibration_is_on(void)
{
  // function variables
  static int calibration_int = -1;
  
  // function body
  
  if (calibration_int < 0) {
    read_input_file(&calibration_int, 28);
    if (calibration_int != 0 && calibration_int != 1) {
      cout << "Found error in input_data file. Wrong ion_current_calibration!\nStoping simulation.\n" << endl;
      exit(1);
    }
  }
  
  if (calibration_int == 1) return true;
  else return false;
}

/**********************************************************/

bool floating_potential_is_on(void)
{
  // function variables
  static int floating_potential_int = -1;
  
  // function body
  
  if (floating_potential_int < 0) {
    read_input_file(&floating_potential_int , 30);
    if (floating_potential_int != 0 && floating_potential_int != 1) {
      cout << "Found error in input_data file. Wrong floating_potential!\nStoping simulation.\n" << endl;
      exit(1);
    }
  }
  
  if (floating_potential_int == 1) return true;
  else return false;
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

