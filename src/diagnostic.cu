/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


/****************************** HEADERS ******************************/

#include "diagnostic.h"

/********************* HOST FUNCTION DEFINITIONS *********************/

void avg_mesh(double *d_foo, double *d_avg_foo, int *count)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const int nn = init_nn();          // number of nodes
  static const int n_save = init_n_save();  // number of iterations to average
  
  dim3 griddim, blockdim;
  cudaError_t cuError;

  // device memory
  
  /*----------------------------- function body -------------------------*/

  // check if restart of avg_foo is needed
  if (*count == n_save) {
    //reset count
    *count = 0;

    //reset avg_foo
    cuError = cudaMemset ((void *) d_avg_foo, 0, nn*sizeof(double));
    cu_check(cuError, __FILE__, __LINE__);
  }

  // set dimensions of grid of blocks and block of threads for kernels
  blockdim = AVG_MESH_BLOCK_DIM;
  griddim = int(nn/AVG_MESH_BLOCK_DIM)+1;

  // call to mesh_sum kernel
  cudaGetLastError();
  mesh_sum<<<griddim, blockdim>>>(d_foo, d_avg_foo, nn);
  cu_sync_check(__FILE__, __LINE__);

  // actualize count
  *count += 1;

  // normalize average if reached desired number of iterations
  if (*count == n_save ) {
    cudaGetLastError();
    mesh_norm<<<griddim, blockdim>>>(d_avg_foo, (double) n_save, nn);
    cu_sync_check(__FILE__, __LINE__); 
  }

  return;
}

/**********************************************************/

void particles_snapshot(particle *d_p, int num_p, string filename)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  particle *h_p;
  FILE *pFile;
  cudaError_t cuError;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // allocate host memory for particle vector
  h_p = (particle *) malloc(num_p*sizeof(particle));
  
  // copy particle vector from device to host
  cuError = cudaMemcpy (h_p, d_p, num_p*sizeof(particle), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  
  // save snapshot to file
  filename.append(".dat");
  pFile = fopen(filename.c_str(), "w");
  for (int i = 0; i < num_p; i++)
  {
    fprintf(pFile, " %.17e %.17e \n", h_p[i].r, h_p[i].v);
  }
  fclose(pFile);
  
  // free host memory
  free(h_p);
  
  return;
}

/**********************************************************/

void mesh_snapshot(double *d_m, string filename) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory 
  static const int nn = init_nn();
  double *h_m;
  FILE *pFile;
  cudaError_t cuError;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // allocate host memory for mesh vector
  h_m = (double *) malloc(nn*sizeof(double));
  
  // copy particle vector from device to host
  cuError = cudaMemcpy (h_m, d_m, nn*sizeof(double), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  
  // save snapshot to file
  filename.append(".dat");
  pFile = fopen(filename.c_str(), "w");
  for (int i = 0; i < nn; i++) 
  {
    fprintf(pFile, " %d %.17e \n", i, h_m[i]);
  }
  fclose(pFile);
  
  // free host memory
  free(h_m);
  
  return;
}

/**********************************************************/

void save_bins(particle *d_p, int num_p, string filename)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double ds = init_ds();      // spacial step
  particle *h_p;
  FILE *pFile;
  cudaError_t cuError;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // allocate host memory for particle vector
  h_p = (particle *) malloc(num_p*sizeof(particle));
  
  // copy particle vector from device to host
  cuError = cudaMemcpy (h_p, d_p, num_p*sizeof(particle), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  
  // save bins to file
  filename.insert(0, "../output/");
  filename.append(".dat");
  pFile = fopen(filename.c_str(), "w");
  for (int i = 0; i < num_p; i++) 
  {
    fprintf(pFile, " %d %d \n", i, int(h_p[i].r/ds));
  }
  fclose(pFile);

  //free host memory for particle vector
  free(h_p);
  
  return;
}

/**********************************************************/

double particle_energy(double *d_phi,  particle *d_p, double m, double q, int num_p)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const int nn = init_nn();        // number of nodes
  static const double ds = init_ds();     // spacial step
  double *h_partial_U;                    // partial energy of each block
  double h_U = 0.0;                       // total energy of particle system

  dim3 griddim, blockdim;
  size_t sh_mem_size;
  cudaError_t cuError;
  
  // device memory
  double *d_partial_U;
  
  /*----------------------------- function body -------------------------*/
  
  // set execution configuration of the kernel that evaluates energy
  blockdim = ENERGY_BLOCK_DIM;
  griddim = int(num_p/ENERGY_BLOCK_DIM)+1;

  // allocate host and device memory for block's energy
  cuError = cudaMalloc ((void **) &d_partial_U, griddim.x*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  h_partial_U = (double *) malloc(griddim.x*sizeof(double));

  // define size of shared memory for energy_kernel
  sh_mem_size = (ENERGY_BLOCK_DIM+nn)*sizeof(double);

  // launch kernel to evaluate energy of the whole system
  cudaGetLastError();
  energy_kernel<<<griddim, blockdim, sh_mem_size>>>(d_partial_U, d_phi, nn, ds, d_p, m, q, num_p); 
  cu_sync_check(__FILE__, __LINE__);

  // copy sistem energy from device to host
  cuError = cudaMemcpy (h_partial_U, d_partial_U, griddim.x*sizeof(double), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);

  // reduction of block's energy
  for (int i = 0; i<griddim.x; i++) h_U += h_partial_U[i];

  //free host and device memory for block's energy
  cuError = cudaFree(d_partial_U);
  cu_check(cuError, __FILE__, __LINE__);
  free(h_partial_U);
  
  return h_U;
}

/**********************************************************/

void log(double t, int num_e, int num_i, double U_e, double U_i, double dtin_i)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  string filename = "../output/log.dat";
  FILE *pFile;
  
  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  // save log to file
  pFile = fopen(filename.c_str(), "a");
  if (pFile == NULL) perror ("Error opening log file file");
  else fprintf(pFile, " %.17e %d %d %.17e %.17e %.17e \n", t, num_e, num_i, U_e, U_i, dtin_i);
  fclose(pFile);

  return;
}

/**********************************************************/

void calibrate_dtin_i(double dtin_i, bool should_increase)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static double factor = 0.1;
  static bool increase_last = true;
  
  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  if (should_increase) dtin_i *= (1.0+factor);
  else dtin_i *= (1.0-factor);

  if (increase_last != should_increase) {
    factor *= 0.9;
    increase_last = should_increase;
  }

  return;
}

/**********************************************************/

double calculate_vd_i(double dtin_i)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double n = init_n();             // plasma density
  static const double ds = init_ds();           // spatial step
  
  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  return 1.0/(n*dtin_i*ds*ds);
}

/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void mesh_sum(double *g_foo, double *g_avg_foo, int nn)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  
  // kernel registers
  double reg_foo, reg_avg_foo;

  int tid = (int) (threadIdx.x + blockIdx.x * blockDim.x);
  
  /*--------------------------- kernel body ----------------------------*/

  // load data from global memory to registers
  if (tid < nn) {
    reg_foo = g_foo[tid];
    reg_avg_foo = g_avg_foo[tid];
  }
  __syncthreads();

  // add foo to avg foo
  if (tid < nn) {
    reg_avg_foo += reg_foo;
  }
  __syncthreads();

  // store data y global memory
  if (tid < nn) {
    g_avg_foo[tid] = reg_avg_foo ;
  }
  
  return;
}

/**********************************************************/

__global__ void mesh_norm(double *g_avg_foo, double norm_cst, int nn)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  
  // kernel registers
  double reg_avg_foo;

  int tid = (int) (threadIdx.x + blockIdx.x * blockDim.x);
  
  /*--------------------------- kernel body ----------------------------*/

  // load data from global memory to registers
  if (tid < nn) reg_avg_foo = g_avg_foo[tid];

  // normalize avg foo
  if (tid < nn) reg_avg_foo /= norm_cst;
  __syncthreads();

  // store data y global memory
  if (tid < nn) g_avg_foo[tid] = reg_avg_foo ;
  
  return;
}

/**********************************************************/

__global__ void energy_kernel(double *g_U, double *g_phi, int nn, double ds,
                              particle *g_p, double m, double q, int num_p)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  double *sh_phi = (double *) sh_mem;   // mesh potential
  double *sh_U = &sh_phi[nn];           // acumulation of energy in each block
  
  // kernel registers
  int tid = (int) (threadIdx.x + blockIdx.x * blockDim.x);
  int tidx = (int) threadIdx.x;
  int bid = (int) blockIdx.x;
  int bdim = (int) blockDim.x;
  
  int ic;
  double dist;
  
  particle reg_p;
  
  /*--------------------------- kernel body ----------------------------*/

  // load potential data from global to shared memory
  for (int i = tidx; i < nn; i += bdim) {
    sh_phi[i] = g_phi[i];
  }

  // initialize energy acumulation's variables
  sh_U[tidx] = 0.0;
  __syncthreads();

  // analize energy of each particle
  if (tid < num_p) {
    // load particle in registers
    reg_p = g_p[tid];
    // calculate what cell the particle is in
    ic = __double2int_rd(reg_p.r/ds);
    // calculate distances from particle to down vertex of the cell
    dist = fabs(__int2double_rn(ic)*ds-reg_p.r)/ds;
    // evaluate potential energy of particle
    sh_U[tidx] = (sh_phi[ic]*(1.0-dist)+sh_phi[ic+1]*dist)*q;
    // evaluate kinetic energy of particle
    sh_U[tidx] += 0.5*m*reg_p.v*reg_p.v;
  }
  __syncthreads();

  // reduction for obtaining total energy in current block
  for (int stride = 1; stride < bdim; stride <<= 1) {
    if ((tidx%(stride<<2) == 0) && (tidx+stride < bdim)) {
      sh_U[tidx] += sh_U[tidx+stride*2];
    }
    __syncthreads();
  }

  // store total energy of current block in global memory
  if (tidx == 0) g_U[bid] = sh_U[0];
  
  return;
}

/**********************************************************/
