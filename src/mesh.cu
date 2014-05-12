/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


/****************************** HEADERS ******************************/

#include "mesh.h"

/********************* HOST FUNCTION DEFINITIONS *********************/

void charge_deposition(double *d_rho, particle *d_e, int num_e, particle *d_i, int num_i) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double ds = init_ds();   // spatial step
  static const int nn = init_nn();      // number of nodes
  
  dim3 griddim, blockdim;
  size_t sh_mem_size;
  cudaError_t cuError;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // initialize device memory to zeros
  cuError = cudaMemset(d_rho, 0, nn*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  
  // set size of shared memory for particle_to_grid kernel
  sh_mem_size = nn*sizeof(double);

  // set dimensions of grid of blocks and block of threads for particle_to_grid kernel (electrons)
  blockdim = CHARGE_DEP_BLOCK_DIM;
  griddim = int(num_e/CHARGE_DEP_BLOCK_DIM)+1;
  
  // call to particle_to_grid kernel (electrons)
  cudaGetLastError();
  particle_to_grid<<<griddim, blockdim, sh_mem_size>>>(ds, nn, d_rho, d_e, num_e, -1.0);
  cu_sync_check(__FILE__, __LINE__);

  // set dimensions of grid of blocks and block of threads for particle_to_grid kernel (ions)
  blockdim = CHARGE_DEP_BLOCK_DIM;
  griddim = int(num_i/CHARGE_DEP_BLOCK_DIM)+1;
  
  // call to particle_to_grid kernel (ions)
  cudaGetLastError();
  particle_to_grid<<<griddim, blockdim, sh_mem_size>>>(ds, nn, d_rho, d_i, num_i, 1.0);
  cu_sync_check(__FILE__, __LINE__);
  
  return;
}

/**********************************************************/

void poisson_solver(double max_error, double *d_rho, double *d_phi) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double ds = init_ds();               // spatial step
  static const int nn = init_nn();                  // number of nodes
  static const double epsilon0 = init_epsilon0();   // electric permitivity of free space
  
  double *h_error;
  double t_error = max_error*10;
  int min_iteration = 2*nn;
  
  dim3 blockdim, griddim;
  size_t sh_mem_size;
  cudaError_t cuError;

  // device memory
  double *d_error;
  
  /*----------------------------- function body -------------------------*/
  
  // set dimensions of grid of blocks and blocks of threads for jacobi kernel
  blockdim = JACOBI_BLOCK_DIM;
  griddim = (int) ((nn-2)/JACOBI_BLOCK_DIM) + 1;
  
  // define size of shared memory for jacobi_iteration kernel
  sh_mem_size = (2*JACOBI_BLOCK_DIM+2)*sizeof(double);
  
  // allocate host and device memory for vector of errors
  cuError = cudaMalloc((void **) &d_error, griddim.x*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  h_error = (double*) malloc(griddim.x*sizeof(double));

  // execute jacobi iterations until solved
  while(min_iteration>=0 || t_error>=max_error) {
    // launch kernel for performing one jacobi iteration
    cudaGetLastError();
    jacobi_iteration<<<griddim, blockdim, sh_mem_size>>>(nn, ds, epsilon0, d_rho, d_phi, d_error);
    cu_sync_check(__FILE__, __LINE__);
    
    // copy error vector from  device to host memory
    cuError = cudaMemcpy(h_error, d_error, griddim.x*sizeof(double), cudaMemcpyDeviceToHost);
    cu_check(cuError, __FILE__, __LINE__);

    // evaluate max error of the iteration
    t_error = 0;
    for (int i = 0; i<griddim.x; i++)
    {
      if (h_error[i] > t_error) t_error = h_error[i];
    }
    
    // actualize counter
    min_iteration--;
  }

  // free device memory
  cudaFree(d_error);
  free(h_error);

  return;
}

/**********************************************************/

void field_solver(double *d_phi, double *d_E) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double ds = init_ds();   // spatial step
  static const int nn = init_nn();      // number of nodes   
  dim3 blockdim, griddim;
  
  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  // set dimensions of grid of blocks and blocks of threads for jacobi kernel
  blockdim = JACOBI_BLOCK_DIM;
  griddim = (int) ((nn-2)/JACOBI_BLOCK_DIM) + 1;
  
  // launch kernel for performing the derivation of the potential to obtain the electric field
  cudaGetLastError();
  field_derivation<<<griddim, blockdim>>>(nn, ds, d_phi, d_E);
  cu_sync_check(__FILE__, __LINE__);

  return;
}

/**********************************************************/



/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void particle_to_grid(double ds, int nn, double *g_rho, particle *g_p, int num_p, double q)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  double *sh_partial_rho = (double *) sh_mem;   // partial rho of each bin
  
  // kernel registers
  int tidx = (int) threadIdx.x;
  int tid = (int) (threadIdx.x + blockIdx.x*blockDim.x);
  int bdim = (int) blockDim.x;
  int ic;                       // cell index of each particle
  particle reg_p;               // register copy of particle analized
  double dist;                  // distance to down vertex of the cell
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory variables

  // initialize charge density in shared memory to 0.0
  for (int i = tidx; i < nn; i+=bdim) {
    sh_partial_rho[i] = 0.0;
  }
  __syncthreads();
  
  //--- deposition of charge
  
  if (tid < num_p) {
    // load particle in registers
    reg_p = g_p[tid];
    // calculate what cell the particle is in
    ic = __double2int_rd(reg_p.r/ds);
    if (ic >= nn-1) printf("error 2 on tid = %d, ic = %d, p.r = %f\n", tidx, ic, reg_p.r);
    // calculate distances from particle to down vertex of the cell
    dist = fabs(__int2double_rn(ic)*ds-reg_p.r)/ds;
    // acumulate charge in partial rho
    atomicAdd(&sh_partial_rho[ic], q*(1.0-dist));    //down vertex
    atomicAdd(&sh_partial_rho[ic+1], q*dist);        //upper vertex
  }
  __syncthreads();

  //---- volume correction (shared memory)
  
  for (int i = tidx+1; i < nn-1; i+=bdim) {
    sh_partial_rho[i] /= ds*ds*ds;
  }
  if (tidx == 0) {
    sh_partial_rho[0] /= 0.5*ds*ds*ds;
    sh_partial_rho[nn-1] /= 0.5*ds*ds*ds;
  }
  __syncthreads();

  //---- charge acumulation in global memory
  
  for (int i = tidx; i < nn; i+=bdim) {
    atomicAdd(&g_rho[i], sh_partial_rho[i]);
  }
  __syncthreads();

  return;
}

/**********************************************************/

__global__ void jacobi_iteration (int nn, double ds, double epsilon0, double *g_rho, double *g_phi, double *g_error)
{
  /*----------------------------- function body -------------------------*/
  
  // shared memory
  double *sh_old_phi= (double *) sh_mem;                           //
  double *sh_error = (double *) &sh_old_phi[JACOBI_BLOCK_DIM+2];   // manually set up shared memory
  
  // registers
  double new_phi, dummy_rho;
  int tid = (int) threadIdx.x;
  int sh_tid = (int) threadIdx.x + 1;
  int g_tid = (int) (threadIdx.x + blockDim.x * blockIdx.x) + 1;
  int bdim = (int) blockDim.x;
  int bid = (int) blockIdx.x;
  int gdim = (int) gridDim.x;
  
  /*------------------------------ kernel body --------------------------*/
  
  // load phi data from global to shared memory
  if (g_tid < nn - 1) sh_old_phi[sh_tid] = g_phi[g_tid];

  // load comunication zones
  if (bid < gdim-1) {
    if (sh_tid == 1) sh_old_phi[sh_tid-1] = g_phi[g_tid-1];
    if (sh_tid == bdim) sh_old_phi[sh_tid+1] = g_phi[g_tid+1];
  } else {
    if (sh_tid == 1) sh_old_phi[sh_tid-1] = g_phi[g_tid-1];
    if (g_tid == nn-2) sh_old_phi[sh_tid+1] = g_phi[g_tid+1];
  }
  __syncthreads();
  
  // load charge density data into registers
  if (g_tid < nn - 1) dummy_rho = ds*ds*g_rho[g_tid]/epsilon0;
  __syncthreads();
  
  // actualize interior mesh points
  if (g_tid < nn - 1) new_phi = 0.5*(dummy_rho + sh_old_phi[sh_tid-1] + sh_old_phi[sh_tid+1]);
  __syncthreads();
  
  // store new values of phi in global memory
  if (g_tid < nn - 1) g_phi[g_tid] = new_phi;
  __syncthreads();

  // evaluate local errors
  if (g_tid < nn - 1) sh_error[tid] = fabs(new_phi-sh_old_phi[sh_tid]);
  __syncthreads();

  // reduction for obtaining maximum error in current block
  for (int stride = 1; stride < bdim; stride <<= 1) {
    if ((tid%(stride*2) == 0) && (tid+stride < bdim) && (g_tid+stride < nn-1)) {
      if (sh_error[tid]<sh_error[tid+stride]) sh_error[tid] = sh_error[tid+stride];
    }
    __syncthreads();
  }
  
  // store maximun error in global memory
  if (tid == 0) g_error[bid] = sh_error[tid];
  
  return;
}

/**********************************************************/

__global__ void field_derivation (int nn, double ds, double *g_phi, double *g_E)
{
  /*---------------------------- kernel variables ------------------------*/
  // shared memory
  __shared__ double sh_phi[JACOBI_BLOCK_DIM+2];
  
  // registers
  double reg_E;
  int sh_tid = (int) threadIdx.x + 1;
  int g_tid = (int) (threadIdx.x + blockDim.x * blockIdx.x) + 1;
  int bdim = (int) blockDim.x;
  int bid = (int) blockIdx.x;
  int gdim = (int) gridDim.x;
  
  /*------------------------------ kernel body --------------------------*/
  
  // load phi data from global to shared memory
  if (g_tid < nn - 1) {
    sh_phi[sh_tid] = g_phi[g_tid];
  }
  // load comunication zones
  if (bid < gdim-1) {
    if (sh_tid == 1) sh_phi[sh_tid-1] = g_phi[g_tid-1];
    if (sh_tid == bdim) sh_phi[sh_tid+1] = g_phi[g_tid+1];
  } else {
    if (sh_tid == 1) sh_phi[sh_tid-1] = g_phi[g_tid-1];
    if (sh_tid == nn-2) sh_phi[sh_tid+1] = g_phi[g_tid+1];
  }
  __syncthreads();
  
  // calculate electric fields in interior points
  if (g_tid < nn - 1) {
    reg_E = (sh_phi[sh_tid-1]-sh_phi[sh_tid+1])/(2.0*ds);
  } 
  __syncthreads();

  // store electric fields of interior points in global memory
  if (g_tid < nn - 1) g_E[g_tid] = reg_E; 
  
  // calculate electric fields at proble and plasma
  if (g_tid == nn-2) {
    reg_E = (sh_phi[sh_tid]-sh_phi[sh_tid+1])/ds;
    g_E[g_tid+1] = reg_E;
  } else if (g_tid == 1) {
    reg_E = (sh_phi[sh_tid-1]-sh_phi[sh_tid])/ds;
    g_E[g_tid-1] = reg_E;
  }

  return;
}

/**********************************************************/



/******************** DEVICE FUNCTION DEFINITIONS ********************/


/**********************************************************/
