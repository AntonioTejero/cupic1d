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
  
  double h_error = max_error*10;
  int min_iteration = 2*nn;
  
  dim3 blockdim, griddim;
  size_t sh_mem_size;
  cudaError_t cuError;

  // device memory
  double *d_error;
  
  /*----------------------------- function body -------------------------*/
  
  // set dimensions of grid of blocks and blocks of threads for jacobi kernel
  blockdim = nn;
  griddim = 1;
  
  // define size of shared memory for jacobi_iteration kernel
  sh_mem_size = (2*nn)*sizeof(double);
  
  // allocate device memory
  cuError = cudaMalloc((void **) &d_error, sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);

  // execute jacobi iterations until solved
  while(min_iteration>=0 || h_error>=max_error) {
    // launch kernel for performing one jacobi iteration
    cudaGetLastError();
    jacobi_iteration<<<griddim, blockdim, sh_mem_size>>>(ds, epsilon0, d_rho, d_phi, d_error);
    cu_sync_check(__FILE__, __LINE__);
    
    // copy error variable from  device to host memory (actualize host error)
    cuError = cudaMemcpy(&h_error, d_error, sizeof(double), cudaMemcpyDeviceToHost);
    cu_check(cuError, __FILE__, __LINE__);
    
    // actualize counter
    min_iteration--;
  }

  // free device memory
  cudaFree(d_error);

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
  size_t sh_mem_size;
  
  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  // set dimensions of grid of blocks and blocks of threads for jacobi kernel
  blockdim = nn;
  griddim = 1;
  
  // define size of shared memory for field_derivation kernel
  sh_mem_size = nn*sizeof(double);
  
  // launch kernel for performing the derivation of the potential to obtain the electric field
  cudaGetLastError();
  field_derivation<<<griddim, blockdim, sh_mem_size>>>(ds, d_phi, d_E);
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
  int ic;                       // cell index of each particle
  particle reg_p;               // register copy of particle analized
  double dist;                  // distance to down vertex of the cell
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory variables

  // initialize charge density in shared memory to 0.0
  if (tidx < nn) sh_partial_rho[tidx] = 0.0;
  __syncthreads();
  
  //--- deposition of charge
  
  if (tid < num_p) {
    // load particle in registers
    reg_p = g_p[tid];
    // calculate what cell the particle is in
    ic = __double2int_rd(reg_p.r/ds);
    // calculate distances from particle to down vertex of the cell
    dist = fabs(__int2double_rn(ic)*ds-reg_p.r)/ds;
    // acumulate charge in partial rho
    atomicAdd(&sh_partial_rho[ic], q*(1.0-dist));    //down vertex
    atomicAdd(&sh_partial_rho[ic+1], q*dist);        //upper vertex
  }
  __syncthreads();
  
  

  //---- volume correction (shared memory)
  
  if (tidx > 0 && tidx < nn-1) {
    sh_partial_rho[tidx] /= ds*ds*ds;
  } else if (tidx == 0 || tidx == nn-1) {
    sh_partial_rho[tidx] /= 2*ds*ds*ds;
  }

  //---- charge acumulation in global memory
  
  if (tidx < nn) atomicAdd(&g_rho[tidx], sh_partial_rho[tidx]);
  __syncthreads();
  return;
}

/**********************************************************/

__global__ void jacobi_iteration (double ds, double epsilon0, double *g_rho, double *g_phi, double *g_error)
{
  /*----------------------------- function body -------------------------*/
  
  // shared memory
  double *phi_old = (double *) sh_mem;                              //
  double *sh_error = (double *) &phi_old[blockDim.x];   // manually set up shared memory
  
  // registers
  double phi_new, rho_dummy;
  int tid = (int) threadIdx.x;
  int bdim = (int) blockDim.x;
  
  /*------------------------------ kernel body --------------------------*/
  
  // load phi data from global to shared memory
  phi_old[tid] = g_phi[tid];
  __syncthreads();
  
  // load charge density data into registers
  rho_dummy = ds*ds*g_rho[tid]/(2.0*epsilon0);
  __syncthreads();
  
  // actualize interior mesh points
  if (tid != 0 && tid != bdim-1) {
    phi_new = 0.5*(rho_dummy + phi_old[tid-1] + phi_old[tid+1]);
  } else {
    phi_new = phi_old[tid];
  }
  __syncthreads();
  
  // evaluate local errors
  sh_error[tid] = fabs(phi_new-phi_old[tid]);
  __syncthreads();
  
  // reduction for obtaining maximum error in current block
  for (int stride = 1; stride < bdim; stride <<= 1) {
    if (tid%(stride*2) == 0) {
      if (tid+stride < bdim) {
        if (sh_error[tid]<sh_error[tid+stride]) sh_error[tid] = sh_error[tid+stride];
      }
    }
    __syncthreads();
  }

  // store new values of phi in global memory
  __syncthreads();
  g_phi[tid] = phi_new;
  
  // store maximun error in global memory
  if (tid == 0) *g_error = sh_error[tid];
 
  return;
}

/**********************************************************/

__global__ void field_derivation (double ds, double *g_phi, double *g_E)
{
  /*---------------------------- kernel variables ------------------------*/
  
  // shared memory
  double *sh_phi = (double *) sh_mem;       // manually set up shared memory variables
  
  // registers
  double reg_E;
  int tid = (int) threadIdx.x;
  int bdim = (int) blockDim.x;
  
  /*------------------------------ kernel body --------------------------*/
  
  // load phi data from global memory to shared memory
  sh_phi[tid] = g_phi[tid];
  __syncthreads();
  
  // calculate electric fields 
  if (tid == 0) {
    reg_E = (sh_phi[tid]-sh_phi[tid+1])/ds;
  } else if ( tid == bdim-1) {
    reg_E = (sh_phi[tid-1]-sh_phi[tid])/ds;
  } else {
    reg_E = (sh_phi[tid-1]-sh_phi[tid+1])/(2.0*ds);
  }
  __syncthreads();
  
  // store electric fields in global memory
  g_E[tid] = reg_E;

  return;
}

/**********************************************************/



/******************** DEVICE FUNCTION DEFINITIONS ********************/


/**********************************************************/
