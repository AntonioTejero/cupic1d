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

void charge_deposition(double *d_rho, particle *d_e, int *d_e_bm, particle *d_i, int *d_i_bm) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double ds = init_ds();   // spatial step
  static const int nn = init_nn();      // number of nodes
  static const int nc = init_nc();      // number of cells
  
  dim3 griddim, blockdim;
  cudaError_t cuError;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // initialize device memory to zeros
  cuError = cudaMemset(d_rho, 0, nn*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  
  // set dimensions of grid of blocks and blocks of threads for particle defragmentation kernel
  griddim = nc;
  blockdim = CHARGE_DEP_BLOCK_DIM;
  
  // call to fast_particle_to_grid kernel
  cudaGetLastError();
  fast_particle_to_grid<<<griddim, blockdim, sh_mem_size>>>(ds, d_rho, d_e, d_e_bm, d_i, d_i_bm);
  cu_sync_check(__FILE__, __LINE__);
  
  return;
}

/**********************************************************/

void poisson_solver(double max_error, double *d_rho, double *d_phi) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double ds = init_ds();               // spatial step
  static const int nnx = init_nnx();                // number of nodes in x dimension
  static const int nny = init_nny();                // number of nodes in y dimension
  static const double epsilon0 = init_epsilon0();   // electric permitivity of free space
  
  double *h_block_error;
  double error = max_error*10;
  int min_iteration = max(nnx, nny);
  
  dim3 blockdim, griddim;
  size_t sh_mem_size;
  cudaError_t cuError;

  // device memory
  double *d_block_error;
  
  /*----------------------------- function body -------------------------*/
  
  // set dimensions of grid of blocks and blocks of threads for jacobi kernel
  blockdim.x = nnx;
  blockdim.y = 512/nnx;
  griddim = (nny-2)/blockdim.y;
  
  // define size of shared memory for jacobi_iteration kernel
  sh_mem_size = (2*blockdim.x*(blockdim.y+1)+blockdim.y)*sizeof(double);
  
  // allocate host memory
  h_block_error = new double[griddim.x];
  
  // allocate device memory
  cuError = cudaMalloc((void **) &d_block_error, griddim.x*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);

  // execute jacobi iterations until solved
  while(min_iteration>=0 || error>=max_error)
  {
    // launch kernel for performing one jacobi iteration
    cudaGetLastError();
    jacobi_iteration<<<griddim, blockdim, sh_mem_size>>>(blockdim, ds, epsilon0, d_rho, d_phi, d_block_error);
    cu_sync_check(__FILE__, __LINE__);
    
    // copy device memory to host memory for analize errors
    cuError = cudaMemcpy(h_block_error, d_block_error, griddim.x*sizeof(double), cudaMemcpyDeviceToHost);
    cu_check(cuError, __FILE__, __LINE__);
    
    // evaluate max error in the iteration
    error = 0;
    for (int i = 0; i < griddim.x; i++)
    {
      if (h_block_error[i]>error) error = h_block_error[i];
    }
    
    // actualize counter
    min_iteration--;
  }

  // free host and device memory
  free(h_block_error);
  cudaFree(d_block_error);
  return;
}

/**********************************************************/

void field_solver(double *d_phi, double *d_Ex, double *d_Ey) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double ds = init_ds();   // spatial step
  static const int nnx = init_nnx();    // number of nodes in x dimension
  static const int nny = init_nny();    // number of nodes in y dimension
  
  dim3 blockdim, griddim;
  size_t sh_mem_size;
  
  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  // set dimensions of grid of blocks and blocks of threads for jacobi kernel
  blockdim.x = nnx;
  blockdim.y = 512/nnx;
  griddim = (nny-2)/blockdim.y;
  
  // define size of shared memory for field_derivation kernel
  sh_mem_size = blockdim.x*(blockdim.y+2)*sizeof(double);
  
  // launch kernel for performing the derivation of the potential to obtain the electric field
  cudaGetLastError();
  field_derivation<<<griddim, blockdim, sh_mem_size>>>(ds, d_phi, d_Ex, d_Ey);
  cu_sync_check(__FILE__, __LINE__);

  return;
}

/**********************************************************/



/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void fast_particle_to_grid(double ds, double *g_rho, particle *g_e, int *g_e_bm, 
                                      particle *g_i, int *g_i_bm)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ double sh_partial_rho[2];   // partial rho of each bin
  __shared__ int sh_e_bm[2];             // electron bookmarks (__shared__)
  __shared__ int sh_i_bm[2];             // ion bookmarks (__shared__)
  
  // kernel registers
  int tid = (int) threadIdx.x;
  int bid = (int) blockIdx.x;
  int bdim = (int) blockDim.x;
  particle p;
  double dist;                  // distance to down vertex of the cell
  double reg_partial_rho[2];    // partial rho of each thread of each bin
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory variables

  // initialize charge density in shared memory to 0.0
  if (tid < 2) sh_partial_rho[tid] = 0.0;
  __syncthreads();
  
  // load bin bookmarks from global memory
  if (tid < 2) {
    sh_e_bm[tid] = g_e_bm[bid*2+tid];
    sh_i_bm[tid] = g_i_bm[bid*2+tid];
  }
  __syncthreads();
  
  //--- deposition of charge
  reg_partial_rho[0] = reg_partial_rho[1] = 0.0;
  __syncthreads();
  
  // electron deposition
  if (sh_e_bm[0] >= 0 && sh_e_bm[1] >= 0) {
    for (int i = sh_e_bm[0]+tid; i<=sh_e_bm[1]; i+=bdim) {
      // load electron in registers
      p = g_e[i];
      // calculate distances from particle to down vertex of the cell
      dist = fabs(__int2double_rn(bid)*ds-p.r)/ds;
      // acumulate charge in partial rho
      reg_partial_rho[0] -= (1.0-dist);  //down vertex
      reg_partial_rho[1] -= dist;        //upper vertex
    }
  }
  __syncthreads();
  
  // ion deposition
  if (sh_i_bm[0] >= 0 && sh_i_bm[1] >= 0) {
    for (int i = sh_i_bm[0]+tid; i<=sh_i_bm[1]; i+=bdim) {
      // load electron in registers
      p = g_i[i];
      // calculate distances from particle to down vertex of the cell
      dist = fabs(__int2double_rn(bid)*ds-p.r)/ds;
      // acumulate charge in partial rho
      reg_partial_rho[0] += (1.0-dist);  //down vertex
      reg_partial_rho[1] += dist;        //upper vertex     // load electron in registers
    }
  }
  __syncthreads();
  
  //---- acumulation of charge
  
  for (int i = tid; i < 2*nnx; i+=bdim) {
    atomicAdd(rho+bid*nnx+i, sh_partial_rho[i]);
  }
  __syncthreads();

  //---- volume correction (global)
  
  if (blockIdx.x > 0) {
    for (int i = threadIdx.x; i<nnx; i+=blockDim.x) {
      rho[nnx*blockIdx.x+i] /= ds*ds*ds; 
    }
  } else {
    for (int i = threadIdx.x; i<nnx; i+=blockDim.x) {
      rho[i] /= 0.5*ds*ds*ds;
    }
    for (int i = nnx+threadIdx.x; i<2*nnx; i+=blockDim.x) {
      rho[nnx*blockDim.x+i] /= 0.5*ds*ds*ds;
    }
  }

  return;
}

/**********************************************************/

__global__ void jacobi_iteration (dim3 blockdim, double ds, double epsilon0, double *rho, double *phi, double *block_error)
{
  /*----------------------------- function body -------------------------*/
  
  // shared memory
  double *phi_old = (double *) sh_mem;                              //
  double *error = (double *) &phi_old[blockdim.x*(blockdim.y+2)];   // manually set up shared memory variables inside whole shared memory
  double *aux_shared = (double *) &error[blockdim.x*blockdim.y];    //
  
  // registers
  double phi_new, rho_dummy;
  int global_mem_index = blockDim.x + blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
  int shared_mem_index = blockDim.x + threadIdx.y*blockDim.x + threadIdx.x;
  int thread_index = threadIdx.x + threadIdx.y*blockDim.x;
  
  /*------------------------------ kernel body --------------------------*/
  
  // load phi data from global memory to shared memory
  phi_old[shared_mem_index] = phi[global_mem_index];
  
  // load comunication zones into shared memory
  if (threadIdx.y == 0)
  {
    phi_old[shared_mem_index-blockDim.x] = phi[global_mem_index-blockDim.x];
  }
  if (threadIdx.y == blockDim.y-1)
  {
    phi_old[shared_mem_index+blockDim.x] = phi[global_mem_index+blockDim.x];
  }
  // load charge density data into registers
  rho_dummy = ds*ds*rho[global_mem_index]/epsilon0;
  __syncthreads();
  
  // actualize cyclic contour conditions
  if (threadIdx.x == 0)
  {
    phi_new = 0.25*(rho_dummy + phi_old[shared_mem_index+blockDim.x-2]
    + phi_old[shared_mem_index+1] + phi_old[shared_mem_index+blockDim.x]
    +phi_old[shared_mem_index-blockDim.x]);
    aux_shared[threadIdx.y] = phi_new;
  }
  __syncthreads();
  if (threadIdx.x == blockDim.x-1)
  {
    phi_new = aux_shared[threadIdx.y];
  }
  
  // actualize interior mesh points
  if (threadIdx.x != 0 && threadIdx.x != blockDim.x-1)
  {
    phi_new = 0.25*(rho_dummy + phi_old[shared_mem_index-1]
    + phi_old[shared_mem_index+1] + phi_old[shared_mem_index+blockDim.x]
    + phi_old[shared_mem_index-blockDim.x]);
  }
  __syncthreads();
  
  // evaluate local errors
  error[thread_index] = fabs(phi_new-phi_old[shared_mem_index]);
  __syncthreads();
  
  // reduction for obtaining maximum error in current block
  for (int stride = 1; stride < blockDim.x*blockDim.y; stride <<= 1)
  {
    if (thread_index%(stride*2) == 0)
    {
      if (thread_index+stride<blockDim.x*blockDim.y)
      {
        if (error[thread_index]<error[thread_index+stride])
          error[thread_index] = error[thread_index+stride];
      }
    }
    __syncthreads();
  }
  
  // store block error in global memory
  if (thread_index == 0)
  {
    block_error[blockIdx.x] = error[0];
  }
  
  // store new values of phi in global memory
  phi[global_mem_index] = phi_new;
  
  return;
}

/**********************************************************/

__global__ void field_derivation (double ds, double *phi_global, double *Ex_global, double *Ey_global)
{
  /*---------------------------- kernel variables ------------------------*/
  
  // shared memory
  double *phi = (double *) sh_mem;       // manually set up shared memory variables inside whole shared memory
  
  // registers
  double Ex, Ey;
  int shared_mem_index = blockDim.x + threadIdx.y*blockDim.x + threadIdx.x;
  int global_mem_index = shared_mem_index + blockIdx.x*(blockDim.x*blockDim.y);
  
  /*------------------------------ kernel body --------------------------*/
  
  // load phi data from global memory to shared memory
  phi[shared_mem_index] = phi_global[global_mem_index];
  
  // load comunication zones into shared memory
  if (threadIdx.y == 0)
  {
    phi[shared_mem_index-blockDim.x] = phi_global[global_mem_index-blockDim.x];
  }
  else if (threadIdx.y == blockDim.y-1)
  {
    phi[shared_mem_index+blockDim.x] = phi_global[global_mem_index+blockDim.x];
  }
  __syncthreads();
  
  // calculate electric fields (except top and bottom)
  if (threadIdx.x == 0)                       // calculate fields in left nodes of simulation (cyclic contour conditions)
  {
    Ex = (phi[shared_mem_index+blockDim.x-2] - phi[shared_mem_index+1])/(2.0*ds);
    Ey = (phi[shared_mem_index-blockDim.x]-phi[shared_mem_index+blockDim.x])/(2.0*ds);
  }
  else if (threadIdx.x == blockDim.x-1)       // calculate fields in right nodes of simulation (cyclic contour conditions)
  {
    Ex = (phi[shared_mem_index-1] - phi[shared_mem_index-blockDim.x+2])/(2.0*ds);
    Ey = (phi[shared_mem_index-blockDim.x]-phi[shared_mem_index+blockDim.x])/(2.0*ds);
  } 
  else                                        // actualize interior mesh points
  {
    Ex = (phi[shared_mem_index-1]-phi[shared_mem_index+1])/(2.0*ds);
    Ey = (phi[shared_mem_index-blockDim.x]-phi[shared_mem_index+blockDim.x])/(2.0*ds);
  }
  __syncthreads();
  
  // store electric fields in global memory (except top and bottom)
  Ex_global[global_mem_index] = Ex;
  Ey_global[global_mem_index] = Ey;
  
  // calculate fields in top and bottom nodes of simulation
  if (blockIdx.x == 0)
  {
    if (threadIdx.y == 0)
    {
      Ey = (phi[threadIdx.x]-phi[shared_mem_index])/ds;
    }
  }
  else if (blockIdx.x == gridDim.x - 1)
  {
    if (threadIdx.y == blockDim.y - 1)
    {
      Ey = (phi[shared_mem_index]-phi[shared_mem_index+blockDim.x])/ds;
    }
  }
  __syncthreads();
  
  // store electric fields in top and bottom nodes in global memory
  if (blockIdx.x == 0)
  {
    if (threadIdx.y == 0)
    {
      Ey_global[global_mem_index-blockDim.x] = Ey;
      Ex_global[global_mem_index-blockDim.x] = 0.0;
    }
  }
  else if (blockIdx.x == gridDim.x - 1)
  {
    if (threadIdx.y == blockDim.y - 1)
    {
      Ey_global[global_mem_index+blockDim.x] = Ey;
      Ex_global[global_mem_index+blockDim.x] = 0.0;
    }
  }
  __syncthreads();
  
  return;
}

/**********************************************************/



/******************** DEVICE FUNCTION DEFINITIONS ********************/

__device__ double atomicAdd(double* address, double val)
{
  /*--------------------------- function variables -----------------------*/
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  
  /*----------------------------- function body -------------------------*/
  do 
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  
  return __longlong_as_double(old);
}

/**********************************************************/

__device__ double atomicSub(double* address, double val)
{
  /*--------------------------- function variables -----------------------*/
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  
  /*----------------------------- function body -------------------------*/
  do 
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val - __longlong_as_double(assumed)));
  } while (assumed != old);
  
  return __longlong_as_double(old);
}

/**********************************************************/
