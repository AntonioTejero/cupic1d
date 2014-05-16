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

void avg_mesh(double *d_foo, double *d_avg_foo)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const int nn = init_nn();          // number of nodes
  static const int n_save = init_n_save();  // number of iterations to average
  static int count = 0;                     // number of iterations averaged
  
  dim3 griddim, blockdim;
  cudaError_t cuError;

  // device memory
  
  /*----------------------------- function body -------------------------*/

  // check if restart of avg_foo is needed
  if (count == n_save) {
    //reset count
    count = 0;

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
  count++;

  // normalize average if reached desired number of iterations
  if (count == n_save ) {
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
