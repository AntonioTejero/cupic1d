/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


/****************************** HEADERS ******************************/

#include "cuda.h"

/********************* HOST FUNCTION DEFINITIONS *********************/

void cu_check(cudaError_t cuError, const string file, const int line)
{
  // function variables
  
  // function body
  
  if (0 == cuError)
  {
    return;
  } else
  {
    cout << "CUDA error found in file " << file << " at line " << line << ". (error code: " << cuError << ")" << endl;
    cout << "Exiting simulation" << endl;
    exit(1);
  }
  
}

/**********************************************************/

void cu_sync_check(const string file, const int line)
{
  // function variables
  cudaError_t cuError;
  
  // function body
  
  cudaDeviceSynchronize();
  cuError = cudaGetLastError();
  if (0 == cuError)
  {
    return;
  } else
  {
    cout << "CUDA error found in file " << file << " at line " << line << ". (error code: " << cuError << ")" << endl;
    cout << "Exiting simulation" << endl;
    exit(1);
  }
  
}

/**********************************************************/

void cuda_reset(double **d_rho, double **d_phi, double **d_Ex, double **d_Ey, particle **d_e, particle **d_i, int **d_e_bm, int **d_i_bm)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const int nnx = init_nnx();      // number of nodes in the x dimension
  static const int nny = init_nny();      // number of nodes in the y dimension
  static const int ncy = init_ncy();      // number of cells in the y dimension
  int Ne = number_of_particles(*d_e_bm);  // number of electrons
  int Ni = number_of_particles(*d_i_bm);  // number of ions
  double *h_rho, *h_phi, *h_Ex, *h_Ey;
  particle *h_e, *h_i;
  int *h_e_bm, *h_i_bm;
  
  cudaError_t cuError;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  cout << "reseting device..." << endl;
  
  // allocate host memory for temporary storage of data
  h_rho = (double*) malloc(nnx*nny*sizeof(double));
  h_phi = (double*) malloc(nnx*nny*sizeof(double));
  h_Ex = (double*) malloc(nnx*nny*sizeof(double));
  h_Ey = (double*) malloc(nnx*nny*sizeof(double));
  h_e = (particle*) malloc(Ne*sizeof(particle));
  h_i = (particle*) malloc(Ni*sizeof(particle));
  h_e_bm = (int*) malloc(2*ncy*sizeof(int));
  h_i_bm = (int*) malloc(2*ncy*sizeof(int));
  
  // copy data from device to host
  cudaGetLastError();
  cuError = cudaMemcpy(h_rho, *d_rho, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(h_phi, *d_phi, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(h_Ex, *d_Ex, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(h_Ey, *d_Ey, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(h_e, *d_e, Ne*sizeof(particle), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(h_i, *d_i, Ni*sizeof(particle), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(h_e_bm, *d_e_bm, 2*ncy*sizeof(int), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(h_i_bm, *d_i_bm, 2*ncy*sizeof(int), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  
  // reset cuda Device
  cuError = cudaDeviceReset();
  cu_check(cuError, __FILE__, __LINE__);
  
  // allocate device memory for data
  cuError = cudaMalloc ((void **) d_rho, nnx*nny*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_phi, nnx*nny*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_Ex, nnx*nny*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_Ey, nnx*nny*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_e, Ne*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_i, Ni*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_e_bm, 2*ncy*sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_i_bm, 2*ncy*sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  
  // copy data from host to device
  cudaGetLastError();
  cuError = cudaMemcpy(*d_rho, h_rho, nnx*nny*sizeof(double), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(*d_phi, h_phi, nnx*nny*sizeof(double), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(*d_Ex, h_Ex, nnx*nny*sizeof(double), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(*d_Ey, h_Ey, nnx*nny*sizeof(double), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(*d_e, h_e, Ne*sizeof(particle), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(*d_i, h_i, Ni*sizeof(particle), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(*d_e_bm, h_e_bm, 2*ncy*sizeof(int), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(*d_i_bm, h_i_bm, 2*ncy*sizeof(int), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  
  // free host memory
  free(h_rho);
  free(h_phi);
  free(h_Ex);
  free(h_Ey);
  free(h_e);
  free(h_i);
  free(h_e_bm);
  free(h_i_bm);
  
  cout << "... reset completed" << endl;
  
  return;
}

/******************** DEVICE KERNELS DEFINITIONS *********************/



/**********************************************************/
