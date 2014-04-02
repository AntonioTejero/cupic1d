/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/

/****************************** HEADERS ******************************/

#include "diagnostic.h"

/********************* HOST FUNCTION DEFINITIONS *********************/

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



/**********************************************************/
