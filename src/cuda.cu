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


/******************** DEVICE KERNELS DEFINITIONS *********************/




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
