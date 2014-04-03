/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


#ifndef CUDA_H
#define CUDA_H

/****************************** HEADERS ******************************/

#include "stdh.h"
#include "init.h"
#include "diagnostic.h"

/************************ SIMBOLIC CONSTANTS *************************/



/************************ FUNCTION PROTOTIPES ************************/

// host function
void cu_check(cudaError_t cuError, const string file, const int line);
void cu_sync_check(const string file, const int line);

// device kernels


// device functions (overload atomic functions for double precision support)
__device__ double atomicAdd(double* address, double val);
__device__ double atomicSub(double* address, double val);

#endif
