/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


#ifndef RAND_H
#define RAND_H

/****************************** HEADERS ******************************/

#include <curand_kernel.h>      //curand library for random number generation (__device__ functions)

/************************ SIMBOLIC CONSTANTS *************************/

#define CURAND_BLOCK_DIM 64      //block dimension for curand kernels

/************************ FUNCTION PROTOTIPES ************************/

#endif
