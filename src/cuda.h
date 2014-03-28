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
void cuda_reset(double **d_rho, double **d_phi, double **d_Ex, double **d_Ey, particle **d_e, particle **d_i, int **d_e_bm, int **d_i_bm);

// device kernels



// device functions 



#endif
