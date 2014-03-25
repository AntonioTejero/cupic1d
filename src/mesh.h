/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


#ifndef MESH_H
#define MESH_H

/****************************** HEADERS ******************************/

#include "stdh.h"
#include "init.h"
#include "dynamic_sh_mem.h"
#include "cuda.h"

/************************ SIMBOLIC CONSTANTS *************************/

#define CHARGE_DEP_BLOCK_DIM 512   //block dimension for particle2grid kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function
void charge_deposition(double *d_rho, particle *d_e, int *d_e_bm, particle *d_i, int *d_i_bm);
void poisson_solver(double max_error, double *d_rho, double *d_phi);
void field_solver(double *d_phi, double *d_E);

// device kernels
__global__ void fast_particle_to_grid(int nn, double ds, double *rho, particle *g_e, int *g_e_bm, 
                                      particle *g_i, int *g_i_bm);
__global__ void jacobi_iteration (dim3 blockdim, double ds, double epsilon0, double *rho, double *phi, 
                                  double *block_error);
__global__ void field_derivation (double ds, double *g_phi, double *g_E);

// device functions (overload atomic functions for double precision support)
__device__ double atomicAdd(double* address, double val);
__device__ double atomicSub(double* address, double val);

#endif
