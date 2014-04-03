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
void charge_deposition(double *d_rho, particle *d_e, int num_e, particle *d_i, int num_i);
void poisson_solver(double max_error, double *d_rho, double *d_phi);
void field_solver(double *d_phi, double *d_E);

// device kernels
__global__ void particle_to_grid(double ds, int nn, double *g_rho, particle *g_p, int num_p, double q);
__global__ void jacobi_iteration (double ds, double epsilon0, double *g_rho, double *g_phi, double *g_error);
__global__ void field_derivation (double ds, double *g_phi, double *g_E);

// device functions 

#endif
