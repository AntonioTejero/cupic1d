/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


#ifndef PARTICLES_H
#define PARTICLES_H

/****************************** HEADERS ******************************/

#include "stdh.h"
#include "init.h"
#include "diagnostic.h"
#include "dynamic_sh_mem.h"
#include "cuda.h"

/************************ SIMBOLIC CONSTANTS *************************/

#define PAR_MOV_BLOCK_DIM 1024   //block dimension for defragmentation kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function
void particle_mover(particle *d_e, int num_e, particle *d_i, int num_i, double *d_E);

// device kernels
__global__ void leap_frog_step(double dt, double m, double q, particle *g_p, int num_p, double *g_E, 
                               int nn, double ds);

// device functions 


#endif
