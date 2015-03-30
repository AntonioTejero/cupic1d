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

#define PAR_MOV_BLOCK_DIM 512   //block dimension for defragmentation kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function
void particle_mover(particle *d_e, int num_e, particle *d_i, int num_i, 
                    particle *d_se, int num_se, particle *d_he, int num_he, double *d_E);

// device kernels
__global__ void leap_frog_step(double q, double m, int num_p, particle *g_p, double dt, 
                               double ds, int nn, double *g_E);

// device functions 


#endif
