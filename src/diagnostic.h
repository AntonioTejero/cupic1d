/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/


#ifndef DIAGNOSTIC_H
#define DIAGNOSTIC_H

/****************************** HEADERS ******************************/

#include "stdh.h"
#include "init.h"
#include "cuda.h"

/************************ SIMBOLIC CONSTANTS *************************/

#define DIAGNOSTIC_BLOCK_DIM 1024   //block dimension for defragmentation kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function
void particles_snapshot(particle *d_p, int num_p, string filename);
void mesh_snapshot(double *d_m, string filename);
void save_bins(particle *d_p, int num_p, string filename);

// device kernels



// device functions 



#endif
