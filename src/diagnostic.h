/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/



#ifndef DIAGNOSTIC_H
#define DIAGNOSTIC_H

/****************************** HEADERS ******************************/

#include "stdh.h"
#include "init.h"
#include "cuda.h"

/************************ SIMBOLIC CONSTANTS *************************/

#define AVG_MESH_BLOCK_DIM 512   //block dimension for mesh_sum and mesh_norm
#define ENERGY_BLOCK_DIM 512     //block dimension for energy solver kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function
void avg_mesh(double *d_foo, double *d_avg_foo, int *count);
void particles_snapshot(particle *d_p, int num_p, string filename);
void mesh_snapshot(double *d_m, string filename);
void save_bins(particle *d_p, int num_p, string filename);
double particle_energy(double *d_phi,  particle *d_p, double m, double q, int num_p);
void log(double t, int num_e, int num_i, double U_e, double U_i, double dtin_i);
void calibrate_dtin_i(double dtin_i, bool should_increase);
double calculate_vd_i(double dtin_i);

// device kernels
__global__ void mesh_sum(double *g_foo, double *g_avg_foo, int nn);
__global__ void mesh_norm(double *g_avg_foo, double norm_cst, int nn);
__global__ void energy_kernel(double *g_U, double *g_phi, int nn, double ds,
                              particle *g_p, double m, double q, int num_p);

// device functions 

#endif
