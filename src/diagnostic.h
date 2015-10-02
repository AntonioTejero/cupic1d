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

#define AVG_MESH_BLOCK_DIM 512      // block dimension for mesh_sum and mesh_norm
#define ENERGY_BLOCK_DIM 512        // block dimension for energy solver kernel
#define PARTICLE2DF_BLOCK_DIM 512   // block dimension for particle2df kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function
void avg_mesh(double *d_foo, double *d_avg_foo, int *count);
void eval_df(double *d_avg_ddf, double *d_avg_vdf, double vmax, double vmin, particle *d_p, int num_p, int *count);
double eval_particle_energy(double *d_phi,  particle *d_p, double m, double q, int num_p);
void particles_snapshot(particle *d_p, int num_p, string filename);
void save_mesh(double *d_m, string filename);
void save_ddf(double *d_avg_ddf, string filename);
void save_vdf(double *d_avg_vdf, double vmax, double vmin, string filename);
void save_log(double t, int num_e, int num_i, int num_se, int num_he, double U_e, double U_i, double U_se, double U_he, 
              double *q_e, double *q_he, double *q_se, double *q_i, double vd_i, double *d_phi);

// device kernels
__global__ void mesh_sum(double *g_foo, double *g_avg_foo, int nn);
__global__ void mesh_norm(double *g_avg_foo, double norm_cst, int nn);
__global__ void particle2df(double *g_avg_ddf, int n_bin_ddf, double L, double *g_avg_vdf, int n_vdf, 
                            int n_bin_vdf, double vmax, double vmin, particle *g_p, int num_p);
__global__ void energy_kernel(double *g_U, double *g_phi, int nn, double ds,
                              particle *g_p, double m, double q, int num_p);

// device functions 

#endif
