/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


#ifndef INIT_H
#define INIT_H

/****************************** HEADERS ******************************/

#include "stdh.h"
#include "random.h"
#include "mesh.h"
#include "particles.h"
#include "dynamic_sh_mem.h"
#include "cuda.h"

/************************ SIMBOLIC CONSTANTS *************************/

#define CST_ME 9.109e-31      // electron mass (kg)
#define CST_E 1.602e-19       // electron charge (C)
#define CST_KB 1.381e-23      // boltzmann constant (m^2 kg s^-2 K^-1)
#define CST_EPSILON 8.854e-12 // free space electric permittivity (s^2 C^2 m^-3 kg^-1)

/************************ FUNCTION PROTOTIPES ************************/

// host functions
void init_dev(void);
void init_sim(double **d_rho, double **d_phi, double **d_E, particle **d_e, particle **d_i, 
              int **d_e_bm, int **d_i_bm, double *t, curandStatePhilox4_32_10_t **state);
void create_particles(particle **d_i, int **d_i_bm, particle **d_e, int **d_e_bm, 
                      curandStatePhilox4_32_10_t **state);
void initialize_mesh(double **d_rho, double **d_phi, double **d_E, particle *d_i, int *d_i_bm, 
                     particle *d_e, int *d_e_bm);
void adjust_leap_frog(particle *d_i, int *d_i_bm, particle *d_e, int *d_e_bm, double *d_E);
void load_particles(particle **d_i, int **d_i_bm, particle **d_e, int **d_e_bm, 
                    curandStatePhilox4_32_10_t **state);
void read_particle_file(string filename, particle **d_p, int **d_bm);
void read_input_file(void *data, int data_size, int n);
double init_qi(void);
double init_qe(void);
double init_mi(void);
double init_me(void);
double init_kti(void);
double init_kte(void);
double init_phi_p(void);
double init_n(void);
double init_Lx(void);
double init_Ly(void);
double init_ds(void);
double init_dt(void);
double init_dtin_i(void);
double init_dtin_e(void);
double init_epsilon0(void);
int init_ncx(void);
int init_ncy(void);
int init_nnx(void);
int init_nny(void);
double init_Dl(void);
int init_n_ini(void);
int init_n_prev(void);
int init_n_save(void);
int init_n_fin(void);

// device kernels
__global__ void init_philox_state(curandStatePhilox4_32_10_t *state);
__global__ void create_particles_kernel(particle *g_p, int *g_p_bm, double kt, double m, int N, 
                                        int nc, double ds,  curandStatePhilox4_32_10_t *state);
__global__ void fix_velocity(double dt, double m, particle *g_p, int *g_bm, double *g_F);

#endif
