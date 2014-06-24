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
void init_sim(double **d_rho, double **d_phi, double **d_E, double **d_avg_rho, double **d_avg_phi, double **d_avg_E, 
              particle **d_e, int *num_e, particle **d_i, int *num_i, double **d_avg_ddf_e, double **d_avg_vdf_e, 
              double **d_avg_ddf_i, double **d_avg_vdf_i, double *t, curandStatePhilox4_32_10_t **state);
void create_particles(particle **d_i, int *num_i, particle **d_e, int *num_e,
                      curandStatePhilox4_32_10_t **state);
void initialize_mesh(double **d_rho, double **d_phi, double **d_E, particle *d_i, int num_i, particle *d_e, int num_e);
void initialize_avg_mesh(double **d_avg_rho, double **d_avg_phi, double **d_avg_E);
void initialize_avg_df(double **d_avg_ddf_e, double **d_avg_vdf_e, double **d_avg_ddf_i, double **d_avg_vdf_i);
void adjust_leap_frog(particle *d_i, int num_i, particle *d_e, int num_e, double *d_E);
void load_particles(particle **d_i, int *num_i, particle **d_e, int *num_e, curandStatePhilox4_32_10_t **state);
void read_particle_file(string filename, particle **d_p, int *num_p);
template <typename type> void read_input_file(type *data, int n);
double init_qi(void);
double init_qe(void);
double init_mi(void);
double init_me(void);
double init_kti(void);
double init_kte(void);
double init_phi_p(void);
double init_n(void);
double init_L(void);
double init_ds(void);
double init_dt(void);
double init_dtin_i(void);
double init_dtin_e(void);
double init_epsilon0(void);
int init_nc(void);
int init_nn(void);
double init_Dl(void);
int init_n_ini(void);
int init_n_prev(void);
int init_n_save(void);
int init_n_fin(void);
int init_n_bin_ddf(void);
int init_n_bin_vdf(void);
int init_n_vdf(void);
int init_n_sigmas_vdf(void);

// device kernels
__global__ void init_philox_state(curandStatePhilox4_32_10_t *state);
__global__ void create_particles_kernel(particle *g_p, int num_p, double kt, double m, double L, 
                                        curandStatePhilox4_32_10_t *state);
__global__ void fix_velocity(double q, double m, int num_p, particle *g_p, double dt, double ds, int nn, double *g_E);

#endif
