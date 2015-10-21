/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


#ifndef CC_H
#define CC_H

/****************************** HEADERS ******************************/

#include "stdh.h"
#include "init.h"
#include "random.h"
#include "diagnostic.h"
#include "cuda.h"
#include "dynamic_sh_mem.h"

/************************ SIMBOLIC CONSTANTS *************************/

#define P_RMV_BLK_SZ 1024       //block dimension for particle remover kernel

/************************ FUNCTION PROTOTIPES ************************/

// host function
void cc (double t, int *num_e, particle **d_e, int *num_he, particle **d_he, int *num_i, particle **d_i, 
         double *vd_i, int *num_se, particle **d_se, double *q_e, double *q_he, double *q_se, double *q_i, 
         double *d_phi, double *d_E, curandStatePhilox4_32_10_t *state);
void abs_emi_cc(double t, double *tin, double dtin, double kt, double vd, double m, double q, double *q_p, 
                int *h_num_p, particle **d_p, double pos, double *d_E, curandStatePhilox4_32_10_t *state);
void recalculate_dtin(double *dtin_e, double *dtin_he, double *dtin_i, double vd_i, double phi_p, double phi_s);
void calibrate_ion_flux(double *vd_i, double E_s, double *phi_s, double phi_p);
void calibrate_ion_flux(double *vd_i, double *d_E, double *phi_s);
double cuasineutral_field(double phi, double phi_p, double mi, double kti, double kthe, double alpha);
double cuasineutral_j2(double phi, double phi_p, double mi, double kti, double kthe, double alpha);
double cuasineutral_ni(double phi, double phi_p, double kthe, double alpha);
double cuasineutral_dni(double phi, double phi_p, double kthe, double alpha);
double cuasineutral_ne(double phi, double phi_p);
double cuasineutral_nhe(double phi, double phi_p, double kthe, double alpha);

// device kernels
__global__ void pEmi(particle *g_p, int num_p, int n_in, double *g_E, double vth, double vd, double qm, int nn, 
                     double pos, double fpt, double fvt, double tin, double dtin, curandStatePhilox4_32_10_t *state);
__global__ void pRemover (particle *g_p, int *g_num_p, double L, int *g_num_abs_p);

#endif 
