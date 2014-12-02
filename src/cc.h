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
void cc (double t, int *num_e, particle **d_e, double *dtin_e, int *num_i, particle **d_i, double *dtin_i, 
         double *q_p, double *d_phi, double *d_E, curandStatePhilox4_32_10_t *state);
void abs_emi_cc(double t, double *tin, double dtin, double kt, double vd, double m, double q, double *q_p, 
                int *h_num_p, particle **d_p, double *d_E, curandStatePhilox4_32_10_t *state);

// device kernels
__global__ void pEmi(particle *g_p, int num_p, int n_in, double *g_E, double vth, double vd, double qm, int nn, 
                     double L, double fpt, double fvt, double tin, double dtin, curandStatePhilox4_32_10_t *state);
__global__ void pRemover (particle *g_p, int *g_num_p, double L, int *g_num_abs_p);

#endif 
