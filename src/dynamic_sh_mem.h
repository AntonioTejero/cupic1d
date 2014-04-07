/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


#ifndef DYNAMIC_SH_MEM_H
#define DYNAMIC_SH_MEM_H

// variable for allowing dynamic allocation of __shared__ memory (used in charge_deposition, poisson_solver, )
extern __shared__ float sh_mem[];

#endif
