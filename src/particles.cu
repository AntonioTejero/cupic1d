/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


/****************************** HEADERS ******************************/

#include "particles.h"

/********************* HOST FUNCTION DEFINITIONS *********************/

void particle_mover(particle *d_e, int num_e, particle *d_i, int num_i, double *d_E) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double me = init_me();     // electron's mass
  static const double mi = init_mi();     // ion's mass
  static const double qe = init_qe();     // electron's charge
  static const double qi = init_qi();     // ions's charge
  static const double ds = init_ds();     // spatial step
  static const double dt = init_dt();     // time step
  static const int nn = init_nn();        // number of nodes  
  
  dim3 griddim, blockdim;
  size_t sh_mem_size;
  cudaError_t cuError;
  
  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  // set dimensions of grid of blocks and blocks of threads for leap_frog kernel
  griddim = 1;
  blockdim = PAR_MOV_BLOCK_DIM;
 
  // set size of __shared__ memory for leap_frog kernel
  sh_mem_size = nn*sizeof(double);

  //---- move electrons
  
  // call to leap_frog_step kernel (electrons)
  cudaGetLastError();
  leap_frog_step<<<griddim, blockdim>>>(dt, me, qe, d_e, num_e, d_E, nn, ds);
  cu_sync_check(__FILE__, __LINE__);
   
  //---- move ions  
  
  // call to leap_frog_step kernel (ions)
  cudaGetLastError();
  leap_frog_step<<<griddim, blockdim>>>(dt, mi, qi, d_i, num_i, d_E, nn, ds);
  cu_sync_check(__FILE__, __LINE__);
  
  return;
}

/**********************************************************/



/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void leap_frog_step(double dt, double m, double q, particle *g_p, int num_p, double *g_E, 
                               int nn, double ds)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  double *sh_E = (double *) sh_mem;   // manually set up shared memory variables inside whole shared memory

  // kernel registers
  int tid = (int) threadIdx.x;  // thread Id
  int bdim = (int) blockDim.x;  // block dimension
  particle reg_p;               // register particles
  int ic;                       // cell index
  double dist;                  // distance from particle to nearest down vertex (normalized to ds)
  double F;                     // force sufered for each register particle

  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory variables
  
  // load fields from global memory
  for (int i = tid; i<nn; i += bdim) {
    sh_E[i] = g_E[i];
  }
  __syncthreads();
  
  //---- Process batches of particles
  
  for (int i = tid; i <= num_p; i += bdim) {
    // load particle data in registers
    reg_p = g_p[i];
    
    // find cell index
    ic = __double2int_rd(reg_p.r/ds);

    // evaluate distance to nearest down vertex (normalized to ds)
    dist = fabs(reg_p.r-__int2double_rn(ic)*ds)/ds;

    // calculate particle's forces
    F = q*(sh_E[ic]*(1.0-dist) + sh_E[ic+1]*dist);

    // move particles
    reg_p.v += dt*F/m;
    reg_p.r += dt*reg_p.v;
    
    // store particle data in global memory
    g_p[i] = reg_p;
  }
  
  return;
}

/**********************************************************/



/******************** DEVICE FUNCTION DEFINITIONS ********************/

