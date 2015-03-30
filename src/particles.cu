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

void particle_mover(particle *d_e, int num_e, particle *d_i, int num_i, 
                    particle *d_se, int num_se, particle *d_he, int num_he, double *d_E) 
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
  
  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  // set size of __shared__ memory for leap_frog kernel
  sh_mem_size = nn*sizeof(double);

  //---- move electrons
  
  // set dimensions of grid of blocks and blocks of threads for leap_frog kernel (electrons)
  blockdim = PAR_MOV_BLOCK_DIM;
  griddim = int(num_e/PAR_MOV_BLOCK_DIM)+1;

  // call to leap_frog_step kernel (electrons)
  cudaGetLastError();
  leap_frog_step<<<griddim, blockdim, sh_mem_size>>>(qe, me, num_e, d_e, dt, ds, nn, d_E);
  cu_sync_check(__FILE__, __LINE__);
   
  //---- move ions  
 
  // set dimensions of grid of blocks and blocks of threads for leap_frog kernel (ions)
  blockdim = PAR_MOV_BLOCK_DIM;
  griddim = int(num_i/PAR_MOV_BLOCK_DIM)+1;
 
  // call to leap_frog_step kernel (ions)
  cudaGetLastError();
  leap_frog_step<<<griddim, blockdim, sh_mem_size>>>(qi, mi, num_i, d_i, dt, ds, nn, d_E);
  cu_sync_check(__FILE__, __LINE__);
 
  //---- move secondary electrons
  
  // set dimensions of grid of blocks and blocks of threads for leap_frog kernel (secondary electrons)
  blockdim = PAR_MOV_BLOCK_DIM;
  griddim = int(num_se/PAR_MOV_BLOCK_DIM)+1;

  // call to leap_frog_step kernel (secondary electrons)
  cudaGetLastError();
  leap_frog_step<<<griddim, blockdim, sh_mem_size>>>(qe, me, num_se, d_se, dt, ds, nn, d_E);
  cu_sync_check(__FILE__, __LINE__);
 
 //---- move hot electrons
  
  // set dimensions of grid of blocks and blocks of threads for leap_frog kernel (hot electrons)
  blockdim = PAR_MOV_BLOCK_DIM;
  griddim = int(num_he/PAR_MOV_BLOCK_DIM)+1;

  // call to leap_frog_step kernel (hot electrons)
  cudaGetLastError();
  leap_frog_step<<<griddim, blockdim, sh_mem_size>>>(qe, me, num_he, d_he, dt, ds, nn, d_E);
  cu_sync_check(__FILE__, __LINE__);
   
  return;
}

/**********************************************************/



/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void leap_frog_step(double q, double m, int num_p, particle *g_p, double dt, 
                               double ds, int nn, double *g_E)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  double *sh_E = (double *) sh_mem;   // manually set up shared memory variables

  // kernel registers
  int tidx = (int) threadIdx.x;
  int tid = (int) threadIdx.x + (int) blockDim.x * (int) blockIdx.x;  // thread Id
  int bdim = (int) blockDim.x;  // block dimension
  particle reg_p;               // register particles
  int ic;                       // cell index
  double dist;                  // distance from particle to nearest down vertex (normalized to ds)
  double F;                     // force suffered for each register particle

  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory variables
  
  // load fields from global memory
  for (int i = tidx; i<nn; i += bdim) {
    sh_E[i] = g_E[i];
  }
  __syncthreads();
  
  //---- Process batches of particles
  
  if (tid < num_p) {
    // load particle data in registers
    reg_p = g_p[tid];
    
    // find cell index
    ic = __double2int_rd(reg_p.r/ds);

    // evaluate distance to nearest down vertex (normalized to ds)
    dist = fabs(reg_p.r-ic*ds)/ds;

    // calculate particle's forces
    F = q*(sh_E[ic]*(1.0-dist) + sh_E[ic+1]*dist);

    // move particles
    reg_p.v += dt*F/m;
    reg_p.r += dt*reg_p.v;
    
    // store particle data in global memory
    g_p[tid] = reg_p;
  }
  
  return;
}

/**********************************************************/


/******************** DEVICE FUNCTION DEFINITIONS ********************/

