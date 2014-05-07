/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


/****************************** HEADERS ******************************/

#include "cc.h"

/********************* HOST FUNCTION DEFINITIONS *********************/

void cc (double t, int *num_e, particle **d_e, int *num_i, particle **d_i, double *d_E,
         curandStatePhilox4_32_10_t *state)
{
  /*--------------------------- function variables -----------------------*/

  // host memory
  static const double me = init_me();           //
  static const double mi = init_mi();           // particle
  static const double kti = init_kti();         // properties
  static const double kte = init_kte();         //
  
  static const double dtin_e = init_dtin_e();   // time between particles insertions
  static const double dtin_i = init_dtin_i();   // sqrt(2.0*PI*m/kT)/(n*ds*ds)
  
  static double tin_e = t+dtin_e;               // time for next electron insertion
  static double tin_i = t+dtin_i;               // time for next ion insertion

  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  //---- electrons contour conditions
  
  abs_emi_cc(t, &tin_e, dtin_e, kte, me, -1.0, num_e, d_e, d_E, state);

  //---- ions contour conditions

  abs_emi_cc(t, &tin_i, dtin_i, kti, mi, 1.0, num_i, d_i, d_E, state);
  
  return;
}

/**********************************************************/

void abs_emi_cc(double t, double *tin, double dtin, double kt, double m, double q, int *h_num_p,
                particle **d_p, double *d_E, curandStatePhilox4_32_10_t *state)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double L = init_L();       //
  static const double ds = init_ds();     // geometric properties
  static const int nn = init_nn();        // of simulation
  
  static const double dt = init_dt();     //
  double fpt = t+dt;                      // timing variables
  double fvt = t+0.5*dt;                  //
  
  int in = 0;                             // number of particles added at plasma frontier
  
  cudaError cuError;                      // cuda error variable
  dim3 griddim, blockdim;                 // kernel execution configurations 

  // device memory
  int *d_num_p;                           // device number of particles
  particle *d_dummy_p;                    // device dummy vector for particle storage
  
  
  /*----------------------------- function body -------------------------*/
  
  // calculate number of particles that flow into the simulation
  if((*tin) < fpt) in = 1 + int((fpt-(*tin))/dtin);
  
  // copy number of particles from host to device 
  cuError = cudaMalloc((void **) &d_num_p, sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy (d_num_p, h_num_p, sizeof(int), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  
  // execution configuration for particle remover kernel
  griddim = 1;
  blockdim = P_RMV_BLK_SZ;

  // execute particle remover kernel
  cudaGetLastError();
  pRemover<<<griddim, blockdim>>>(*d_p, d_num_p, L);
  cu_sync_check(__FILE__, __LINE__);

  // copy new number of particles from device to host (and free device memory)
  cuError = cudaMemcpy (h_num_p, d_num_p, sizeof(int), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaFree(d_num_p);
  cu_check(cuError, __FILE__, __LINE__);

  // resize of particle vector with new number of particles
  cuError = cudaMalloc((void **) &d_dummy_p, ((*h_num_p)+in)*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(d_dummy_p, *d_p, (*h_num_p)*sizeof(particle), cudaMemcpyDeviceToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaFree(*d_p);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc((void **) d_p, ((*h_num_p)+in)*sizeof(particle));   
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(*d_p, d_dummy_p, (*h_num_p)*sizeof(particle), cudaMemcpyDeviceToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaFree(d_dummy_p);
  cu_check(cuError, __FILE__, __LINE__);
  
  // add particles
  if (in != 0) {
    // execution configuration for pEmi kernel
    griddim = 1;
    blockdim = CURAND_BLOCK_DIM;

    // launch kernel to add particles
    cudaGetLastError();
    pEmi<<<griddim, blockdim>>>(*d_p, *h_num_p, in, d_E, sqrt(kt/m), q/m, nn, L, fpt, fvt, *tin, dtin, state);
    cu_sync_check(__FILE__, __LINE__);

    // actualize time for next particle insertion
    (*tin) += double(in)*dtin;

    // actualize number of particles
    *h_num_p += in;
  }

  return;
}

/**********************************************************/


/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void pEmi(particle *g_p, int num_p, int n_in, double *g_E, double sigma, double qm, 
                     int nn, double L, double fpt, double fvt, double tin, double dtin, 
                     curandStatePhilox4_32_10_t *state)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ double sh_E;

  // kernel registers
  particle reg_p;
  int tid = (int) threadIdx.x + (int) blockIdx.x * (int) blockDim.x;
  int tpb = (int) blockDim.x;
  curandStatePhilox4_32_10_t local_state;
  double2 rnd;
 
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory
  if (tid == 0) sh_E = g_E[nn-1];
  __syncthreads();

  //---- initialize registers
  local_state = state[tid];
  __syncthreads();

  //---- generate particles
  for (int i = tid; i < n_in; i+=tpb) {
    // generate register particles
    reg_p.r = L;
    rnd = curand_normal2_double(&local_state);
    reg_p.v = -sqrt(rnd.x*rnd.x+rnd.y*rnd.y)*sigma;
    
    // simple push
    reg_p.r += (fpt-(tin+double(i)*dtin))*reg_p.v;
    reg_p.v += (fvt-(tin+double(i)*dtin))*sh_E*qm;

    // store new particles in global memory
    g_p[num_p+i] = reg_p;
  }
  __syncthreads();

  //---- store local state in global memory
  state[tid] = local_state;

  return;
}

/**********************************************************/

__global__ void pRemover (particle *g_p, int *num_p, double L)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ int g_tail;
  
  // kernel registers
  int tid = (int) threadIdx.x;
  int bdim = (int) blockDim.x;
  int N = *num_p;
  int ite = (N/bdim)*bdim;
  int reg_tail;
  particle reg_p;
 
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory
  if (tid == 0) g_tail = 0;
  __syncthreads();

  //---- analize full batches of particles
  for (int i = tid; i<ite; i+=bdim) {
    // load particles from global memory to registers
    reg_p = g_p[i];

    // analize particle
    if (reg_p.r >= 0 && reg_p.r <= L) {
      reg_tail = atomicAdd(&g_tail, 1);
    } else {
      reg_tail = -1;
    }
    __syncthreads();

    // store accepted particles in global memory
    if (reg_tail >= 0) g_p[reg_tail] = reg_p;

    __syncthreads();
  }
  __syncthreads();

  //---- analize last batch of particles
  if (g_tail+tid < N) {
    // loag particles from global memory to registers
    reg_p = g_p[g_tail+tid];

    // analize particle
    if (reg_p.r >= 0 && reg_p.r <= L) {
      reg_tail = atomicAdd(&g_tail, 1);
    } else {
      reg_tail = -1;
    }
  }
  __syncthreads();

  // store accepted particles of last batch in global memory
  if (g_tail+tid < N && reg_tail >= 0) g_p[g_tail+reg_tail] = reg_p;
  
  // store new number of particles in global memory
  if (tid == 0) *num_p = g_tail;
  
  return; 
}

/**********************************************************/
