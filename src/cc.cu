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

void cc (double t, int *num_e, particle **d_e, double *dtin_e, int *num_i, particle **d_i, double *dtin_i, 
         double *q_p, double *d_phi, double *d_E, curandStatePhilox4_32_10_t *state)
{
  /*--------------------------- function variables -----------------------*/

  // host memory
  static const double me = init_me();                 //
  static const double mi = init_mi();                 //
  static const double kte = init_kte();               // particle
  static const double kti = init_kti();               // properties
  static const double vd_e = init_vd_e();             //
  static const double vd_i = init_vd_i();             //
  
  static double tin_e = t+(*dtin_e);                  // time for next electron insertion
  static double tin_i = t+(*dtin_i);                  // time for next ion insertion

  static bool fp_is_on = floating_potential_is_on();  // probe is floating or not
  static int nc = init_nc();                          // number of cells
  static double ds = init_ds();                       // spatial step
  static double epsilon0 = init_epsilon0();           // epsilon0 in simulation units
  
  static const double phi_s = -0.5*init_mi()*init_vd_i()*init_vd_i();
  double dummy_phi_p;                                 // dummy probe potential

  cudaError cuError;                                  // cuda error variable

  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  //---- electrons contour conditions
  
  abs_emi_cc(t, &tin_e, *dtin_e, kte, vd_e, me, -1.0, q_p,  num_e, d_e, d_E, state);

  //---- ions contour conditions

  abs_emi_cc(t, &tin_i, *dtin_i, kti, vd_i, mi, +1.0, q_p, num_i, d_i, d_E, state);

  //---- actualize probe potential because of the change in probe charge
  if (fp_is_on) {
    dummy_phi_p = 0.5*(*q_p)*nc/(ds*epsilon0);
    if (dummy_phi_p > phi_s) dummy_phi_p = phi_s;
    cuError = cudaMemcpy (&d_phi[0], &dummy_phi_p, sizeof(double), cudaMemcpyHostToDevice);
    cu_check(cuError, __FILE__, __LINE__);
    recalculate_dtin_i(dtin_e, dtin_i, dummy_phi_p);
  }
  
  return;
}

/**********************************************************/

void abs_emi_cc(double t, double *tin, double dtin, double kt, double vd, double m, double q, double *q_p, 
                int *h_num_p, particle **d_p, double *d_E, curandStatePhilox4_32_10_t *state)
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
  int h_num_abs_p;                        // host number of particles absorved at the probe
  
  cudaError cuError;                      // cuda error variable
  dim3 griddim, blockdim;                 // kernel execution configurations 

  // device memory
  int *d_num_p;                           // device number of particles
  int *d_num_abs_p;                       // device number of particles absorved at the probe
  particle *d_dummy_p;                    // device dummy vector for particle storage
  
  /*----------------------------- function body -------------------------*/
  
  // calculate number of particles that flow into the simulation
  if((*tin) < fpt) in = 1 + int((fpt-(*tin))/dtin);
  
  // copy number of particles from host to device 
  cuError = cudaMalloc((void **) &d_num_p, sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy (d_num_p, h_num_p, sizeof(int), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  
  // initialize number of particles absorbed at the probe 
  cuError = cudaMalloc((void **) &d_num_abs_p, sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemset((void *) d_num_abs_p, 0, sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  
  // execution configuration for particle remover kernel
  griddim = 1;
  blockdim = P_RMV_BLK_SZ;

  // execute particle remover kernel
  cudaGetLastError();
  pRemover<<<griddim, blockdim>>>(*d_p, d_num_p, L, d_num_abs_p);
  cu_sync_check(__FILE__, __LINE__);

  // copy number of particles absorbed at the probe from device to host (and free device memory)
  cuError = cudaMemcpy (&h_num_abs_p, d_num_abs_p, sizeof(int), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaFree(d_num_abs_p);
  cu_check(cuError, __FILE__, __LINE__);

  // actualize probe acumulated charge
  *q_p += q*h_num_abs_p;

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
    pEmi<<<griddim, blockdim>>>(*d_p, *h_num_p, in, d_E, sqrt(kt/m), vd, q/m, nn, L, fpt, fvt, *tin, dtin, state);
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

__global__ void pEmi(particle *g_p, int num_p, int n_in, double *g_E, double vth, double vd, double qm, int nn, 
                     double L, double fpt, double fvt, double tin, double dtin, curandStatePhilox4_32_10_t *state)
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
    if (vth > 0.0) {
      rnd = curand_normal2_double(&local_state);
      reg_p.v = -sqrt(rnd.x*rnd.x+rnd.y*rnd.y)*vth-vd;
    } else reg_p.v = -vd;
    
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

__global__ void pRemover (particle *g_p, int *g_num_p, double L, int *g_num_abs_p)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ int sh_tail;
  __shared__ int sh_num_abs_p;
  
  // kernel registers
  int tid = (int) threadIdx.x;
  int bdim = (int) blockDim.x;
  int N = *g_num_p;
  int ite = (N/bdim)*bdim;
  int reg_tail;
  particle reg_p;
 
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory
  if (tid == 0) {
    sh_tail = 0;
    sh_num_abs_p = 0;
  }
  __syncthreads();

  //---- analize full batches of particles
  for (int i = tid; i<ite; i+=bdim) {
    // load particles from global memory to registers
    reg_p = g_p[i];

    // analize particle
    if (reg_p.r >= 0 && reg_p.r <= L) {
      reg_tail = atomicAdd(&sh_tail, 1);
    } else {
      reg_tail = -1;
      if (reg_p.r < 0.0) atomicAdd(&sh_num_abs_p, 1);
    }
    __syncthreads();

    // store accepted particles in global memory
    if (reg_tail >= 0) g_p[reg_tail] = reg_p;

    __syncthreads();
  }
  __syncthreads();

  //---- analize last batch of particles
  if (ite+tid < N) {
    // loag particles from global memory to registers
    reg_p = g_p[ite+tid];

    // analize particle
    if (reg_p.r >= 0 && reg_p.r <= L) {
      reg_tail = atomicAdd(&sh_tail, 1);
    } else {
      reg_tail = -1;
      if (reg_p.r < 0.0) atomicAdd(&sh_num_abs_p, 1);
    }
  }
  __syncthreads();

  // store accepted particles of last batch in global memory
  if (ite+tid < N && reg_tail >= 0) g_p[reg_tail] = reg_p;
  
  // store new number of particles in global memory
  if (tid == 0) {
    *g_num_p = sh_tail;
    *g_num_abs_p = sh_num_abs_p;
  }
  
  return; 
}

/**********************************************************/
