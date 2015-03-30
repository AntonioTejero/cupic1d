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

void cc (double t, int *num_e, particle **d_e, double *dtin_e, int *num_he, particle **d_he, double *dtin_he, 
         int *num_i, particle **d_i, double *dtin_i, double *vd_i, int *num_se, particle **d_se, double *q_p,
         double *d_phi, double *d_E, curandStatePhilox4_32_10_t *state)
{
  /*--------------------------- function variables -----------------------*/

  // host memory
  static const double me = init_me();                       //
  static const double mi = init_mi();                       //
  static const double kte = init_kte();                     // 
  static const double kti = init_kti();                     // particle 
  static const double ktse = init_ktse();                   // properties 
  static const double kthe = init_ktse();                   // 
  static const double vd_e = init_vd_e();                   // 
  static const double vd_se = init_vd_se();                 // 
  static const double vd_he = init_vd_se();                 // 
  
  static const bool fp_is_on = floating_potential_is_on();  // probe is floating or not
  static const bool flux_cal_is_on = calibration_is_on();   // probe is floating or not
  static const int nc = init_nc();                          // number of cells
  static const double a_p = init_a_p();                     // area of the probe 
  static const double L = init_L();                         // lenght of the simulation
  static const double epsilon0 = init_epsilon0();           // epsilon0 in simulation units
  static const double dtin_se = init_dtin_se();             // time between secondary electron insertions 

  static double tin_e = t+(*dtin_e);                        // time for next electron insertion
  static double tin_he = t+(*dtin_he);                      // time for next hot electron insertion
  static double tin_i = t+(*dtin_i);                        // time for next ion insertion
  static double tin_se = t+dtin_se;                         // time for next secondary electron insertion
  
  double phi_s;                                             // sheath edge potential 
  double phi_p;                                             // probe potential

  cudaError cuError;                                        // cuda error variable

  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  //---- electrons contour conditions
  
  abs_emi_cc(t, &tin_e, *dtin_e, kte, vd_e, me, -1.0, q_p,  num_e, d_e, L, d_E, state);

  //---- hot electrons contour conditions
  
  abs_emi_cc(t, &tin_he, *dtin_he, kthe, vd_he, me, -1.0, q_p, num_he, d_he, L, d_E, state);

  //---- ions contour conditions

  abs_emi_cc(t, &tin_i, *dtin_i, kti, *vd_i, mi, +1.0, q_p, num_i, d_i, L, d_E, state);
  
  //---- secondary electrons contour conditions
  
  abs_emi_cc(t, &tin_se, dtin_se, ktse, vd_se, me, -1.0, q_p, num_se, d_se, 0.0, d_E, state);

  //---- copy probe and sheath edge potentials in host memory in case fp or flux_cal are on
  if (fp_is_on || flux_cal_is_on) {
    cuError = cudaMemcpy (&phi_p, &d_phi[0], sizeof(double), cudaMemcpyDeviceToHost);
    cu_check(cuError, __FILE__, __LINE__);
    cuError = cudaMemcpy (&phi_s, &d_phi[nc], sizeof(double), cudaMemcpyDeviceToHost);
    cu_check(cuError, __FILE__, __LINE__);
  }
  
  //---- actulize ion drift velocity in order to ensure zero field at sheath edge
  if (flux_cal_is_on) {
    calibrate_ion_flux(vd_i, d_E, &phi_s);
  }

  //---- actualize probe potential because of the change in probe charge
  if (fp_is_on) {
    phi_p = 0.5*(*q_p)*L/(a_p*epsilon0);
    if (phi_p > phi_s) phi_p = phi_s;
  }
  
  //---- store new probe and sheath edge potentials in d_phi and recalculate electron and ion dtin 
  if (fp_is_on || flux_cal_is_on) {
    cuError = cudaMemcpy (&d_phi[0], &phi_p, sizeof(double), cudaMemcpyHostToDevice);
    cu_check(cuError, __FILE__, __LINE__);
    cuError = cudaMemcpy (&d_phi[nc], &phi_s, sizeof(double), cudaMemcpyHostToDevice);
    cu_check(cuError, __FILE__, __LINE__);
    recalculate_dtin(dtin_e, dtin_he, dtin_i, *vd_i, phi_p, phi_s);
  }
 
  return;
}

/**********************************************************/

void abs_emi_cc(double t, double *tin, double dtin, double kt, double vd, double m, double q, double *q_p, 
                int *h_num_p, particle **d_p, double pos, double *d_E, curandStatePhilox4_32_10_t *state)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double L = init_L();       // geometric properties
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
    pEmi<<<griddim, blockdim>>>(*d_p, *h_num_p, in, d_E, sqrt(kt/m), vd, q/m, nn, pos, fpt, fvt, *tin, dtin, state);
    cu_sync_check(__FILE__, __LINE__);

    // actualize time for next particle insertion
    (*tin) += double(in)*dtin;

    // actualize number of particles
    *h_num_p += in;
  }

  return;
}

/**********************************************************/

void recalculate_dtin(double *dtin_e, double *dtin_he, double *dtin_i, double vd_i, double phi_p, double phi_s)
{
  /*--------------------------- function variables -----------------------*/

  // host memory
  static const double n = init_n();
  static const double alpha = init_alpha();
  static const double a_p = init_a_p();
  static const double me = init_me();
  static const double kte = init_kte();
  static const double vd_e = init_vd_e();
  static const double kthe = init_kthe();
  static const double vd_he = init_vd_he();
  static const double mi = init_mi();
  static const double kti = init_kti();
  
  // device memory
  
  /*----------------------------- function body -------------------------*/

  //---- recalculate electron dtin
  *dtin_e = n*sqrt(kte/(2.0*PI*me))*exp(-0.5*me*vd_e*vd_e/kte);  // thermal component of input flux
  *dtin_e -= 0.5*n*vd_e*(1.0+erf(sqrt(0.5*me/kte)*(-vd_e)));     // drift component of input flux
  *dtin_e *= exp(phi_s)*0.5*(1.0+erf(sqrt(phi_s-phi_p)));        // correction on density at sheath edge
  *dtin_e *= a_p;                                                // number of particles that enter the simulation per unit of time
  *dtin_e = 1.0/(*dtin_e);                                       // time between consecutive particles injection

  //---- recalculate hot electron dtin
  *dtin_he = alpha*n*sqrt(kthe/(2.0*PI*me))*exp(-0.5*me*vd_he*vd_he/kthe);  // thermal component of input flux
  *dtin_he -= 0.5*alpha*n*vd_he*(1.0+erf(sqrt(0.5*me/kthe)*(-vd_he)));      // drift component of input flux
  *dtin_he *= exp(phi_s)*0.5*(1.0+erf(sqrt(phi_s-phi_p)));                  // correction on density at sheath edge
  *dtin_he *= a_p;                                                // number of particles that enter the simulation per unit of time
  *dtin_he = 1.0/(*dtin_he);                                      // time between consecutive particles injection

  //---- recalculate ion dtin
  *dtin_i = (alpha+1.0)*n*sqrt(kti/(2.0*PI*mi))*exp(-0.5*mi*vd_i*vd_i/kti);  // thermal component of input flux
  *dtin_i -= 0.5*(alpha+1.0)*n*vd_i*(1.0+erf(sqrt(0.5*mi/kti)*(-vd_i)));     // drift component of input flux
  *dtin_i *= exp(phi_s)*0.5*(1.0+erf(sqrt(phi_s-phi_p)));        // correction on density at sheath edge
  *dtin_i *= a_p;                                                // number of particles that enter the simulation per unit of time
  *dtin_i = 1.0/(*dtin_i);                                       // time between consecutive particles injection

  return;
}

/**********************************************************/

void calibrate_ion_flux(double *vd_i, double *d_E, double *phi_s)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double mi = init_mi();
  static const int nc = init_nc();

  double E_mean;
  double *h_E;
  const double increment = 1.0e-6;
 
  cudaError cuError;                            // cuda error variable

  // device memory
  
  /*----------------------------- function body -------------------------*/
 
  //---- Actualize ion drift velocity acording to the value of electric field at plasma frontier

  // allocate host memory for field
  h_E = (double*) malloc(5*sizeof(double));
  
  // copy field from device to host memory
  cuError = cudaMemcpy (h_E, &d_E[nc-5], 5*sizeof(double), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);

  // check mean value of electric field at plasma frontier
  E_mean = 0.0;
  for (int i=0; i<=5; i++) {
    E_mean += h_E[i];
  }
  E_mean /= 5.0;
 
  // free host memory for field
  free(h_E);

  // actualize ion drift velocity
  if (E_mean<0 && *vd_i > -1.0/sqrt(mi)) {
    *vd_i -= increment;
  } else if (E_mean>0 && *vd_i < 0.0) {
    *vd_i += increment;
  }

  // actualize sheath edge potential
  *phi_s = -0.5*mi*(*vd_i)*(*vd_i);
    
  return;
}

/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void pEmi(particle *g_p, int num_p, int n_in, double *g_E, double vth, double vd, double qm, int nn, 
                     double pos, double fpt, double fvt, double tin, double dtin, curandStatePhilox4_32_10_t *state)
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
    reg_p.r = pos;
    if (vth > 0.0) {
      rnd = curand_normal2_double(&local_state);
      if (pos > 0.0) {
        reg_p.v = -sqrt(rnd.x*rnd.x+rnd.y*rnd.y)*vth+vd;
      } else {
        reg_p.v = sqrt(rnd.x*rnd.x+rnd.y*rnd.y)*vth+vd;
      }
    } else reg_p.v = vd;
    
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
