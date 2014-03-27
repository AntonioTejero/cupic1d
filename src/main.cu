/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


/****************************** HEADERS ******************************/

#include "stdh.h"
#include "init.h"
#include "cc.h"
#include "mesh.h"
#include "particles.h"
#include "diagnostic.h"

/************************ FUNCTION PROTOTIPES *************************/




/*************************** MAIN FUNCTION ****************************/

int main (int argc, const char* argv[])
{
  /*--------------------------- function variables -----------------------*/
  
  // host variables definition
  double t;                             // time of simulation
  const double dt = init_dt();          // time step
  const int n_ini = init_n_ini();       // number of first iteration
  const int n_prev = init_n_prev();     // number of iterations before start analizing
  const int n_save = init_n_save();     // number of iterations between diagnostics
  const int n_fin = init_n_fin();       // number of last iteration
  int num_e, num_i;                     // number of particles (electrons and ions)
  char filename[50];                    // filename for saved data
  ifstream ifile;
  ofstream ofile;

  // device variables definition
  double *d_rho, *d_phi, *d_E;          // mesh properties
  particle *d_e, *d_i;                  // particles vectors
  curandStatePhilox4_32_10_t *state;    // philox state for __device__ random number generation 

  /*----------------------------- function body -------------------------*/

  // initialize device and simulation
  init_dev();
  init_sim(&d_rho, &d_phi, &d_E, &d_e, &num_e, &d_i, &num_i, &t, &state);

  cout << "t = " << t << endl;
  sprintf(filename, "../output/particles/electrons_t_%d", n_ini);
  particles_snapshot(d_e, filename, t);
  sprintf(filename, "../output/particles/ions_t_%d", n_ini);
  particles_snapshot(d_i, filename, t);
  t += dt;

  for (int i = n_ini+1; i <= n_fin; i++, t += dt) {
    // deposit charge into the mesh nodes
    charge_deposition(d_rho, d_e, d_i);
    cout << "Charge deposited" << endl;
    
    // solve poisson equation
    poisson_solver(1.0e-4, d_rho, d_phi);
    cout << "Poisson eq. solved" << endl;
    
    // derive electric fields from potential
    field_solver(d_phi, d_E);
    cout << "Fields soved" << endl;
    
    // move particles
    particle_mover(d_e, d_i, d_E);
    cout << "Particles moved" << endl;

    // contour condition
    cc(t, &d_e, &d_i, d_E, state);
    cout << "Contour conditions applied" << endl;

    // store data
    if (i>=n_prev && i%n_save==0) {
      sprintf(filename, "../output/particles/electrons_t_%d", i);
      particles_snapshot(d_e, filename, t);
      sprintf(filename, "../output/particles/ions_t_%d", i);
      particles_snapshot(d_i, filename, t);
      sprintf(filename, "../output/charge/charge_t_%d", i-1);
      mesh_snapshot(d_rho, filename);
      sprintf(filename, "../output/potential/potential_t_%d", i-1);
      mesh_snapshot(d_phi, filename);
      sprintf(filename, "../output/particles/bm_electrons_t_%d", i);
      save_bm(d_e_bm, filename);
      sprintf(filename, "../output/particles/bins_electrons_t_%d", i);
      save_bins(d_e_bm, d_e, filename);
    }
     
    // print simulation time
    cout << "t = " << t << endl;
  }

  ifile.open("../input/input_data");
  ofile.open("../input/input_data_new");
  if (ifile.is_open() && ofile.is_open()) {
    ifile.getline(filename, 50);
    ofile << filename << endl;
    ifile.getline(filename, 50);
    ofile << "n_ini = " << n_fin << ";" << endl;
    ifile.getline(filename, 50);
    while (!ifile.eof()) {
      ofile << filename << endl;
      ifile.getline(filename, 50);
    }
  }
  ifile.close();
  ofile.close();
  system("mv ../input/input_data_new ../input/input_data");
  
  cout << "Simulation finished!" << endl;
  return 0;
}
