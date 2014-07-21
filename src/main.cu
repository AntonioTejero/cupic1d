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
  int nn = init_nn();                   // number of nodes
  double U_e, U_i;                      // system energy for electrons and ions
  double mi = init_mi();                // ion mass
  double dtin_i = init_dtin_i();        // time between ion insertion
  char filename[50];                    // filename for saved data

  double foo;
  ifstream ifile;
  ofstream ofile;
  cudaError_t cuError;

  // device variables definition
  double *d_rho, *d_phi, *d_E;              // mesh properties
  double *d_avg_rho, *d_avg_phi, *d_avg_E;  // mesh averaged properties
  double *d_avg_ddf_e, *d_avg_vdf_e;        // density and velocity distribution function for electrons
  double v_max_e = init_v_max_e();          // maximun velocity of electrons (for histograms)
  double v_min_e = init_v_min_e();          // minimun velocity of electrons (for histograms)
  double *d_avg_ddf_i, *d_avg_vdf_i;        // density and velocity distribution function for ions
  double v_max_i = init_v_max_i();          // maximun velocity of ions (for histograms)
  double v_min_i = init_v_min_i();          // minimun velocity of ions (for histograms)
  int count_df_e = 0;                       // |
  int count_df_i = 0;                       // |
  int count_rho = 0;                        // |-> counters for avg data
  int count_phi = 0;                        // |
  int count_E = 0;                          // |
  particle *d_e, *d_i;                      // particles vectors
  curandStatePhilox4_32_10_t *state;        // philox state for __device__ random number generation 

  /*----------------------------- function body -------------------------*/

  //---- INITIALITATION OF SIMULATION

  // initialize device and simulation variables
  init_dev();
  init_sim(&d_rho, &d_phi, &d_E, &d_avg_rho, &d_avg_phi, &d_avg_E, &d_e, &num_e, &d_i, &num_i, 
           &d_avg_ddf_e, &d_avg_vdf_e, &d_avg_ddf_i, &d_avg_vdf_i, &t, &state);

  //---- CALIBRATION OF ION CURRENT

  if (calibration_is_on()) {
    cout << "Starting calibration of dtin_i parameter..." << endl;
    for (int i = n_ini+1; i <= n_fin; i++, t += dt) {
      // simulate one time step
      charge_deposition(d_rho, d_e, num_e, d_i, num_i);
      poisson_solver(1.0e-4, d_rho, d_phi);
      field_solver(d_phi, d_E);
      particle_mover(d_e, num_e, d_i, num_i, d_E);
      cc(t, &num_e, &d_e, &num_i, &d_i, dtin_i, d_E, state);
      
      // average mesh variables and distribution functions
      avg_mesh(d_rho, d_avg_rho, &count_rho);
      avg_mesh(d_phi, d_avg_phi, &count_phi);
      avg_mesh(d_E, d_avg_E, &count_E);
      eval_df(d_avg_ddf_e, d_avg_vdf_e, v_max_e, v_min_e, d_e, num_e, &count_df_e);
      eval_df(d_avg_ddf_i, d_avg_vdf_i, v_max_i, v_min_i, d_i, num_i, &count_df_i);
 
      cout << " t = " << t << endl;

      // store data
      if (i>=n_prev && i%n_save==0) {
        // save particles (snapshot)
        sprintf(filename, "../output/particles/electrons_t_%d", i);
        particles_snapshot(d_e, num_e, filename);
        sprintf(filename, "../output/particles/ions_t_%d", i);
        particles_snapshot(d_i, num_i, filename);

        // save mesh properties
        sprintf(filename, "../output/charge/avg_charge_t_%d", i-1);
        save_mesh(d_avg_rho, filename);
        sprintf(filename, "../output/potential/avg_potential_t_%d", i-1);
        save_mesh(d_avg_phi, filename);
        sprintf(filename, "../output/field/avg_field_t_%d", i-1);
        save_mesh(d_avg_E, filename);

        // save distribution functions
        sprintf(filename, "../output/particles/electrons_ddf_t_%d", i-1);
        save_ddf(d_avg_ddf_e, filename);
        sprintf(filename, "../output/particles/ions_ddf_t_%d", i-1);
        save_ddf(d_avg_ddf_i, filename);
        sprintf(filename, "../output/particles/electrons_vdf_t_%d", i-1);
        save_vdf(d_avg_vdf_e, v_max_e, v_min_e, filename);
        sprintf(filename, "../output/particles/ions_vdf_t_%d", i-1);
        save_vdf(d_avg_vdf_i, v_max_i, v_min_i, filename);

        // save log
        U_e = eval_particle_energy(d_phi,  d_e, 1.0, -1.0, num_e);
        U_i = eval_particle_energy(d_phi,  d_i, mi, 1.0, num_i);
        save_log(t, num_e, num_i, U_e, U_i, dtin_i);

        // calibrate ion current
        cuError = cudaMemcpy (&foo, d_avg_phi+nn-1, sizeof(double), cudaMemcpyDeviceToHost);
        cu_check(cuError, __FILE__, __LINE__);
        calibrate_dtin_i(&dtin_i, foo > -5.0e-3);
      }
    }
  }

  //---- SIMULATION BODY
  
  // save initial state
  sprintf(filename, "../output/particles/electrons_t_%d", n_ini);
  particles_snapshot(d_e, num_e, filename);
  sprintf(filename, "../output/particles/ions_t_%d", n_ini);
  particles_snapshot(d_i, num_i, filename);
  t += dt;

  for (int i = n_ini+1; i <= n_fin; i++, t += dt) {
    // simulate one time step
    charge_deposition(d_rho, d_e, num_e, d_i, num_i);
    poisson_solver(1.0e-4, d_rho, d_phi);
    field_solver(d_phi, d_E);
    particle_mover(d_e, num_e, d_i, num_i, d_E);
    cc(t, &num_e, &d_e, &num_i, &d_i, dtin_i, d_E, state);

    // average mesh variables and distribution functions
    avg_mesh(d_rho, d_avg_rho, &count_rho);
    avg_mesh(d_phi, d_avg_phi, &count_phi);
    avg_mesh(d_E, d_avg_E, &count_E);
    eval_df(d_avg_ddf_e, d_avg_vdf_e, v_max_e, v_min_e, d_e, num_e, &count_df_e);
    eval_df(d_avg_ddf_i, d_avg_vdf_i, v_max_i, v_min_i, d_i, num_i, &count_df_i);

    // store data
    if (i>=n_prev && i%n_save==0) {
      // save particles (snapshot)
      sprintf(filename, "../output/particles/electrons_t_%d", i);
      particles_snapshot(d_e, num_e, filename);
      sprintf(filename, "../output/particles/ions_t_%d", i);
      particles_snapshot(d_i, num_i, filename);

      // save mesh properties
      sprintf(filename, "../output/charge/avg_charge_t_%d", i-1);
      save_mesh(d_avg_rho, filename);
      sprintf(filename, "../output/potential/avg_potential_t_%d", i-1);
      save_mesh(d_avg_phi, filename);
      sprintf(filename, "../output/field/avg_field_t_%d", i-1);
      save_mesh(d_avg_E, filename);

      // save distribution functions
      sprintf(filename, "../output/particles/electrons_ddf_t_%d", i-1);
      save_ddf(d_avg_ddf_e, filename);
      sprintf(filename, "../output/particles/ions_ddf_t_%d", i-1);
      save_ddf(d_avg_ddf_i, filename);
      sprintf(filename, "../output/particles/electrons_vdf_t_%d", i-1);
      save_vdf(d_avg_vdf_e, v_max_e, v_min_e, filename);
      sprintf(filename, "../output/particles/ions_vdf_t_%d", i-1);
      save_vdf(d_avg_vdf_i, v_max_i, v_min_i, filename);

      // save log
      U_e = eval_particle_energy(d_phi,  d_e, 1.0, -1.0, num_e);
      U_i = eval_particle_energy(d_phi,  d_i, mi, 1.0, num_i);
      save_log(t, num_e, num_i, U_e, U_i, dtin_i);
    }
  }

  //---- END OF SIMULATION

  // update input data file and finish simulation
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
