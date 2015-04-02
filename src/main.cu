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
  const double dt = init_dt();          // time step
  const int n_ini = init_n_ini();       // number of first iteration
  const int n_prev = init_n_prev();     // number of iterations before start analizing
  const int n_save = init_n_save();     // number of iterations between diagnostics
  const int n_fin = init_n_fin();       // number of last iteration
  const int nn = init_nn();             // number of nodes
  double t;                             // time of simulation
  int num_e, num_i, num_se, num_he;     // number of particles (electrons and ions)
  double U_e, U_i, U_se, U_he;          // system energy for electrons and ions
  double mi = init_mi();                // ion mass
  double vd_i = init_vd_i();            // drift velocity of ions 
  double q_p = 0;                       // probe's acumulated charge
  char filename[50];                    // filename for saved data

  ifstream ifile;
  ofstream ofile;

  // device variables definition
  double *d_rho, *d_phi, *d_E;              // mesh properties
  double *d_avg_rho, *d_avg_phi, *d_avg_E;  // mesh averaged properties
  double *d_avg_ddf_i, *d_avg_vdf_i;        // density and velocity distribution function for ions
  double *d_avg_ddf_e, *d_avg_vdf_e;        // density and velocity distribution function for electrons
  double *d_avg_ddf_se, *d_avg_vdf_se;      // density and velocity distribution function for secondary electrons
  double *d_avg_ddf_he, *d_avg_vdf_he;      // density and velocity distribution function for hot electrons
  double v_max_i = init_v_max_i();          // maximun velocity of ions (for histograms)
  double v_min_i = init_v_min_i();          // minimun velocity of ions (for histograms)
  double v_max_e = init_v_max_e();          // maximun velocity of electrons (for histograms)
  double v_min_e = init_v_min_e();          // minimun velocity of electrons (for histograms)
  double v_max_se = init_v_max_se();        // maximun velocity of secondary electrons (for histograms)
  double v_min_se = init_v_min_se();        // minimun velocity of secondary electrons (for histograms)
  double v_max_he = init_v_max_he();        // maximun velocity of hot electrons (for histograms)
  double v_min_he = init_v_min_he();        // minimun velocity of hot electrons (for histograms)
  int count_df_e = 0;                       // |
  int count_df_i = 0;                       // |
  int count_df_se = 0;                      // |
  int count_df_he = 0;                      // |
  int count_rho = 0;                        // |-> counters for avg data
  int count_phi = 0;                        // |
  int count_E = 0;                          // |
  particle *d_e, *d_i, *d_se, *d_he;        // particles vectors
  curandStatePhilox4_32_10_t *state;        // philox state for __device__ random number generation 

  /*----------------------------- function body -------------------------*/

  //---- INITIALITATION OF SIMULATION

  // initialize device and simulation variables
  init_dev();
  init_sim(&t, &d_rho, &d_phi, &d_E, &d_avg_rho, &d_avg_phi, &d_avg_E, &d_e, &num_e, &d_i, &num_i, &d_se, &num_se, &d_he, 
           &num_he, &d_avg_ddf_e, &d_avg_vdf_e, &d_avg_ddf_i, &d_avg_vdf_i, &d_avg_ddf_se, &d_avg_vdf_se, &d_avg_ddf_he, 
           &d_avg_vdf_he, &state);

  // save initial state
  sprintf(filename, "../output/particles/electrons_t_%d", n_ini);
  particles_snapshot(d_e, num_e, filename);
  sprintf(filename, "../output/particles/ions_t_%d", n_ini);
  particles_snapshot(d_i, num_i, filename);
  sprintf(filename, "../output/particles/selectrons_t_%d", n_ini);
  particles_snapshot(d_se, num_se, filename);
  sprintf(filename, "../output/particles/helectrons_t_%d", n_ini);
  particles_snapshot(d_he, num_he, filename);
  sprintf(filename, "../output/charge/avg_charge_t_%d", n_ini);
  save_mesh(d_avg_rho, filename);
  sprintf(filename, "../output/potential/avg_potential_t_%d", n_ini);
  save_mesh(d_avg_phi, filename);
  sprintf(filename, "../output/field/avg_field_t_%d", n_ini);
  save_mesh(d_avg_E, filename);
  t += dt;

  //---- SIMULATION BODY
  
  for (int i = n_ini+1; i <= n_fin; i++, t += dt) {
    // simulate one time step
    charge_deposition(d_rho, d_e, num_e, d_i, num_i, d_se, num_se, d_he, num_he);
    poisson_solver(1.0e-4, d_rho, d_phi);
    field_solver(d_phi, d_E);
    particle_mover(d_e, num_e, d_i, num_i, d_se, num_se, d_he, num_he, d_E);
    cc(t, &num_e, &d_e, &num_he, &d_he, &num_i, &d_i, &vd_i, &num_se, &d_se, &q_p, d_phi, d_E, state);

    // average mesh variables and distribution functions
    avg_mesh(d_rho, d_avg_rho, &count_rho);
    avg_mesh(d_phi, d_avg_phi, &count_phi);
    avg_mesh(d_E, d_avg_E, &count_E);
    eval_df(d_avg_ddf_e, d_avg_vdf_e, v_max_e, v_min_e, d_e, num_e, &count_df_e);
    eval_df(d_avg_ddf_i, d_avg_vdf_i, v_max_i, v_min_i, d_i, num_i, &count_df_i);
    eval_df(d_avg_ddf_se, d_avg_vdf_se, v_max_se, v_min_se, d_se, num_se, &count_df_se);
    eval_df(d_avg_ddf_he, d_avg_vdf_he, v_max_he, v_min_he, d_he, num_he, &count_df_he);

    // store data
    if (i>=n_prev && i%n_save==0) {
      // save particles (snapshot)
      sprintf(filename, "../output/particles/electrons_t_%d", i);
      particles_snapshot(d_e, num_e, filename);
      sprintf(filename, "../output/particles/ions_t_%d", i);
      particles_snapshot(d_i, num_i, filename);
      sprintf(filename, "../output/particles/selectrons_t_%d", i);
      particles_snapshot(d_se, num_se, filename);
      sprintf(filename, "../output/particles/helectrons_t_%d", i);
      particles_snapshot(d_he, num_he, filename);

      // save mesh properties
      sprintf(filename, "../output/charge/avg_charge_t_%d", i);
      save_mesh(d_avg_rho, filename);
      sprintf(filename, "../output/potential/avg_potential_t_%d", i);
      save_mesh(d_avg_phi, filename);
      sprintf(filename, "../output/field/avg_field_t_%d", i);
      save_mesh(d_avg_E, filename);

      // save distribution functions
      sprintf(filename, "../output/particles/electrons_ddf_t_%d", i);
      save_ddf(d_avg_ddf_e, filename);
      sprintf(filename, "../output/particles/ions_ddf_t_%d", i);
      save_ddf(d_avg_ddf_i, filename);
      sprintf(filename, "../output/particles/selectrons_ddf_t_%d", i);
      save_ddf(d_avg_ddf_se, filename);
      sprintf(filename, "../output/particles/helectrons_ddf_t_%d", i);
      save_ddf(d_avg_ddf_he, filename);
      sprintf(filename, "../output/particles/electrons_vdf_t_%d", i);
      save_vdf(d_avg_vdf_e, v_max_e, v_min_e, filename);
      sprintf(filename, "../output/particles/ions_vdf_t_%d", i);
      save_vdf(d_avg_vdf_i, v_max_i, v_min_i, filename);
      sprintf(filename, "../output/particles/selectrons_vdf_t_%d", i);
      save_vdf(d_avg_vdf_se, v_max_se, v_min_se, filename);
      sprintf(filename, "../output/particles/helectrons_vdf_t_%d", i);
      save_vdf(d_avg_vdf_he, v_max_he, v_min_he, filename);

      // save log
      U_e = eval_particle_energy(d_phi,  d_e, 1.0, -1.0, num_e);
      U_i = eval_particle_energy(d_phi,  d_i, mi, 1.0, num_i);
      U_se = eval_particle_energy(d_phi,  d_se, 1.0, -1.0, num_se);
      U_he = eval_particle_energy(d_phi,  d_he, 1.0, -1.0, num_he);
      save_log(t, num_e, num_i, num_se, num_he, U_e, U_i, U_se, U_he, vd_i, d_phi);
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
