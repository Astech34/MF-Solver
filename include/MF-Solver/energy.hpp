#pragma once

#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/kspace.hpp"
#include "MF-Solver/model.hpp"
#include "MF-Solver/scf.hpp"

double calculate_band_energy(const Eigensystem& sys, double mu, double T);
double kanamori_dc(const hopping_model& hopping, const scf_state& scf, const kfree_params& kfp);
