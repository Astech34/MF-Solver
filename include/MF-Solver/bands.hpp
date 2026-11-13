#pragma once

#include <string>

#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/model.hpp"

void save_bands(const hopping_model& model, const scf_state& scf_state, const kfree_params& kfp, int num_points, const std::string& filename);

// Gaussian-broadened DOS (per unit cell) on a grid_size x grid_size k-grid.
// CSV columns: energy, mu, dos. grid_size must be even.
void save_dos(const hopping_model& model, const scf_state& scf, const kfree_params& kfp,
              int grid_size, int num_energies, double sigma, const std::string& filename);
