#pragma once

#include <string>

#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/model.hpp"

void save_bands(const hopping_model& model, const kfree_params& kfp, int num_points, const std::string& filename);
