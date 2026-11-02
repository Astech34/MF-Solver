#pragma once
#include "MF-Solver/model.hpp"
#include "MF-Solver/hamiltonian.hpp"
#include <string>
#include <vector>

void save_bands(const hopping_model& model, const kfree_params& kfp, int num_points, const std::string& filename);
