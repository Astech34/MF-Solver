#pragma once
#include "MF-Solver/eigensystem.hpp"
#include "MF-Solver/dims.hpp"
#include "MF-Solver/basis_matrix.hpp"

basis_matrix compute_density_matrix(const Eigensystem& sys, const Dims dims, double mu, double T);