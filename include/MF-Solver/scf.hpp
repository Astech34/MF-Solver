#pragma once
#include <Eigen/Dense>
#include <complex>

#include "MF-Solver/model.hpp"
#include "MF-Solver/basis_matrix.hpp"

struct scf_state {
    basis_matrix rho;     // Density matrix
    double mu = 0.0;          // Chemical potential
    int iteration = 0;
    double tolerance = 0.0;

    explicit scf_state(const hopping_model& hopping)
        : rho(hopping) {}
};
