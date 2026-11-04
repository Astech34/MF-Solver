#pragma once

#include "MF-Solver/basis_matrix.hpp"
#include "MF-Solver/dims.hpp"

struct scf_state {
    basis_matrix rho;         // Density matrix
    double mu = 0.0;          // Chemical potential
    int iteration = 0;
    double tolerance = 0.0;

    explicit scf_state(const Dims& dims)
        : rho(dims) {}
};
