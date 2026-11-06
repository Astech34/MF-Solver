#pragma once

#include "MF-Solver/basis_matrix.hpp"
#include "MF-Solver/dims.hpp"
#include "MF-Solver/observable.hpp"

struct scf_state {
    basis_matrix rho;         // Density matrix
    double mu = 0.0;          // Chemical potential
    int iteration = 0;
    double tolerance = 0.0;
    ScalarList constraint_params;

    explicit scf_state(const Dims& dims)
        : rho(dims), constraint_params(dims.num_layers, dims.num_sites, 0.0) {}
};

enum class MixerType { LinearDIIS };
