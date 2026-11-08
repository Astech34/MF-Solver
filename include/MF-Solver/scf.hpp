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
    double energy = 0.0;
    bool isConverged = false;

    explicit scf_state(const Dims& dims)
        : rho(dims), constraint_params(dims.num_layers, dims.num_sites, 0.0) {}
};

enum class MixerType { Linear , DIIS};

// Forward declarations (hamiltonian.hpp includes this header)
struct hopping_model;
struct kfree_params;

struct KanamoriResult {
    scf_state state;
    double energy;
    bool converged;
};

// Self-consistent Kanamori mean-field loop. Currently only linear mixing is
// implemented; `mixer` is accepted for future use.
KanamoriResult runKanamoriSCF(const hopping_model& hopping, const scf_state& scf,
                              const kfree_params& kfp,
                              double alpha, int grid_size,
                              double T, double N_target,
                              MixerType mixer, int max_iter, bool comp_dft);
