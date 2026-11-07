#pragma once

#include <Eigen/Dense>
#include <vector>

#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/model.hpp"
#include "MF-Solver/scf.hpp"

// Eigenvalues / eigenvectors of H(k) on a k-grid (one entry per k-point)
struct Eigensystem {
    std::vector<Eigen::VectorXd>  evals;
    std::vector<Eigen::MatrixXcd> evecs;  // columns are eigenvectors
};

// Diagonalize H(k) = H0(k) + H_local[scf] on a grid_size x grid_size k-grid
Eigensystem compute_eigensystem_grid(const hopping_model& model, const scf_state& scf,
                                     const kfree_params& kfp, int grid_size);
