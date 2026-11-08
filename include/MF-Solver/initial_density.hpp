#pragma once

#include <Eigen/Dense>

#include "MF-Solver/basis_matrix.hpp"
#include <random>

enum class rho_init_kind { unpolarized_hfill , random_seed};

// Maybe useful later to add for random noise, etc.
struct rho_init_params {
    rho_init_kind kind = rho_init_kind::unpolarized_hfill;
    int seed = 42;
    double N_target = 6;  // default half filling
};

// unpolarized half filled
inline void init_unpolarized_hfill(basis_matrix& rho)
{
    const int n_dim = rho.dims.total();
    rho.dat = 0.5 * Eigen::MatrixXcd::Identity(n_dim, n_dim);
}

inline Eigen::MatrixXcd random_hermitian_perturbation(int n_dim, double epsilon, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    Eigen::MatrixXcd B(n_dim, n_dim);
    for (int i = 0; i < n_dim; i++)
        for (int j = 0; j < n_dim; j++)
            B(i, j) = std::complex<double>(dist(rng), dist(rng));

    Eigen::MatrixXcd H = (B + B.adjoint()) / 2.0;
    H *= epsilon / H.norm();
    return H;
}

inline Eigen::MatrixXcd build_random_density_matrix(basis_matrix& rho, unsigned seed, double N_target){

    Eigen::MatrixXcd Hrandom = random_hermitian_perturbation(rho.dims.total(), 1.0, seed);

    // Diagonalize
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(Hrandom);
    Eigen::VectorXcd evals = solver.eigenvalues();
    Eigen::MatrixXcd evecs = solver.eigenvectors();

    Eigen::MatrixXcd rho_mat = Eigen::MatrixXcd::Zero(rho.dims.total(), rho.dims.total());
    for (int m = 0; m < N_target; m++) {
        rho_mat += evecs.col(m) * evecs.col(m).adjoint();
    }

    return rho_mat;
}

// Change filling
inline void init_random_seed(basis_matrix& rho, int seed, double N_target){
    rho.dat = build_random_density_matrix(rho, seed, N_target);
}

inline void initialize_rho(basis_matrix& rho, const rho_init_params& p)
{
    switch (p.kind) {
        case rho_init_kind::unpolarized_hfill:
            init_unpolarized_hfill(rho);
            break;
        case rho_init_kind::random_seed:
            init_random_seed(rho, p.seed, p.N_target);
            break;
    }

    // Possibly add check hermitian here.
}
