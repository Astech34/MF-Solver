#pragma once

#include <Eigen/Dense>

#include "MF-Solver/basis_matrix.hpp"

enum class rho_init_kind { unpolarized_hfill };

// Maybe useful later to add for random noise, etc.
struct rho_init_params {
    rho_init_kind kind = rho_init_kind::unpolarized_hfill;
};

// unpolarized half filled
inline void init_unpolarized_hfill(basis_matrix& rho)
{
    const int n_dim = rho.dims.total();
    rho.dat = 0.5 * Eigen::MatrixXcd::Identity(n_dim, n_dim);
}

inline void initialize_rho(basis_matrix& rho, const rho_init_params& p)
{
    switch (p.kind) {
        case rho_init_kind::unpolarized_hfill:
            init_unpolarized_hfill(rho);
            break;
    }

    // Possibly add check hermitian here.
}
