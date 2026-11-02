#pragma once
#include <Eigen/Dense>

struct scf_state {
    Eigen::MatrixXcd rho; // Density matrix
    double mu; // Chemical Potential
    int iteration = 0;
    double tolerance = 0.0;

    // Kronecker Indexing
    // Non-const overload: allows modifying elements (e.g., wrapper(r, c) = val)
    double& operator()(Eigen::Index row, Eigen::Index col) {
        return rho(row, col);
    }

    // Const overload: allows reading elements from const instances
    const double& operator()(Eigen::Index row, Eigen::Index col) const {
        return rho(row, col);
    }

};