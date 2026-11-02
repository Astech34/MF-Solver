#pragma once

#include "MF-Solver/model.hpp"

using cd = std::complex<double>;

// Kronecker product of two complex matrices: result[i*B.rows()+k, j*B.cols()+l] = A[i,j] * B[k,l]
// Write a test to test this.
inline Eigen::MatrixXcd kron(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B)
{
    Eigen::MatrixXcd result(A.rows() * B.rows(), A.cols() * B.cols());
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            result.block(i*B.rows(), j*B.cols(), B.rows(), B.cols()) = A(i,j) * B;
    return result;
}

struct kfree_params{
    double lambda = 0;
    double theta = 0;
    double phi = 0;
};

struct SpinMatrices {
    Eigen::Matrix<cd, 2, 2> sx, sy, sz;
};

struct complete_model {
    hopping_model hopping;
};

SpinMatrices spin_matrices(double theta, double phi);
Eigen::MatrixXcd SOC(const hopping_model& hopping, const kfree_params& kfp);
Eigen::MatrixXcd H_local(const hopping_model& hopping, const kfree_params& kfp);
