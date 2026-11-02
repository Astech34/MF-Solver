#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/model.hpp"

#include <cmath>

// Spin Matrices
SpinMatrices spin_matrices(double theta, double phi) {
    SpinMatrices s;

    s.sx << cd(0,0),   cd(0.5,0),
            cd(0.5,0), cd(0,0);

    s.sy << cd(0,0),   cd(0,-0.5),
            cd(0,0.5), cd(0,0);

    s.sz << cd(0.5,0), cd(0,0),
            cd(0,0),   cd(-0.5,0);

    if (theta != 0.0 || phi != 0.0) {
        const double c  = std::cos(theta / 2.0);
        const double sv = std::sin(theta / 2.0);
        const cd     ep = std::exp(cd(0,  phi/2));
        const cd     em = std::exp(cd(0, -phi/2));

        Eigen::Matrix<cd, 2, 2> U;
        U << em * cd(c, 0),  -cd(sv,0) * em,
              cd(sv,0) * ep,   ep * cd(c, 0);

        s.sx = U.adjoint() * s.sx * U;
        s.sy = U.adjoint() * s.sy * U;
        s.sz = U.adjoint() * s.sz * U;
    }

    return s;
}

Eigen::MatrixXcd SOC(const hopping_model& hopping, const kfree_params& kfp){
    SpinMatrices Smat = spin_matrices(kfp.theta, kfp.phi);
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(hopping.L.Lx.rows()*Smat.sx.rows(), hopping.L.Lx.rows()*Smat.sx.rows());
    result += kfp.lambda * kron(Smat.sx, hopping.L.Lx);
    result += kfp.lambda * kron(Smat.sy, hopping.L.Ly);
    result += kfp.lambda * kron(Smat.sz, hopping.L.Lz);
    return result;
}

// Our ordering will be
// (layer, site, spin, orbital)
Eigen::MatrixXcd H_local(const hopping_model& hopping, const kfree_params& kfp){
    const int n = 2 * hopping.dim();
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(n, n);

    // SOC
    result += kron(hopping.ILayer(), kron(hopping.ISite(), SOC(hopping, kfp)));

    // Eventually add kanamori here.

    return result;
}
