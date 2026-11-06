#include "MF-Solver/observable.hpp"
#include "MF-Solver/basis_matrix.hpp"
#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/model.hpp"
#include "MF-Solver/scf.hpp"

Moments compute_LMoments(const hopping_model& hopping, const scf_state& scf){

    // Index ordering: (layer, site, spin, orbital), orbital fastest

    // Matrix dim
    const int mdim = hopping.dims.num_spin * hopping.dims.num_orbitals;

    const Eigen::MatrixXcd Lx_op = kron(hopping.ISpin(), hopping.L.Lx);
    const Eigen::MatrixXcd Ly_op = kron(hopping.ISpin(), hopping.L.Ly);
    const Eigen::MatrixXcd Lz_op = kron(hopping.ISpin(), hopping.L.Lz);

    const int nlayer = hopping.dims.num_layers;
    const int nsite = hopping.dims.num_sites;
    const int nb = hopping.dims.num_spin * hopping.dims.num_orbitals;

    Moments out(nlayer, nsite, Eigen::Vector3d::Zero());

    for (int l = 0; l < nlayer; ++l) {
        for (int s = 0; s < nsite; ++s) {
            const int base = (l * nsite + s) * nb;
            const auto blk = scf.rho.dat.block(base, base, nb, nb);

            // <O> = Tr(rho O) for rho_ij = <c_j^† c_i>;
            // use O.transpose() instead if rho_ij = <c_i^† c_j>
            out(l, s) = Eigen::Vector3d(
                (blk * Lx_op).trace().real(),
                (blk * Ly_op).trace().real(),
                (blk * Lz_op).trace().real());
        }
    }
    return out;
}

// Spin Moments
Moments compute_SMoments(const hopping_model& hopping, const scf_state& scf){
    const auto [Sx, Sy, Sz] = spin_matrices(0, 0);

    // Index ordering: (layer, site, spin, orbital), orbital fastest

    // Matrix dim
    const int mdim = hopping.dims.num_spin * hopping.dims.num_orbitals;

    const Eigen::MatrixXcd Sx_op = kron(Sx, hopping.IOrbital());
    const Eigen::MatrixXcd Sy_op = kron(Sy, hopping.IOrbital());
    const Eigen::MatrixXcd Sz_op = kron(Sz, hopping.IOrbital());

    const int nlayer = hopping.dims.num_layers;
    const int nsite = hopping.dims.num_sites;
    const int nb = hopping.dims.num_spin * hopping.dims.num_orbitals;

    Moments out(nlayer, nsite, Eigen::Vector3d::Zero());

    for (int l = 0; l < nlayer; ++l) {
        for (int s = 0; s < nsite; ++s) {
            const int base = (l * nsite + s) * nb;
            const auto blk = scf.rho.dat.block(base, base, nb, nb);

            // <O> = Tr(rho O) for rho_ij = <c_j^† c_i>;
            // use O.transpose() instead if rho_ij = <c_i^† c_j>
            out(l, s) = Eigen::Vector3d(
                (blk * Sx_op).trace().real(),
                (blk * Sy_op).trace().real(),
                (blk * Sz_op).trace().real());
        }
    }
    return out;
}
