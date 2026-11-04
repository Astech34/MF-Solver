#include <cmath>

#include <Eigen/Dense>

#include "MF-Solver/models/kagome.hpp"   // hopping_model, HkFunction
#include "MF-Solver/basis_matrix.hpp"
#include "MF-Solver/hamiltonian.hpp"

// ---------------------------------------------------------------------------
// Nearest-neighbor kagome model, one s-like orbital per site.
//   H(k) = -2t * [ 0        cos(k.dAB)  cos(k.dAC) ]
//                [ cos(k.dAB)  0        cos(k.dBC) ]
//                [ cos(k.dAC)  cos(k.dBC)  0       ]
// Lattice constant = 1, nearest-neighbor distance = 1/2.
// Flip the sign of t if you prefer the +2t convention.
// ---------------------------------------------------------------------------
hopping_model make_kagome(KagomeParams params)
{
    hopping_model m;
    m.name         = "kagome";
    m.dims.num_layers   = 1;
    m.dims.num_sites    = 3;      // A, B, C
    m.dims.num_spin     = 2;
    m.dims.num_orbitals = 2;
    m.k_dim        = 2;
    m.a      = 1;

    const double t = params.t; //unpack params

    // L matrices (in orbital basis)
    m.L.Lx = Eigen::MatrixXcd::Zero(2, 2);
    m.L.Ly = Eigen::MatrixXcd::Zero(2, 2);

    const std::complex<double> I(0.0, 1.0);

    Eigen::MatrixXcd Lz(2, 2);
    Lz << 0.0, -I,
        I,   0.0;
    m.L.Lz = Lz;

    // Lattice vectors as columns. Third column is a dummy z vector so the
    // matrix stays invertible (needed for the reciprocal vectors).
    m.lattice_vectors.col(0) = m.a * Eigen::Vector3d(1.0, 0.0, 0.0);
    m.lattice_vectors.col(1) = m.a * Eigen::Vector3d(0.5, 0.5 * std::sqrt(3.0), 0.0);
    m.lattice_vectors.col(2) = m.a * Eigen::Vector3d(0.0, 0.0, 1.0);

    // Nearest Neighbor
    // from https://journals.aps.org/rmp/pdf/10.1103/1g9n-wm38?casa_token=Hf_8h40YQowAAAAA%3AXh-Cx2idjkpTIsXiyiU8kUp9Zhn6k-chcqbWsuEqzhBTp1SORbWEptLZ8NjrUMsjgMGzViPh_Ye-hA
    const Eigen::Vector3d a1 = m.a * Eigen::Vector3d(1.0, 0.0, 0.0); //AB
    const Eigen::Vector3d a2 = m.a * 0.5 * Eigen::Vector3d(1.0, std::sqrt(3.0), 0.0); //AC
    const Eigen::Vector3d a3 = a2 - a1; //BC

    // Symmetry Path
    // Check These
    m.symmetry_path = {
        {0.0, 0.0, 0.0, "Γ"},
        {2.0/3.0, 1.0/3.0, 0.0, "K"},
        {0.5, 0.0, 0.0, "M"},
        {0.0, 0.0, 0.0, "Γ"}
    };

    // Copy Identity Matrices
    const auto IL = m.ILayer();
    const auto IS = m.ISpin();
    const auto IO = m.IOrbital();

    // k must be Cartesian. The lambda captures t and the bonds by value,
    // so it stays valid after this function returns.
    m.H0 = [t, a1, a2, a3, IL, IS, IO, dims = m.dims](const Eigen::Vector3d& k) {
        Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(3, 3);

        const double hAB = 2 * t * std::cos(0.5 * k.dot(a1));
        const double hAC = 2 * t * std::cos(0.5 * k.dot(a2));
        const double hBC = 2 * t * std::cos(0.5 * k.dot(a3));

        H(0, 1) = hAB;  H(1, 0) = hAB;
        H(0, 2) = hAC;  H(2, 0) = hAC;
        H(1, 2) = hBC;  H(2, 1) = hBC;

        // H is site basis
        // (layer, site, spin, orbital)

        // H

        basis_matrix H_aug(dims);

        H_aug.dat = kron(IL, kron(H, kron(IS, IO)));
        return H_aug;
    };

    return m;
}