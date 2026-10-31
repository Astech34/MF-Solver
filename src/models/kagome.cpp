#include <cmath>

#include <Eigen/Dense>

#include "model.hpp"   // hopping_model, HkFunction

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
    m.num_sites    = 3;      // A, B, C
    m.num_orbitals = 1;
    m.num_layers   = 1;
    m.k_dim        = 2;
    m.a      = 1;

    t = params.t; //unpack params

    // Lattice vectors as columns. Third column is a dummy z vector so the
    // matrix stays invertible (needed for the reciprocal vectors).
    m.lattice_vectors.col(0) = m.a * Eigen::Vector3d(1.0, 0.0, 0.0);
    m.lattice_vectors.col(1) = m.a * Eigen::Vector3d(1.0, std::sqrt(3.0), 0.0);
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
        {1.0/3.0 * m.a, 1.0/3.0 * m.a, 0.0, "K"},
        {0.5 * m.a, 0.0, 0.0, "M"},
        {0.0, 0.0, 0.0, "Γ"}
    };

    // k must be Cartesian. The lambda captures t and the bonds by value,
    // so it stays valid after this function returns.
    m.H0 = [t, a1, a2, a3](const Eigen::Vector3d& k) {
        Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(3, 3);

        const double hAB = 2 * t * std::cos(k.dot(a1));
        const double hAC = 2 * t * std::cos(k.dot(a2));
        const double hBC = 2 * t * std::cos(k.dot(a3));

        H(0, 1) = hAB;  H(1, 0) = hAB;
        H(0, 2) = hAC;  H(2, 0) = hAC;
        H(1, 2) = hBC;  H(2, 1) = hBC;

        return H;
    };

    return m;
}