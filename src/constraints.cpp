#include "MF-Solver/constraints.hpp"
#include "MF-Solver/basis_matrix.hpp"
#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/model.hpp"
#include "MF-Solver/observable.hpp"
#include "MF-Solver/scf.hpp"

// Spin tilt penalty for each layer x site point
SpinTilts spin_tilts(const hopping_model& hopping, const scf_state& scf) {
    //Make sure theta is zero here
    const int nlayer = hopping.dims.num_layers;
    const int nsite = hopping.dims.num_sites;

    const Moments smom = compute_SMoments(hopping, scf);

    SpinTilts tilts(nlayer, nsite, Eigen::Vector3d::Zero());
    for (int l = 0; l < nlayer; ++l) {
        for (int s = 0; s < nsite; ++s) {
            Eigen::Vector3d s_vec = smom(l, s);
            const double m = s_vec.norm();
            Eigen::Vector3d s_vec_n = (m > 0.0) ? Eigen::Vector3d(s_vec / m) : Eigen::Vector3d::Zero().eval();
            s_vec_n.z() -= 1.0;  // m̂ - ẑ

            tilts(l, s) = s_vec_n;
        }
    }

    return tilts;
}

// Not completely necessary might add layer for testing
/*
SpinTilts constraint_violation(const hopping_model& hopping, const scf_state& scf) {
    
}
*/

basis_matrix ConstrainField(const hopping_model& hopping, const scf_state& scf) {
    
    const SpinTilts tilts = spin_tilts(hopping, scf);
    const int nlayer = hopping.dims.num_layers;
    const int nsite = hopping.dims.num_sites;

    const SpinMatrices  sm  = spin_matrices(0, 0); //should this have angle?

    basis_matrix con_field_matrix(hopping.dims);
    for (int l = 0; l < nlayer; ++l) {
        for (int s = 0; s < nsite; ++s) {
            // n·s on each (layer,site) point.
             auto con_field = [&](const Eigen::Vector3d& n) {
                const Eigen::Matrix<cd, 2, 2> n_dot_s = n.x() * sm.sx + n.y() * sm.sy + n.z() * sm.sz;
                return kron(n_dot_s, hopping.IOrbital());
            };

            const Eigen::Vector3d& n = tilts(l, s);
            con_field_matrix.site_block(l, s) = scf.constraint_params(l, s) * con_field(n);
        }
    }

    return con_field_matrix;
}