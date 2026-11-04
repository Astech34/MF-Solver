#pragma once

#include "MF-Solver/basis_matrix.hpp"
#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/model.hpp"
#include "MF-Solver/scf.hpp"

// Kanamori mean-field Hamiltonian in the (layer, site, spin, orbital) basis
basis_matrix kanamori(const hopping_model& hopping, const scf_state& scf, const kfree_params& kfp);
