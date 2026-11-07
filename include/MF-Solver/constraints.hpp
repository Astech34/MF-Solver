#pragma once

#include "MF-Solver/basis_matrix.hpp"
#include "MF-Solver/model.hpp"
#include "MF-Solver/observable.hpp"
#include "MF-Solver/scf.hpp"

using SpinTilts = MomentList;

// Spin tilt (m_hat - z_hat) for each (layer, site)
SpinTilts spin_tilts(const hopping_model& hopping, const scf_state& scf);

// Violation for mixing
ScalarList constraint_violation(const hopping_model& hopping, const scf_state& scf);

// Constraining-field matrix built from the spin tilts
basis_matrix ConstrainField(const hopping_model& hopping, const scf_state& scf);
