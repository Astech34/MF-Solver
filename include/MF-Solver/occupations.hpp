#pragma once

#include <functional>
#include <vector>

#include "MF-Solver/basis_matrix.hpp"
#include "MF-Solver/dims.hpp"
#include "MF-Solver/kspace.hpp"

struct scf_state;

// Orbital occupations n = Re rho_ii, indexed by (layer, site, spin, orbital)
struct Occupations {
    Dims dims;
    std::vector<double> v;   // flat, same ordering as basis_matrix (orbital fastest)

    explicit Occupations(const Dims& d) : dims(d), v(d.total(), 0.0) {}

    double&       operator()(int layer, int site, int spin, int orbital)       { return v[idx(layer, site, spin, orbital)]; }
    const double& operator()(int layer, int site, int spin, int orbital) const { return v[idx(layer, site, spin, orbital)]; }

private:
    int idx(int layer, int site, int spin, int orbital) const {
        return ((layer * dims.num_sites + site) * dims.num_spin + spin) * dims.num_orbitals + orbital;
    }
};

// Root finder (Brent's method) on [a, b]; f(a) and f(b) must have opposite signs
double brent(std::function<double(double)> f, double a, double b,
             double tol = 1e-12, int max_iter = 200);

// Total particle number per k-point at chemical potential mu
double NTotal(const Eigensystem& sys, double mu, double T);

// Chemical potential giving N_target particles; NaN if no root is found
double find_mu(const Eigensystem& sys, double T, double N_target);

basis_matrix compute_density_matrix(const Eigensystem& sys, const Dims dims, double mu, double T);

// Orbital occupations from the diagonal of the SCF density matrix
Occupations compute_occupations(const scf_state& scf);
