#pragma once

#include <functional>

#include "MF-Solver/basis_matrix.hpp"
#include "MF-Solver/dims.hpp"
#include "MF-Solver/kspace.hpp"

// Root finder (Brent's method) on [a, b]; f(a) and f(b) must have opposite signs
double brent(std::function<double(double)> f, double a, double b,
             double tol = 1e-12, int max_iter = 200);

// Total particle number per k-point at chemical potential mu
double NTotal(const Eigensystem& sys, double mu, double T);

// Chemical potential giving N_target particles; NaN if no root is found
double find_mu(const Eigensystem& sys, double T, double N_target);

basis_matrix compute_density_matrix(const Eigensystem& sys, const Dims dims, double mu, double T);
