#pragma once
#include <Eigen/Dense>
#include <complex>

#include "MF-Solver/model.hpp"

struct scf_state {
    using cplx = std::complex<double>;

    // Index ordering: (layer, site, spin, orbital), orbital fastest
    struct Dims {
        int num_layers   = 0;
        int num_sites    = 0;
        int num_spin     = 2;
        int num_orbitals = 0;

        int total() const { return num_layers * num_sites * num_spin * num_orbitals; }
    };

    Dims dims;                // Dimensions of the system
    Eigen::MatrixXcd rho;     // Density matrix
    double mu = 0.0;          // Chemical potential
    int iteration = 0;
    double tolerance = 0.0;

    explicit scf_state(const hopping_model& model) {
        dims.num_layers   = model.num_layers;
        dims.num_sites    = model.num_sites;
        dims.num_orbitals = model.num_orbitals;
        rho = Eigen::MatrixXcd::Zero(dims.total(), dims.total());
    }

    // Combined index for (layer, site, spin, orbital)
    int idx(int layer, int site, int spin, int orbital) const {
        return ((layer * dims.num_sites + site) * dims.num_spin + spin) * dims.num_orbitals + orbital;
    }

    // Flat indexing
    cplx& operator()(Eigen::Index row, Eigen::Index col) { return rho(row, col); }
    const cplx& operator()(Eigen::Index row, Eigen::Index col) const { return rho(row, col); }

    // Labeled indexing: rho(li, ri, si, mi,  lj, rj, sj, mj)
    cplx& operator()(int li, int ri, int si, int mi,
                     int lj, int rj, int sj, int mj) {
        return rho(idx(li, ri, si, mi), idx(lj, rj, sj, mj));
    }
    const cplx& operator()(int li, int ri, int si, int mi,
                           int lj, int rj, int sj, int mj) const {
        return rho(idx(li, ri, si, mi), idx(lj, rj, sj, mj));
    }
};
