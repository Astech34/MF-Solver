#pragma once
#include <Eigen/Dense>
#include <cassert>
#include <complex>

#include "MF-Solver/model.hpp"

struct basis_matrix {
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
    Eigen::MatrixXcd dat;     // Matrix Data

    explicit basis_matrix(const hopping_model& model) {
        dims.num_layers   = model.num_layers;
        dims.num_sites    = model.num_sites;
        dims.num_orbitals = model.num_orbitals;
        dat = Eigen::MatrixXcd::Zero(dims.total(), dims.total());
    }

    // Combined index for (layer, site, spin, orbital)
    int idx(int layer, int site, int spin, int orbital) const {
        return ((layer * dims.num_sites + site) * dims.num_spin + spin) * dims.num_orbitals + orbital;
    }

    // Flat indexing
    cplx& operator()(Eigen::Index row, Eigen::Index col) { return dat(row, col); }
    const cplx& operator()(Eigen::Index row, Eigen::Index col) const { return dat(row, col); }

    // Labeled indexing: rho(li, ri, si, mi,  lj, rj, sj, mj)
    cplx& operator()(int li, int ri, int si, int mi,
                     int lj, int rj, int sj, int mj) {
        return dat(idx(li, ri, si, mi), idx(lj, rj, sj, mj));
    }
    const cplx& operator()(int li, int ri, int si, int mi,
                           int lj, int rj, int sj, int mj) const {
        return dat(idx(li, ri, si, mi), idx(lj, rj, sj, mj));
    }

    // Arithmetic (operands must share the same dimensions)
    basis_matrix& operator+=(const basis_matrix& other) {
        assert(same_dims(other));
        dat += other.dat;
        return *this;
    }
    basis_matrix& operator-=(const basis_matrix& other) {
        assert(same_dims(other));
        dat -= other.dat;
        return *this;
    }
    friend basis_matrix operator+(basis_matrix lhs, const basis_matrix& rhs) {
        lhs += rhs;
        return lhs;
    }
    friend basis_matrix operator-(basis_matrix lhs, const basis_matrix& rhs) {
        lhs -= rhs;
        return lhs;
    }

private:
    bool same_dims(const basis_matrix& o) const {
        return dims.num_layers == o.dims.num_layers && dims.num_sites == o.dims.num_sites &&
               dims.num_spin == o.dims.num_spin && dims.num_orbitals == o.dims.num_orbitals;
    }
};
