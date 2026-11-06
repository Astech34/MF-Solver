#pragma once
#include <Eigen/Dense>
#include <cassert>
#include <complex>

#include "MF-Solver/dims.hpp"

// Initalize by passing in dims (hoppings.dim)

// Index ordering: (layer, site, spin, orbital), orbital fastest
struct basis_matrix {
    using cplx = std::complex<double>;

    Dims dims;                // Dimensions of the system
    Eigen::MatrixXcd dat;     // Matrix Data

    explicit basis_matrix(const Dims& dms) {
        dims = dms;
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

    // Block
    using Block      = Eigen::Block<Eigen::MatrixXcd>;
    using ConstBlock = Eigen::Block<const Eigen::MatrixXcd>;

    // Size of one (spin, orbital) block
    int nb() const { return dims.num_spin * dims.num_orbitals; }

    // (spin, orbital) block coupling (li, ri) to (lj, rj)
    Block site_block(int li, int ri, int lj, int rj) {
        return dat.block(idx(li, ri, 0, 0), idx(lj, rj, 0, 0), nb(), nb());
    }
    ConstBlock site_block(int li, int ri, int lj, int rj) const {
        return dat.block(idx(li, ri, 0, 0), idx(lj, rj, 0, 0), nb(), nb());
    }

    // On-site (diagonal) block for (l, s)
    Block      site_block(int l, int s)       { return site_block(l, s, l, s); }
    ConstBlock site_block(int l, int s) const { return site_block(l, s, l, s); }

    // Explicit setters
    template <typename Derived>
    void set_site_block(int li, int ri, int lj, int rj, const Eigen::MatrixBase<Derived>& M) {
        assert(M.rows() == nb() && M.cols() == nb());
        site_block(li, ri, lj, rj) = M;
    }
    template <typename Derived>
    void set_site_block(int l, int s, const Eigen::MatrixBase<Derived>& M) {
        set_site_block(l, s, l, s, M);
    }

private:
    bool same_dims(const basis_matrix& o) const {
        return dims.num_layers == o.dims.num_layers && dims.num_sites == o.dims.num_sites &&
               dims.num_spin == o.dims.num_spin && dims.num_orbitals == o.dims.num_orbitals;
    }
};
