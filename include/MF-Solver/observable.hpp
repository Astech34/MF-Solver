#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <vector>

// Forward declarations (scf.hpp includes this header, so avoid a cycle)
struct hopping_model;
struct scf_state;

template <typename T>
struct LayerSiteList {
    int nlayer = 0, nsite = 0;
    std::vector<T> v;   // flat, index = layer*nsite + site

    LayerSiteList(int nl, int ns, const T& zero)
        : nlayer(nl), nsite(ns), v(nl * ns, zero) {}

    T&       operator()(int layer, int site)       { return v[layer * nsite + site]; }
    const T& operator()(int layer, int site) const { return v[layer * nsite + site]; }

    T layer_total(int layer) const {
        T tot = v[layer * nsite];
        for (int s = 1; s < nsite; ++s) tot += (*this)(layer, s);
        return tot;
    }

    // Element-wise arithmetic (shapes must match)
    LayerSiteList& operator+=(const LayerSiteList& o) {
        assert(nlayer == o.nlayer && nsite == o.nsite);
        for (size_t i = 0; i < v.size(); ++i) v[i] += o.v[i];
        return *this;
    }
    LayerSiteList& operator-=(const LayerSiteList& o) {
        assert(nlayer == o.nlayer && nsite == o.nsite);
        for (size_t i = 0; i < v.size(); ++i) v[i] -= o.v[i];
        return *this;
    }
    LayerSiteList& operator*=(double a) {
        for (auto& x : v) x *= a;
        return *this;
    }

    friend LayerSiteList operator+(LayerSiteList a, const LayerSiteList& b) { return a += b; }
    friend LayerSiteList operator-(LayerSiteList a, const LayerSiteList& b) { return a -= b; }
    friend LayerSiteList operator*(LayerSiteList a, double s) { return a *= s; }
    friend LayerSiteList operator*(double s, LayerSiteList a) { return a *= s; }
};

using MomentList = LayerSiteList<Eigen::Vector3d>;
using ScalarList = LayerSiteList<double>;
using Moments    = MomentList;

// Orbital (L) and spin (S) moments per (layer, site): <O> = Tr(rho O)
Moments compute_LMoments(const hopping_model& hopping, const scf_state& scf);
Moments compute_SMoments(const hopping_model& hopping, const scf_state& scf);
