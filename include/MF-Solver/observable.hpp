#pragma once

#include <Eigen/Dense>
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
};

using MomentList = LayerSiteList<Eigen::Vector3d>;
using ScalarList = LayerSiteList<double>;
using Moments    = MomentList;

// Orbital (L) and spin (S) moments per (layer, site): <O> = Tr(rho O)
Moments compute_LMoments(const hopping_model& hopping, const scf_state& scf);
Moments compute_SMoments(const hopping_model& hopping, const scf_state& scf);
