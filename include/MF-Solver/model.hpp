#pragma once

#include "MF-Solver/dims.hpp"
#include "MF-Solver/basis_matrix.hpp"

#include <Eigen/Dense>
#include <complex>
#include <functional>
#include <string>
#include <vector>
#include <utility>

//
using HkFunction = std::function<basis_matrix(const Eigen::Vector3d& k)>;

struct LMatrices {
    Eigen::MatrixXcd Lx, Ly, Lz;
};

// Create the struct which holds the h(k) hopping model
struct hopping_model{
    std::string name;

    // Our ordering will be
    // (layer, site, spin, orbital)

    Dims dims; // Dimensions of the system

    // Identity matrix for each dimension
    Eigen::MatrixXcd ILayer() const { return Eigen::MatrixXcd::Identity(dims.num_layers, dims.num_layers); }
    Eigen::MatrixXcd ISite() const { return Eigen::MatrixXcd::Identity(dims.num_sites, dims.num_sites); }
    Eigen::MatrixXcd ISpin() const { return Eigen::MatrixXcd::Identity(dims.num_spin, dims.num_spin); }
    Eigen::MatrixXcd IOrbital() const { return Eigen::MatrixXcd::Identity(dims.num_orbitals, dims.num_orbitals); }

    int k_dim = 2; // dimension of k-space either 2 or 3

    // lattice vectors
    int a = 1; // lattice constant, default to 1
    Eigen::Matrix3d lattice_vectors;

    // Encode Symmetry path for plotting band structure
    struct KPoint {double kx, ky, kz; std::string label;};
    std::vector<KPoint> symmetry_path;

    // model params
    struct model_params{
        double t; // hopping parameter
    } params;

    // The function h(k) in complete augmented basis
    HkFunction H0;

    // Pre Computed L matrices
    LMatrices L;

};