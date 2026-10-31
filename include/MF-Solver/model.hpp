#pragma once

#include <Eigen/Dense>
#include <complex>
#include <functional>
#include <string>
#include <vector>
#include <utility>

//
using HkFunction = std::function<Eigen::MatrixXcd(const Eigen::Vector3d& k)>;

struct LMatrices {
    Eigen::MatrixXcd Lx, Ly, Lz;
};

// Create the struct which holds the h(k) hopping model
struct hopping_model{
    std::string name;

    int num_sites; // sites per cell
    int num_orbitals; // orbitals per site
    int num_layers; // layers per cell

    // Total matrix dimension
    int dim() const { return num_sites * num_orbitals * num_layers; }

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

    // The function h(k)
    HkFunction H0;

    // Pre Computed L matrices
    LMatrices L;

};