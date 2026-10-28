#pragma once

#include <Eigen/Dense>
#include <complex>
#include <utility>

//
using HkFunction = std::function<Eigen::MatrixXcd(const Eigen::Vector3d& k)>;

// Create the struct which holds the h(k) hopping model
struct hopping_model{
    std::string name;

    int num_sites; // sites per cell
    int num_orbitals; // orbitals per site
    int num_layers; // layers per cell

    int k_dim = 2; // dimension of k-space either 2 or 3

    // lattice vectors
    int a = 1; // lattice constant, default to 1
    Eigen::Matrix3d lattice_vectors;

    // The function h(k)
    HkFunction H0;

};