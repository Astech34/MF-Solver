#include "MF-Solver/bands.hpp"
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <iomanip>

// Currently only for hopping model
void save_bands(const hopping_model& model, const kfree_params& kfp, int num_points, const std::string& filename){

    struct PathPoint { double kx, ky; double path_coord; std::string label; };
    std::vector<PathPoint> path;
    path.reserve(num_points);

    const int n_seg       = static_cast<int>(model.symmetry_path.size()) - 1;
    const int pts_per_seg = num_points / n_seg;
    double path_coord = 0.0;

    // Reciprocal vectors as columns: B = 2π (A^-1)^T. symmetry_path is in reduced coords.
    const Eigen::Matrix3d B = 2.0 * M_PI * model.lattice_vectors.inverse().transpose();
    auto to_cart = [&B](const hopping_model::KPoint& p) -> Eigen::Vector3d {
        return B * Eigen::Vector3d(p.kx, p.ky, p.kz);
    };

    for (int seg = 0; seg < n_seg; seg++) {
        const Eigen::Vector3d k0 = to_cart(model.symmetry_path[seg]);
        const Eigen::Vector3d k1 = to_cart(model.symmetry_path[seg+1]);
        const double seg_len = (k1 - k0).norm();
        const double d_path  = seg_len / pts_per_seg;

        for (int i = 0; i < pts_per_seg; i++) {
            const double t = static_cast<double>(i) / pts_per_seg;
            const Eigen::Vector3d k = k0 + t * (k1 - k0);
            std::string lbl = (i == 0) ? model.symmetry_path[seg].label : "";
            path.push_back({k.x(), k.y(), path_coord, lbl});
            path_coord += d_path;
        }
    }

    const Eigen::Vector3d k_end = to_cart(model.symmetry_path.back());
    path.push_back({k_end.x(), k_end.y(), path_coord, model.symmetry_path.back().label});

    std::ofstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("save_bands: could not open " + filename);

    // Header
    file << "path_coord,kx,ky,label";
    for (int b = 0; b < 2 * model.dim(); b++)
        file << ",band_" << b;
    file << "\n";

    file << std::fixed << std::setprecision(8);

    for (const auto& pt : path) {
        Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(2 * model.dim(), 2 * model.dim());

        // (layer, site, spin, orbital)
        // Kinetic Hamiltonian
        H += model.H0(Eigen::Vector3d(pt.kx, pt.ky, 0.0));

        // Local Hamiltonian
        H += H_local(model, kfp);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(H);
        const auto& evals = solver.eigenvalues();

        file << pt.path_coord << "," << pt.kx << "," << pt.ky << ","
             << pt.label;
        for (int b = 0; b < 2 * model.dim(); b++)
            file << "," << evals[b];
        file << "\n";
    }

    std::cout << "Band structure written to " << filename
              << " (" << path.size() << " k-points)\n";
}