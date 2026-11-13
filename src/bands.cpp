#include "MF-Solver/bands.hpp"
#include "MF-Solver/basis_matrix.hpp"
#include "MF-Solver/initial_density.hpp"
#include "MF-Solver/scf.hpp"
#include "MF-Solver/kspace.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <iomanip>

// Currently only for hopping model
void save_bands(const hopping_model& model, const scf_state& scf, const kfree_params& kfp, int num_points, const std::string& filename){

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
    file << "path_coord,kx,ky,label,mu";
    for (int b = 0; b < model.dims.total(); b++)
        file << ",band_" << b;
    file << "\n";

    file << std::fixed << std::setprecision(8);

    for (const auto& pt : path) {
        basis_matrix H(model.dims);

        // (layer, site, spin, orbital)
        // Kinetic Hamiltonian
        H += model.H0(Eigen::Vector3d(pt.kx, pt.ky, 0.0));

        // Local Hamiltonian
        H += H_local(model, scf, kfp);


        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(H.dat);
        const auto& evals = solver.eigenvalues();

        file << pt.path_coord << "," << pt.kx << "," << pt.ky << ","
             << pt.label << "," << scf.mu;
        for (int b = 0; b < model.dims.total(); b++)
            file << "," << evals[b];
        file << "\n";
    }

    std::cout << "Band structure written to " << filename
              << " (" << path.size() << " k-points)\n";
}

// Density of states per unit cell from H(k) = H0(k) + H_local[scf] on a
// grid_size x grid_size k-grid, with Gaussian broadening of width sigma:
//   D(E) = (1/N_k) sum_{k,n} exp(-(E - e_nk)^2 / 2 sigma^2) / (sqrt(2 pi) sigma)
// Integrates to dims.total() (number of bands, spin included).
void save_dos(const hopping_model& model, const scf_state& scf, const kfree_params& kfp,
              int grid_size, int num_energies, double sigma, const std::string& filename){

    if (num_energies < 2)
        throw std::runtime_error("save_dos: num_energies must be >= 2");
    if (sigma <= 0.0)
        throw std::runtime_error("save_dos: sigma must be positive");

    const Eigensystem es = compute_eigensystem_grid(model, scf, kfp, grid_size);
    const int n_k = static_cast<int>(es.evals.size());

    // Energy window: band extrema padded by 5 sigma so the tails are captured
    double e_min =  std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();
    for (const auto& ev : es.evals) {
        e_min = std::min(e_min, ev.minCoeff());
        e_max = std::max(e_max, ev.maxCoeff());
    }
    e_min -= 5.0 * sigma;
    e_max += 5.0 * sigma;

    const double dE   = (e_max - e_min) / (num_energies - 1);
    const double norm = 1.0 / (std::sqrt(2.0 * M_PI) * sigma * n_k);
    const double inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    const double cutoff  = 6.0 * sigma;  // Gaussian negligible beyond this

    std::vector<double> dos(num_energies, 0.0);

    // Bin each eigenvalue into the energies within its cutoff window
    for (const auto& ev : es.evals) {
        for (int b = 0; b < ev.size(); b++) {
            const double e = ev[b];
            const int i_lo = std::max(0, static_cast<int>(std::floor((e - cutoff - e_min) / dE)));
            const int i_hi = std::min(num_energies - 1, static_cast<int>(std::ceil((e + cutoff - e_min) / dE)));
            for (int i = i_lo; i <= i_hi; i++) {
                const double x = e_min + i * dE - e;
                dos[i] += std::exp(-x * x * inv_2s2);
            }
        }
    }

    std::ofstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("save_dos: could not open " + filename);

    file << "energy,mu,dos\n";
    file << std::fixed << std::setprecision(8);
    for (int i = 0; i < num_energies; i++)
        file << e_min + i * dE << "," << scf.mu << "," << dos[i] * norm << "\n";

    std::cout << "DOS written to " << filename
              << " (" << n_k << " k-points, " << num_energies
              << " energies, sigma = " << sigma << ")\n";
}