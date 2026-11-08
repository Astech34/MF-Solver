#include "MF-Solver/kspace.hpp"

#include <cmath>

#include "MF-Solver/basis_matrix.hpp"
#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/model.hpp"

// Only 2D for now.
Eigensystem compute_eigensystem_grid(const hopping_model& model, const scf_state& scf,
                                     const kfree_params& kfp, const int grid_size) {
    const int N = grid_size * grid_size;

    Eigensystem result;
    result.evals.resize(N);
    result.evecs.resize(N);

    const Eigen::Matrix3d recip_vectors = model.compute_recip_vectors();
    Eigen::Vector3d b1 = recip_vectors.col(0), b2 = recip_vectors.col(1), b3 = recip_vectors.col(2);

    const double area = b1.cross(b2).norm();

    if (N % 2)
        throw std::runtime_error("Grid size must be even for 2D k-space sampling.");

    // Check this
    // Precompute u,v points [0, 1)
    std::vector<double> uv_lin(grid_size);
    for (int i = 0; i < grid_size; i++)
        uv_lin[i] = static_cast<double>(i) / grid_size - 0.5;

    // k-independent piece, computed once
    const basis_matrix Hlocal = H_local(model, scf, kfp);

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < N; idx++) {
        const Eigen::Vector3d k = uv_lin[idx % grid_size] * b1 + uv_lin[idx / grid_size] * b2;

        // Local to each iteration so threads don't share it
        const Eigen::MatrixXcd H = model.H0(k).dat + Hlocal.dat; //H0 called from m threads (make sure const)

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(H);
        result.evals[idx] = solver.eigenvalues();
        result.evecs[idx] = solver.eigenvectors();
    }

    return result;
}
