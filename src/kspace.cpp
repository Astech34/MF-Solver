#include "MF-Solver/kspace.hpp"

#include <cmath>

#include "MF-Solver/basis_matrix.hpp"
#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/model.hpp"

Eigensystem compute_eigensystem_grid(const hopping_model& model, const scf_state& scf,
                                     const kfree_params& kfp, const int grid_size) {
    const int N = grid_size * grid_size;

    Eigensystem result;
    result.evals.resize(N);
    result.evecs.resize(N);

    // Precompute k-points (linspace from -pi to pi, endpoint excluded to avoid
    // duplicating the BZ boundary). TODO: check BZ sampling for the lattice.
    std::vector<double> k_lin(grid_size);
    for (int i = 0; i < grid_size; i++)
        k_lin[i] = -M_PI + i * (2.0 * M_PI / grid_size);

    // k-independent piece, computed once
    const basis_matrix Hlocal = H_local(model, scf, kfp);

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < N; idx++) {
        const double kx = k_lin[idx % grid_size];
        const double ky = k_lin[idx / grid_size];

        // Local to each iteration so threads don't share it
        const Eigen::MatrixXcd H = model.H0(Eigen::Vector3d(kx, ky, 0.0)).dat + Hlocal.dat;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(H);
        result.evals[idx] = solver.eigenvalues();
        result.evecs[idx] = solver.eigenvectors();
    }

    return result;
}
