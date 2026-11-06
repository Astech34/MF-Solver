#include "MF-Solver/kspace.hpp"
#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/model.hpp"
#include "MF-Solver/basis_matrix.hpp"

Eigensystem compute_eigensystem_grid(const hopping_model& model, const scf_state& scf,const kfree_params& kfp, const int grid_size) {
    const int N = grid_size * grid_size;

    Eigensystem result;
    result.evals.resize(N);
    result.evecs.resize(N);

    // Precompute k-points (linspace from -pi to pi)
    // We exclude the endpoint to avoid duplicate k-points at the BZ boundary,
    // but this is a minor detail
    // Check this
    std::vector<double> k_lin(grid_size);
    for (int i = 0; i < grid_size; i++)
        k_lin[i] = -M_PI + i * (2.0 * M_PI / (grid_size));

    // Precompute k-independent pieces once
    basis_matrix Hlocal = H_local(model, kfp, scf);

    // Assemble Hamiltonian
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(model.dims.total(), model.dims.total());

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < N; idx++) {
        const int ix = idx % grid_size;
        const int iy = idx / grid_size;

        const double kx = k_lin[ix];
        const double ky = k_lin[iy];

        H = model.H0(Eigen::Vector3d(kx, ky, 0.0)) + Hlocal.dat;
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(H);

        result.evals[idx] = solver.eigenvalues();
        result.evecs[idx] = solver.eigenvectors();
    }

    return result;
}