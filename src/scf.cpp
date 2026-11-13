#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>

#include "MF-Solver/scf.hpp"
#include "MF-Solver/constraints.hpp"
#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/kspace.hpp"
#include "MF-Solver/occupations.hpp"
#include "MF-Solver/energy.hpp"
#include "MF-Solver/model.hpp"

namespace {

// Collinear, no-SOC projection for comparison with DFT+U:
// drop imaginary parts and zero all spin-off-diagonal elements.
void apply_collinear_constraint(Eigen::MatrixXcd& rho, const Dims& d) {
    rho = rho.real().cast<cd>().eval();
    const int n = static_cast<int>(rho.rows());
    auto spin_of = [&](int i) { return (i / d.num_orbitals) % d.num_spin; };
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (spin_of(i) != spin_of(j)) rho(i, j) = cd(0.0, 0.0);
}

double dot(const ScalarList& a, const ScalarList& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.v.size(); ++i) sum += a.v[i] * b.v[i];
    return sum;
}

double l2_norm(const ScalarList& a) { return std::sqrt(dot(a, a)); }

// Frobenius inner product Re<A, B>_F
double frob(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B) {
    return (A.array().conjugate() * B.array()).sum().real();
}

// Enforce Hermiticity (and the collinear projection if requested) on a mixed density.
void finalize_rho(Eigen::MatrixXcd& rho, const Dims& d, bool comp_dft) {
    rho = (0.5 * (rho + rho.adjoint())).eval();
    if (comp_dft) apply_collinear_constraint(rho, d);
}

// One Pulay history entry: input state and its residual.
struct DiisEntry {
    Eigen::MatrixXcd rho;      // rho_in
    Eigen::MatrixXcd rho_err;  // rho_out - rho_in
    ScalarList       lam;      // constraint fields in
    ScalarList       lam_err;  // constraint violation g
};

// Solve the Pulay system for the mixing coefficients c (Σc_i = 1).
// B_ij = Re<e_i, e_j>_F + w_con * <g_i, g_j>, normalised by its largest diagonal.
// Oldest entries are dropped while B is ill-conditioned.
Eigen::VectorXd diis_coefficients(std::deque<DiisEntry>& hist, double w_con) {
    constexpr double max_cond = 1e10;
    constexpr double ridge    = 1e-12;

    while (true) {
        const int m = static_cast<int>(hist.size());
        Eigen::MatrixXd B(m, m);
        for (int i = 0; i < m; ++i)
            for (int j = i; j < m; ++j)
                B(i, j) = B(j, i) = frob(hist[i].rho_err, hist[j].rho_err)
                                  + w_con * dot(hist[i].lam_err, hist[j].lam_err);

        const double scale = B.diagonal().maxCoeff();
        if (scale > 0.0) B /= scale;

        if (m > 1) {
            const Eigen::VectorXd sv = Eigen::JacobiSVD<Eigen::MatrixXd>(B).singularValues();
            if (sv(m - 1) <= 0.0 || sv(0) / sv(m - 1) > max_cond) {
                hist.pop_front();
                continue;
            }
        }

        //   [B  -1] [c]   [ 0]
        //   [-1  0] [λ] = [-1]
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m + 1, m + 1);
        A.topLeftCorner(m, m) = B + ridge * Eigen::MatrixXd::Identity(m, m);
        A.col(m).head(m).setConstant(-1.0);
        A.row(m).head(m).setConstant(-1.0);

        Eigen::VectorXd b = Eigen::VectorXd::Zero(m + 1);
        b(m) = -1.0;

        return A.colPivHouseholderQr().solve(b).head(m);
    }
}

} // namespace

KanamoriResult runKanamoriSCF(const hopping_model& hopping, const scf_state& scf,
                              const kfree_params& kfp,
                              double alpha, int grid_size,
                              double T, double N_target,
                              MixerType mixer, int max_iter, bool comp_dft)
{
    // TODO: move into Params
    const double rho_tol  = 1e-7;
    const double con_tol  = 999;   // effectively disabled for testing
    const double eta_con  = 0.1;   // constraint-field step size
    const int    diis_max = 8;     // Pulay history length
    const double w_con    = 1.0;   // weight of constraint residual in the DIIS metric

    const Dims& dims = hopping.dims;

    // scf_in holds the INPUT state (density + constraint fields) for each iteration
    scf_state scf_in = scf;
    if (comp_dft) apply_collinear_constraint(scf_in.rho.dat, dims);

    std::deque<DiisEntry> diis_hist;
    const std::string s_type = (mixer == MixerType::Linear) ? "[Linear]" : "[DIIS]";

    for (int it = 0; it < max_iter; ++it) {
        // 1. Diagonalize H[rho_in, lambda_in]
        const Eigensystem sys = compute_eigensystem_grid(hopping, scf_in, kfp, grid_size);
        const double mu = find_mu(sys, T, N_target);
        if (std::isnan(mu)) {
            std::cerr << "\nrunKanamoriSCF: could not find mu at iteration " << it << "\n";
            break;
        }

        // 2. Output density
        Eigen::MatrixXcd rho_out = compute_density_matrix(sys, dims, mu, T).dat;
        if (comp_dft) apply_collinear_constraint(rho_out, dims);

        scf_state scf_out = scf_in;
        scf_out.rho.dat = rho_out;
        scf_out.mu      = mu;

        // 3. Residuals, evaluated before anything is updated
        const Eigen::MatrixXcd rho_err = rho_out - scf_in.rho.dat;
        const ScalarList g             = constraint_violation(hopping, scf_out);
        const double rho_diff          = rho_err.norm();
        const double con_err           = l2_norm(g);

        std::cout << "\r" << s_type << " Iter " << std::setw(5) << it
                  << "  |Δρ|_F = " << std::scientific << std::setprecision(4) << rho_diff
                  << "  |g| = " << con_err << "   " << std::flush;

        // 4. Convergence
        if (rho_diff < rho_tol && con_err < con_tol) {
            const double bandsum = calculate_band_energy(sys, mu, T);
            const double dc      = kanamori_dc(hopping, scf_out, kfp);
            scf_out.energy      = bandsum - dc;
            scf_out.isConverged = true;

            std::cout << std::fixed << std::setprecision(8)
                      << "\nKanamori SCF converged in " << it << " iterations\n"
                      << "  mu          = " << mu      << "\n"
                      << "  Band energy = " << bandsum << "\n"
                      << "  DC          = " << dc      << "\n"
                      << "  E_total     = " << scf_out.energy << "\n"
                      << std::defaultfloat;
            return {scf_out, scf_out.energy, true};
        }

        // 5. Mix. Both mixers take a damped step from the input state:
        //      rho <- rho_in + alpha * e,   lambda <- lambda_in + eta * g
        //    DIIS replaces (rho_in, e, lambda_in, g) by their Pulay-optimal combination.
        //    With a single history entry this reduces exactly to linear mixing.
        if (mixer == MixerType::Linear) {
            scf_in.rho.dat           += alpha * rho_err;
            scf_in.constraint_params += eta_con * g;
        } else {
            diis_hist.push_back({scf_in.rho.dat, rho_err, scf_in.constraint_params, g});
            if (static_cast<int>(diis_hist.size()) > diis_max) diis_hist.pop_front();

            const Eigen::VectorXd c = diis_coefficients(diis_hist, w_con);

            Eigen::MatrixXcd rho_opt = Eigen::MatrixXcd::Zero(rho_out.rows(), rho_out.cols());
            Eigen::MatrixXcd err_opt = Eigen::MatrixXcd::Zero(rho_out.rows(), rho_out.cols());
            ScalarList lam_opt(dims.num_layers, dims.num_sites, 0.0);
            ScalarList g_opt  (dims.num_layers, dims.num_sites, 0.0);
            for (int i = 0; i < c.size(); ++i) {
                rho_opt += c(i) * diis_hist[i].rho;
                err_opt += c(i) * diis_hist[i].rho_err;
                lam_opt += c(i) * diis_hist[i].lam;
                g_opt   += c(i) * diis_hist[i].lam_err;
            }

            scf_in.rho.dat           = rho_opt + alpha * err_opt;
            scf_in.constraint_params = lam_opt + eta_con * g_opt;
        }
        finalize_rho(scf_in.rho.dat, dims, comp_dft);

        scf_in.mu = mu;
    }

    std::cerr << "\nrunKanamoriSCF: not converged after " << max_iter << " iterations\n";
    scf_in.isConverged = false;
    return {scf_in, std::nan(""), false};
}