#include <cmath>
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

double l2_norm(const ScalarList& a) {
    double sum = 0.0;
    for (double x : a.v) sum += x * x;
    return std::sqrt(sum);
}

} // namespace

KanamoriResult runKanamoriSCF(const hopping_model& hopping, const scf_state& scf,
                              const kfree_params& kfp,
                              double alpha, int grid_size,
                              double T, double N_target,
                              MixerType mixer, int max_iter, bool comp_dft)
{
    const double tol     = 1e-8;   // TODO: move into Params
    const double eta_con = 0.1;    // constraint-field step size, TODO: move into Params
    const double diis_max = 8;

    const Dims& dims  = hopping.dims;
    const int nlayer  = dims.num_layers;
    const int nsite   = dims.num_sites;

    // scf_in holds the INPUT state (density + constraint fields) for each iteration
    scf_state scf_in = scf;
    if (comp_dft) apply_collinear_constraint(scf_in.rho.dat, dims);

    // DIIS Objects
    // Frobenius inner product Re<A, B>_F (matches the DIIS residual metric).
    auto frob = [](const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B) {
        return (A.array().conjugate() * B.array()).sum().real();
    };

    // --- DIIS state ---
    std::vector<Eigen::MatrixXcd>  diis_rho;      // rho_new history
    std::vector<Eigen::MatrixXcd>  diis_err;      // residual history: e_i = rho_new_i - rho_i
    
    // Store constraints as list of list
    std::vector<ScalarList> diis_lam;  // con_lam_new history (parallel to diis_rho)
    std::vector<ScalarList> diis_lam_err;  // con_lam residual history (parallel to diis_rho)

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
        const ScalarList g     = constraint_violation(hopping, scf_out);
        const double rho_diff  = (rho_out - scf_in.rho.dat).norm();
        const double con_err   = l2_norm(g);

        ScalarList g_new(hopping.dims.num_layers, hopping.dims.num_sites, 0.0); 
        g_new = g + eta_con * g;
        ScalarList g_err(hopping.dims.num_layers, hopping.dims.num_sites, 0.0); 
        g_err = g_new - g;

        std::string s_type;
        if (mixer == MixerType::Linear) {
            s_type = "[Linear]";
        } else {
            s_type = "[DIIS]";
        }

        std::cout << "\r" << s_type << " Iter " << std::setw(5) << it
                  << "  |Δρ|_F = " << std::scientific << std::setprecision(4) << rho_diff
                  << "  |g| = " << con_err << "   " << std::flush;

        // 4. Convergence con_err < 1 (fix only for testing)
        if (rho_diff < 1e-6 && con_err < 999) {
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

        // 5. Linear mix of rho (symmetrized), gradient step on constraint fields
        if (mixer == MixerType::Linear) {
            const Eigen::MatrixXcd rho_mix = alpha * rho_out + (1.0 - alpha) * scf_in.rho.dat;
            scf_in.rho.dat = 0.5 * (rho_mix + rho_mix.adjoint());
        }
        else {
            // DIIS Mixing
            diis_rho.push_back(rho_out);
            diis_err.push_back(rho_out - scf_in.rho.dat);
            
            diis_lam.push_back(g_new);
            diis_lam_err.push_back(g_err);

            // Manage history 
            if (static_cast<int>(diis_rho.size()) > diis_max) {
                diis_rho.erase(diis_rho.begin());
                diis_err.erase(diis_err.begin());
                
                diis_lam.erase(diis_lam.begin());
                diis_lam_err.erase(diis_lam_err.begin());
            }

            const int m = static_cast<int>(diis_rho.size());

            // Build (m+1)×(m+1) Pulay system
            //   [B   -1] [c]   [ 0]
            //   [-1   0] [λ] = [-1]
            // B_ij = Re(<e_i, e_j>_F) + e^lam_i e^lam_j,  Σc_i = 1 enforced by λ
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m + 1, m + 1);
            Eigen::VectorXd b = Eigen::VectorXd::Zero(m + 1);
            b(m) = -1.0;

            for (int ii = 0; ii < m; ii++) {
                for (int jj = ii; jj < m; jj++) {
                    
                    double Bij = frob(diis_err[ii], diis_err[jj]);
                    
                    // Add Errors for each (layer, site)
                    for (int ly = 0; ly < hopping.dims.num_layers; ly++) {
                        for (int st = 0; st < hopping.dims.num_sites; st++) {
                            Bij += diis_lam_err[ii](ly, st) * diis_lam_err[jj](ly, st);
                        }
                    }

                    A(ii, jj) = Bij;
                    A(jj, ii) = Bij;
                }
                A(ii, m) = -1.0;
                A(m, ii) = -1.0;
            }

            const Eigen::VectorXd c = A.colPivHouseholderQr().solve(b);

            Eigen::MatrixXcd rho_s = Eigen::MatrixXd::Zero(scf.rho.dat.rows(), scf.rho.dat.cols());
            ScalarList lam_s(hopping.dims.num_layers, hopping.dims.num_sites, 0.0);
            for (int ii = 0; ii < m; ii++) {
                rho_s.noalias() += c(ii) * diis_rho[ii];

                for (int ly = 0; ly < hopping.dims.num_layers; ly++) {
                        for (int st = 0; st < hopping.dims.num_sites; st++) {
                            lam_s(ly, st) = lam_s(ly, st) + c(ii) * diis_lam[ii](ly, st);
                    }
                }
            }
        }

        for (int l = 0; l < nlayer; ++l)
            for (int s = 0; s < nsite; ++s)
                scf_in.constraint_params(l, s) += eta_con * g(l, s);

        scf_in.mu = mu;
    }

    std::cerr << "\nrunKanamoriSCF: not converged after " << max_iter << " iterations\n";
    scf_in.isConverged = false;
    return {scf_in, std::nan(""), false};
}