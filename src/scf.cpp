#include "MF-Solver/scf.hpp"
#include "MF-Solver/occupations.hpp"
#include "MF-Solver/energy.hpp"
#include "MF-Solver/model.hpp"

KanamoriResult runKanamoriSCF(const hopping_model& hopping, const scf_state& scf,
                               double alpha, int grid_size,
                               double T, double N_target,
                               const Params& p, const KanamoriParams& kp,
                               MixerType mixer, int max_iter, int comp_dft) {

    double tol        = 1e-9; // make from import later
    constexpr int    diis_max   = 10;
    constexpr int    diis_start = 30;

    eigen::MatrixXcd rho = scf.rho.dat;

    // DFT Constraint.
    if (comp_dft == 1.0) {
        // To Compare with Collinear DFT+U without SOC we make real
        rho = rho.real().cast<cd>();
        // Then we set off diagonal elements to zero.
        auto spin_of = [](int i) { return (i / 3) % 2; };
        for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 12; ++j)
            if (spin_of(i) != spin_of(j))
                rho(i, j) = cd(0.0, 0.0);
    }


    // --- Constraint multiplier state ---
    const bool solve_lam = (p.con_eta > 0.0);
    double     lam1       = p.con_lam1;
    double     lam2       = p.con_lam2;
    
    //double     lam_L     = p.con_lam_L;
    Params     p_iter    = p;  // copy of p whose con_lam is refreshed every iteration

    // Frobenius inner product Re<A, B>_F (matches the DIIS residual metric).
    auto frob = [](const Mat12& A, const Mat12& B) {
        return (A.array().conjugate() * B.array()).sum().real();
    };

    // --- LinearDIIS state ---
    std::vector<Mat12>  diis_rho;      // rho_new history
    std::vector<Mat12>  diis_err;      // residual history: e_i = rho_new_i - rho_i
    
    std::vector<double> diis_lam1;      // con_lam_new history (parallel to diis_rho)
    std::vector<double> diis_lam_err1;  // con_lam residual history
    std::vector<double> diis_lam2;      // con_lam_new history (parallel to diis_rho)
    std::vector<double> diis_lam_err2;  // con_lam residual history
    
    //std::vector<double> diis_lam_L;      // con_lam_L_new history (parallel to diis_rho)
    //std::vector<double> diis_lam_L_err;  // con_lam_L residual history

    constexpr int no_improve_max    = 9999999;
    constexpr int linear_reset_steps = 150;
    double best_diis_diff   = std::numeric_limits<double>::max();
    int    no_improve_count = 0;
    int    linear_remaining = 0;  // counts down during post-stall linear phase

    for (int i = 0; i < max_iter; i++) {
        p_iter.con_lam1 = lam1;
        p_iter.con_lam2 = lam2;

        const Eigensystem sys = compute_eigensystem_kanamori(rho, grid_size, p_iter, kp);
        const double mu = find_mu(sys, T, N_target);
        if (std::isnan(mu)) {
            std::cout << "runKanamoriSCF: could not find mu at iteration " << i << "\n";
            break;
        }

        const Mat12 rho_new = compute_density_matrix(sys, mu, T);

        // Constraint channel: violation of the new density matrix and the dual step it
        // implies. Both are identically zero when con_lam is not being solved for.
        const auto [g1, g2, g1_L, g2_L]  = constraint_violation(rho_new, p_iter);
        const double g       = solve_lam ? (g1 + g2) : 0.0;
        //const double g_L    = solve_lam ? (g1_L + g2_L) : 0.0;

        //Spin Constraint Params
        const double lam_new1 = lam1 + p.con_eta * g1;
        const double lam_err1 = lam_new1 - lam1;
        const double lam_new2 = lam2 + p.con_eta * g2;
        const double lam_err2 = lam_new2 - lam2;

        const double rho_diff = (rho_new - rho).norm();
        const double lam_diff = std::sqrt(lam_err1 * lam_err1 + lam_err2 * lam_err2);
        // Aggregate for stall tracker
        const double diff     = std::sqrt(rho_diff * rho_diff + lam_err1 * lam_err1 + lam_err2 * lam_err2);


        const bool using_diis = (mixer == MixerType::LinearDIIS)
                              && (i >= diis_start) && (linear_remaining == 0);
        const char* tag = using_diis ? " [DIIS]" : "  [mix]";
        std::cout << "\rIterations" << std::setw(5) << i << tag
                  << ", |Δρ|_F = " << std::scientific << std::setprecision(4) << rho_diff;
        if (solve_lam)
            std::cout << ", con_lam1 = " << std::fixed << std::setprecision(6) << lam1
        << ", con_lam2 = " << std::fixed << std::setprecision(6) << lam2
                      << ", g = " << std::scientific << std::setprecision(4) << g;
        std::cout << "   " << std::flush;

        bool is_done = false;
        if (using_diis){
            if (rho_diff < tol && lam_diff < p.thresh_con) {
                is_done = true;
            }
        }
        else{
            if (rho_diff < tol){
                is_done = true;
            }
        }

        if (is_done) {
            const double bandsum = calculate_band_energy(sys, mu, T);
            const double dc = kanamori_dc_layer(rho.block<6,6>(0, 0), kp)
                            + kanamori_dc_layer(rho.block<6,6>(6, 6), kp);
            std::cout << "\nKanamori SCF converged! mu = " << mu
                      << "\n Band Energy = " << bandsum
                      << "\n DC Correction = " << dc
                      << ", E_total = " << bandsum - dc << "\n";
            if (solve_lam)
                std::cout << " con_lam1 = " << lam1
                          << " con_lam2 = " << lam2
                          << ", constraint violation spin g = " << g << "\n";
                          //<< " con_lam_L = " << lam_L
                          //<< ", constraint violation orbital g = " << g_L << "\n";
            return {rho0, rho, mu, bandsum - dc, true, lam1, lam2};
        }
        // ====== Linear Mix ======
        if (!using_diis) {
            if (linear_remaining > 0) --linear_remaining;
            rho = alpha * rho_new + (1.0 - alpha) * rho;
            lam1 = alpha * lam_new1 + (1.0 - alpha) * lam1;
            lam2 = alpha * lam_new2 + (1.0 - alpha) * lam2;
            //lam_L = alpha * lam_L_new + (1.0 - alpha) * lam_L;
        } 
        // ====== DIIS ====== 
        else {
            // Track improvement; on stall, do linear_reset_steps of linear mixing then retry DIIS.
            if (diff < best_diis_diff) {
                best_diis_diff  = diff;
                no_improve_count = 0;
            } else {
                ++no_improve_count;
            }

            if (no_improve_count >= no_improve_max && !is_done) {
                std::cout << "\n[DIIS stalled " << no_improve_max
                          << " iters — " << linear_reset_steps << " linear mix steps]\n";
                diis_rho.clear();
                diis_err.clear();
                diis_lam1.clear();
                diis_lam_err1.clear();
                diis_lam2.clear();
                diis_lam_err2.clear();
                //diis_lam_L.clear();
                //diis_lam_L_err.clear();
                best_diis_diff   = std::numeric_limits<double>::max();
                no_improve_count = 0;
                linear_remaining = linear_reset_steps;
                rho = alpha * rho_new + (1.0 - alpha) * rho;
                lam1 = alpha * lam_new1 + (1.0 - alpha) * lam1;
                lam2 = alpha * lam_new2 + (1.0 - alpha) * lam2;
                //lam_L = alpha * lam_L_new + (1.0 - alpha) * lam_L;
                continue;
            }

            // ===== Pulay DIIS ====
            // History entries are the full pair (rho_new, con_lam_new) with the matching
            // pair of residuals; the extrapolation coefficients c_i are shared, so the
            // multiplier is extrapolated on exactly the same footing as the density matrix.
            diis_rho.push_back(rho_new);
            diis_err.push_back(rho_new - rho);
            
            diis_lam1.push_back(lam_new1);
            diis_lam_err1.push_back(lam_err1);
            diis_lam2.push_back(lam_new2);
            diis_lam_err2.push_back(lam_err2);
            
            //diis_lam_L.push_back(lam_L_new);
            //diis_lam_L_err.push_back(lam_L_err);

            if (static_cast<int>(diis_rho.size()) > diis_max) {
                diis_rho.erase(diis_rho.begin());
                diis_err.erase(diis_err.begin());
                
                diis_lam1.erase(diis_lam1.begin());
                diis_lam_err1.erase(diis_lam_err1.begin());
                diis_lam2.erase(diis_lam2.begin());
                diis_lam_err2.erase(diis_lam_err2.begin());
                //diis_lam_L.erase(diis_lam_L.begin());
                //diis_lam_L_err.erase(diis_lam_L_err.begin());
            }

            const int m = static_cast<int>(diis_rho.size());

            if (m < 2) {
                rho = rho_new;
                lam1 = lam_new1;
                lam2 = lam_new2;
                //lam_L = lam_L_new;
            } else {
                // Build (m+1)×(m+1) Pulay system
                //   [B   -1] [c]   [ 0]
                //   [-1   0] [λ] = [-1]
                // B_ij = Re(<e_i, e_j>_F) + e^lam_i e^lam_j,  Σc_i = 1 enforced by λ
                Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m + 1, m + 1);
                Eigen::VectorXd b = Eigen::VectorXd::Zero(m + 1);
                b(m) = -1.0;

                for (int ii = 0; ii < m; ii++) {
                    for (int jj = ii; jj < m; jj++) {
                        const double Bij = frob(diis_err[ii], diis_err[jj])
                                         + diis_lam_err1[ii] * diis_lam_err1[jj]
                                         + diis_lam_err2[ii] * diis_lam_err2[jj];
                                         //+ diis_lam_L_err[ii] * diis_lam_L_err[jj];
                        A(ii, jj) = Bij;
                        A(jj, ii) = Bij;
                    }
                    A(ii, m) = -1.0;
                    A(m, ii) = -1.0;
                }

                const Eigen::VectorXd c = A.colPivHouseholderQr().solve(b);

                rho = Mat12::Zero();
                lam1 = 0.0;
                lam2 = 0.0;
                //lam_L = 0.0;
                for (int ii = 0; ii < m; ii++) {
                    rho.noalias() += c(ii) * diis_rho[ii];
                    lam1           += c(ii) * diis_lam1[ii];
                    lam2           += c(ii) * diis_lam2[ii];
                    //lam_L         += c(ii) * diis_lam_L[ii];
                }
            }
        }
    }

    //throw std::runtime_error("runKanamoriSCF: failed to converge within max_iter");
    return {rho0, rho, 0, 0, false, lam1, lam2};
}