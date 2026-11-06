#include "MF-Solver/kspace.hpp"

// -----------------------------------------------------------------------------
// Brent's method
// Finds x in [a, b] such that f(x) = 0, given f(a)*f(b) < 0.
// Combines bisection, secant, and inverse quadratic interpolation.
// -----------------------------------------------------------------------------
double brent(std::function<double(double)> f, double a, double b,
             double tol, int max_iter) {
    double fa = f(a), fb = f(b);
    assert(fa * fb < 0.0 && "brent: bracket [a,b] must straddle zero");

    // Ensure |f(b)| <= |f(a)| — b is always the best guess
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c  = a, fc = fa;
    double s  = b, fs = fb;
    double d  = 0.0;
    bool   mflag = true;  // bisection was used last iteration

    for (int i = 0; i < max_iter; i++) {
        if (std::abs(b - a) < tol || std::abs(fs) < std::numeric_limits<double>::epsilon())
            return s;

        if (fa != fc && fb != fc) {
            // Inverse quadratic interpolation
            s = a*fb*fc / ((fa-fb)*(fa-fc))
              + b*fa*fc / ((fb-fa)*(fb-fc))
              + c*fa*fb / ((fc-fa)*(fc-fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Conditions to fall back to bisection
        const double mid = (a + b) / 2.0;
        const bool cond1 = !((3*a + b)/4.0 < s && s < b) && !((3*a + b)/4.0 > s && s > b);
        const bool cond2 =  mflag && std::abs(s - b) >= std::abs(b - c) / 2.0;
        const bool cond3 = !mflag && std::abs(s - b) >= std::abs(c - d) / 2.0;
        const bool cond4 =  mflag && std::abs(b - c) < tol;
        const bool cond5 = !mflag && std::abs(c - d) < tol;

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s     = mid;
            mflag = true;
        } else {
            mflag = false;
        }

        fs = f(s);
        d  = c;
        c  = b; fc = fb;

        if (fa * fs < 0.0) { b = s; fb = fs; }
        else               { a = s; fa = fs; }

        // Keep b as the best guess
        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }

    throw std::runtime_error("brent: failed to converge within max_iter");
}

// -----------------------------------------------------------------------------
// NTotal — Fermi-Dirac sum over all k-points and bands, normalised per k-point
// -----------------------------------------------------------------------------
double NTotal(const Eigensystem& sys, double mu, double T) {
    const int N_k     = static_cast<int>(sys.evals.size());
    const int N_bands = 12;

    double n = 0.0;
    for (const auto& ev : sys.evals)
        for (int b = 0; b < N_bands; b++) {
            const double x = std::clamp((ev[b] - mu) / T, -500.0, 500.0);
            n += 1.0 / (std::exp(x) + 1.0);
        }
    return n / static_cast<double>(N_k);
}

// -----------------------------------------------------------------------------
// find_mu — mirrors scipy brentq on NTotal(mu) - N_target
// -----------------------------------------------------------------------------
// Check normalization over k vs N particles
double find_mu(const Eigensystem& sys, double T, double N_target) {

    // Find the lowest and highest energy along k points
    // to bound the chemical potential.
    double e_min =  std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();

    for (const auto& ev : sys.evals) {
        e_min = std::min(e_min, ev.minCoeff());
        e_max = std::max(e_max, ev.maxCoeff());
    }

    // Add some padding to either end 
    const double mu_min = e_min - 5.0 * T;
    const double mu_max = e_max + 5.0 * T;

    try {
        return brent([&](double mu) { return NTotal(sys, mu, T) - N_target; }, mu_min, mu_max);
    } catch (const std::runtime_error&) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

basis_matrix compute_density_matrix(const Eigensystem& sys, const Dims dims, double mu, double T) {
    const int N_k = static_cast<int>(sys.evals.size());

    eigen::MatrixXcd rho = eigen::MatrixXcd::Zero(dims.total(), dims.total());

    #pragma omp parallel
    {
        eigen::MatrixXcd rho_local = eigen::MatrixXcd::Zero(dims.total(), dims.total());

        #pragma omp for schedule(static)
        for (int idx = 0; idx < N_k; idx++) {
            for (int band = 0; band < 12; band++) {
                const double x = std::clamp((sys.evals[idx][band] - mu) / T, -500.0, 500.0);
                const double f = 1.0 / (std::exp(x) + 1.0);
                const Eigen::Vector<cd, 12> v = sys.evecs[idx].col(band);
                rho_local.noalias() += f * v.conjugate() * v.transpose(); 
            }
        }

        #pragma omp critical
        rho += rho_local;
    }

    basis_matrix r_rho;
    r_rho.dat = rho / static_cast<double>(N_k);

    return r_rho;
}