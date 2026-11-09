#include <cmath>
#include <iomanip>
#include <iostream>
#include <optional>
#include <vector>

#include "MF-Solver/bands.hpp"
#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/initial_density.hpp"
#include "MF-Solver/models/kagome.hpp"
#include "MF-Solver/output.hpp"
#include "MF-Solver/params.hpp"
#include "MF-Solver/scf.hpp"

// Ground-state search for the Kanamori mean-field kagome model.
// Runs the SCF from successive random seeds until n_match seeds converge to the
// same lowest energy (within e_tol), guarding against landing in a metastable state.
int main() {
    // --- Model -------------------------------------------------------------
    hopping_model model = make_kagome({1.0});  // t = 1.0

    // --- Interaction / SOC parameters -------------------------------------
    kfree_params kfp;  // theta = phi = 0
    kfp.lambda  = 0.1;
    kfp.U       = 5.0;
    kfp.J       = 0.7;
    kfp.U_prime = kfp.U - 2.0 * kfp.J;

    kfp.theta = 0.0;
    kfp.phi   = 0.0;

    // --- Simulation parameters --------------------------------------------
    simulation_params sim;
    sim.N_target  = 2.0;
    sim.grid_size = 20;
    const double alpha    = 0.4;    // density damping
    const int    max_iter = 99999;
    const bool   comp_dft = false;  // true: collinear, no-SOC projection of rho

    // --- Ground-state search parameters -----------------------------------
    const int    first_seed = 1;
    const int    max_seeds  = 30;    // cap on the number of SCF runs
    const int    n_match    = 5;     // seeds that must agree on the lowest energy
    const double e_tol      = 1e-5;  // energies within e_tol count as the same state

    // --- Seed loop ---------------------------------------------------------
    std::optional<KanamoriResult> best;
    std::vector<int> best_seeds;     // seeds that reached the current lowest energy

    for (int seed = first_seed; seed < first_seed + max_seeds; ++seed) {
        scf_state state(model.dims);
        rho_init_params rp;
        rp.kind     = rho_init_kind::random_seed;
        rp.seed     = seed;
        rp.N_target = sim.N_target;
        initialize_rho(state.rho, rp);

        const KanamoriResult result = runKanamoriSCF(
            model, state, kfp, alpha, sim.grid_size, sim.T, sim.N_target,
            MixerType::DIIS, max_iter, comp_dft);

        std::cout << std::fixed << std::setprecision(8) << "Seed " << seed << ": ";
        if (!result.converged) {
            std::cout << "not converged\n\n";
            continue;
        }

        if (!best || result.energy < best->energy - e_tol) {
            best = result;
            best_seeds = {seed};
        } else if (std::abs(result.energy - best->energy) <= e_tol) {
            best_seeds.push_back(seed);
        }

        std::cout << "E = " << result.energy
                  << "   lowest = " << best->energy
                  << "   (" << best_seeds.size() << "/" << n_match << " seeds)\n\n";

        if (static_cast<int>(best_seeds.size()) >= n_match) break;
    }

    // --- Report ------------------------------------------------------------
    if (!best) {
        std::cerr << "No seed converged.\n";
        return 1;
    }

    const bool confirmed = static_cast<int>(best_seeds.size()) >= n_match;
    if (!confirmed)
        std::cerr << "Warning: seed cap reached; lowest energy found by only "
                  << best_seeds.size() << "/" << n_match << " seeds.\n";

    std::cout << (confirmed ? "Ground state" : "Lowest state found")
              << " (seeds:";
    for (int s : best_seeds) std::cout << " " << s;
    std::cout << ")\n"
              << "E_total = " << best->energy << "\n"
              << "mu      = " << best->state.mu << "\n\n";

    print_occupations(compute_occupations(best->state));
    std::cout << "\n";
    print_moments(compute_SMoments(model, best->state), "Spin moments");
    std::cout << "\n";
    print_moments(compute_LMoments(model, best->state), "Orbital moments");
    std::cout << "\n";

    std::cout << "Saving Bands...\n";
    save_bands(model, best->state, kfp, 100, "runs/bands.csv");

    return confirmed ? 0 : 1;
}
