#include <iomanip>
#include <iostream>

#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/initial_density.hpp"
#include "MF-Solver/models/kagome.hpp"
#include "MF-Solver/output.hpp"
#include "MF-Solver/params.hpp"
#include "MF-Solver/bands.hpp"
#include "MF-Solver/scf.hpp"

// Single Kanamori self-consistent mean-field calculation on the kagome model.
int main() {
    // --- Model -------------------------------------------------------------
    hopping_model model = make_kagome({1.0});  // t = 1.0

    // --- Interaction / SOC parameters -------------------------------------
    kfree_params kfp;  // theta = phi = 0
    kfp.lambda  = 0.1;
    kfp.U       = 5.0;
    kfp.J       = 0.7;
    kfp.U_prime = kfp.U - 2.0 * kfp.J;
    //kfp.U_prime = kfp.J;

    kfp.theta = 0.0;
    kfp.phi = 0.0;

    // --- Simulation parameters --------------------------------------------
    simulation_params sim;  // grid_size = 100, T = 0.05, N_target = 12
    // Half filling: dims.total() single-particle states per k-point
    sim.N_target = 2.0;
    sim.grid_size = 20;
    const double alpha    = 0.4;   // linear mixing weight for rho
    const int    max_iter = 99999;
    const bool   comp_dft = false; // true: collinear, no-SOC projection of rho

    // --- Initial state: unpolarized, half filled, zero constraint fields ---
    scf_state state(model.dims);
    rho_init_params rp;
    rp.kind = rho_init_kind::random_seed;
    rp.seed = 3;
    rp.N_target = sim.N_target;
    initialize_rho(state.rho, rp);

    // --- Run SCF -----------------------------------------------------------
    const KanamoriResult result = runKanamoriSCF(
        model, state, kfp, alpha, sim.grid_size, sim.T, sim.N_target,
        MixerType::DIIS, max_iter, comp_dft);

    // --- Report ------------------------------------------------------------
    if (!result.converged) {
        std::cerr << "SCF did not converge.\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(8)
              << "E_total = " << result.energy << "\n"
              << "mu      = " << result.state.mu << "\n\n";

    print_occupations(compute_occupations(result.state));
    std::cout << "\n";
    print_moments(compute_SMoments(model, result.state), "Spin moments");
    std::cout << "\n";
    print_moments(compute_LMoments(model, result.state), "Orbital moments");
    std::cout << "\n";

    std::cout << "Saving Bands...\n";
    std::string filename = "runs/bands.csv";
    save_bands(model, result.state, kfp, 100, filename);

    return 0;
}
