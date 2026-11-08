#include <iomanip>
#include <iostream>

#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/initial_density.hpp"
#include "MF-Solver/models/kagome.hpp"
#include "MF-Solver/params.hpp"
#include "MF-Solver/bands.hpp"
#include "MF-Solver/scf.hpp"

// Single Kanamori self-consistent mean-field calculation on the kagome model.
int main() {
    // --- Model -------------------------------------------------------------
    hopping_model model = make_kagome({1.0});  // t = 1.0

    // --- Interaction / SOC parameters -------------------------------------
    kfree_params kfp;  // theta = phi = 0
    kfp.lambda  = 0.5;
    kfp.U       = 3.0;
    kfp.J       = 0.5;
    kfp.U_prime = kfp.U - 2.0 * kfp.J;

    kfp.theta = M_PI / 2;
    kfp.phi = 0.0;

    // --- Simulation parameters --------------------------------------------
    simulation_params sim;  // grid_size = 100, T = 0.05, N_target = 12
    // Half filling: dims.total() single-particle states per k-point
    sim.N_target = 0.5 * model.dims.total();
    const double alpha    = 0.3;   // linear mixing weight for rho
    const int    max_iter = 500;
    const bool   comp_dft = false; // true: collinear, no-SOC projection of rho

    // --- Initial state: unpolarized, half filled, zero constraint fields ---
    scf_state state(model.dims);
    initialize_rho(state.rho, rho_init_params{});

    // --- Run SCF -----------------------------------------------------------
    const KanamoriResult result = runKanamoriSCF(
        model, state, kfp, alpha, sim.grid_size, sim.T, sim.N_target,
        MixerType::LinearDIIS, max_iter, comp_dft);

    // --- Report ------------------------------------------------------------
    if (!result.converged) {
        std::cerr << "SCF did not converge.\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(8)
              << "E_total = " << result.energy << "\n"
              << "mu      = " << result.state.mu << "\n";

    std::cout << "Saving Bands...\n";
    std::string filename = "runs/bands.csv";
    save_bands(model, result.state, kfp, 100, filename);

    return 0;
}
