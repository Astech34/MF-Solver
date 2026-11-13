#include <string>

#include "MF-Solver/bands.hpp"
#include "MF-Solver/models/kagome.hpp"
#include "MF-Solver/initial_density.hpp"

int main(){
    hopping_model model = make_kagome({1.0}); // t = 1.0
    int num_points = 100;
    kfree_params kfp; // lambda = theta = phi = 0
    kfp.lambda = 0.5;
    kfp.U = 5.0;
    kfp.J = 0.5;
    kfp.U_prime = kfp.U - 2 * kfp.J;

    kfp.theta = M_PI / 2;

    std::string filename = "runs/bands.csv";
    
    // Create scf state and initalize rho (for testing)
    scf_state scf(model.dims);

    rho_init_params rho_params;
    rho_params.kind = rho_init_kind::unpolarized_hfill;
    initialize_rho(scf.rho, rho_params);

    save_bands(model, scf, kfp, num_points, filename);

    // DOS: 100x100 k-grid, 1000 energies, Gaussian broadening sigma = 0.05 t
    save_dos(model, scf, kfp, 100, 1000, 0.05, "runs/dos.csv");
}
