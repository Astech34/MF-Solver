#include <string>

#include "MF-Solver/bands.hpp"
#include "MF-Solver/models/kagome.hpp"

int main(){
    hopping_model model = make_kagome({1.0}); // t = 1.0
    int num_points = 100;
    std::string filename = "runs/bands.csv";
    kfree_params kfp; // lambda = theta = phi = 0
    kfp.lambda = 0.5;
    kfp.U = 5.0;
    kfp.J = 0.5;
    kfp.U_prime = kfp.U - 2 * kfp.J;
    save_bands(model, kfp, num_points, filename);
}
