#include <string>

#include "MF-Solver/bands.hpp"
#include "MF-Solver/models/kagome.hpp"

int main(){
    hopping_model model = make_kagome({1.0}); // t = 1.0
    int num_points = 100;
    std::string filename = "runs/bands.csv";
    save_bands(model, num_points, filename);
}