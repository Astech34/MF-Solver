#pragma once

struct simulation_params{
    int grid_size = 100;  // Number of k-points along each dimension
    double T = 0.05;
    double N_target = 12.0;
}