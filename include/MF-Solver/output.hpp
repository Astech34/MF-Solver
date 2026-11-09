#pragma once

#include <iostream>
#include <string>

#include "MF-Solver/observable.hpp"
#include "MF-Solver/occupations.hpp"

// Print occupations per (layer, site) as a spin × orbital table with totals
void print_occupations(const Occupations& occ, std::ostream& os = std::cout);

// Print moments per (layer, site) as (x, y, z, |m|) with the total,
// e.g. print_moments(compute_SMoments(model, state), "Spin moments")
void print_moments(const Moments& m, const std::string& title, std::ostream& os = std::cout);
