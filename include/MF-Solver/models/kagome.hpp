#pragma once

#include "MF-Solver/model.hpp"

struct KagomeParams{
    double t; //hopping parameter
};

hopping_model make_kagome(KagomeParams params);