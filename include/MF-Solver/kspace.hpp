#pragma once

#include <Eigen/Dense>
#include <vector>

struct Eigensystem {
    std::vector<Eigen::VectorXd> evals;
    std::vector<Eigen::MatrixXd> evecs;
};