#include <gtest/gtest.h>
#include "MF-Solver/basis_matrix.hpp"
#include "MF-Solver/dims.hpp"
#include "MF-Solver/hamiltonian.hpp"
#include <complex>
#include <Eigen/Dense>

inline Eigen::MatrixXcd test_matrix_16()
{
    // Imaginary parts of columns 0-7  (entry = 0 + x*i)
    static const double im_part[16][8] = {
        { 2.5,  5.0,  3.0,  6.0,  3.0,  6.0,  3.6,  7.2},
        { 7.5, 10.0,  9.0, 12.0,  9.0, 12.0, 10.8, 14.4},
        { 4.0,  8.0,  3.5,  7.0,  4.8,  9.6,  4.2,  8.4},
        {12.0, 16.0, 10.5, 14.0, 14.4, 19.2, 12.6, 16.8},
        { 0.5,  1.0,  0.6,  1.2,  1.0,  2.0,  1.2,  2.4},
        { 1.5,  2.0,  1.8,  2.4,  3.0,  4.0,  3.6,  4.8},
        { 0.8,  1.6,  0.7,  1.4,  1.6,  3.2,  1.4,  2.8},
        { 2.4,  3.2,  2.1,  2.8,  4.8,  6.4,  4.2,  5.6},
        { 7.5, 15.0,  9.0, 18.0,  9.0, 18.0, 10.8, 21.6},
        {22.5, 30.0, 27.0, 36.0, 27.0, 36.0, 32.4, 43.2},
        {12.0, 24.0, 10.5, 21.0, 14.4, 28.8, 12.6, 25.2},
        {36.0, 48.0, 31.5, 42.0, 43.2, 57.6, 37.8, 50.4},
        { 1.5,  3.0,  1.8,  3.6,  3.0,  6.0,  3.6,  7.2},
        { 4.5,  6.0,  5.4,  7.2,  9.0, 12.0, 10.8, 14.4},
        { 2.4,  4.8,  2.1,  4.2,  4.8,  9.6,  4.2,  8.4},
        { 7.2,  9.6,  6.3,  8.4, 14.4, 19.2, 12.6, 16.8}
    };

    // Real parts of columns 8-15  (entry = x + 0*i)
    static const double re_part[16][8] = {
        { 5.0, 10.0,  6.0, 12.0,  6.0, 12.0,  7.2, 14.4},
        {15.0, 20.0, 18.0, 24.0, 18.0, 24.0, 21.6, 28.8},
        { 8.0, 16.0,  7.0, 14.0,  9.6, 19.2,  8.4, 16.8},
        {24.0, 32.0, 21.0, 28.0, 28.8, 38.4, 25.2, 33.6},
        { 1.0,  2.0,  1.2,  2.4,  2.0,  4.0,  2.4,  4.8},
        { 3.0,  4.0,  3.6,  4.8,  6.0,  8.0,  7.2,  9.6},
        { 1.6,  3.2,  1.4,  2.8,  3.2,  6.4,  2.8,  5.6},
        { 4.8,  6.4,  4.2,  5.6,  9.6, 12.8,  8.4, 11.2},
        {10.0, 20.0, 12.0, 24.0, 12.0, 24.0, 14.4, 28.8},
        {30.0, 40.0, 36.0, 48.0, 36.0, 48.0, 43.2, 57.6},
        {16.0, 32.0, 14.0, 28.0, 19.2, 38.4, 16.8, 33.6},
        {48.0, 64.0, 42.0, 56.0, 57.6, 76.8, 50.4, 67.2},
        { 2.0,  4.0,  2.4,  4.8,  4.0,  8.0,  4.8,  9.6},
        { 6.0,  8.0,  7.2,  9.6, 12.0, 16.0, 14.4, 19.2},
        { 3.2,  6.4,  2.8,  5.6,  6.4, 12.8,  5.6, 11.2},
        { 9.6, 12.8,  8.4, 11.2, 19.2, 25.6, 16.8, 22.4}
    };

    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(16, 16);
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 8; ++j) {
            M(i, j)     = std::complex<double>(0.0, im_part[i][j]);
            M(i, j + 8) = std::complex<double>(re_part[i][j], 0.0);
        }
    }
    return M;
}

TEST(BasisMatrix, block_view){
    // Create a sample basis matrix

    using cd = std::complex<double>;
    const cd I(0.0, 1.0);

    Eigen::MatrixXcd Lm    {{1.0*I, 2.0}, {3.0*I, 4.0}};
    Eigen::MatrixXcd Sitem {{0.5,   0.6}, {0.1,   0.2}};
    Eigen::MatrixXcd Spinm {{5.0,   6.0}, {8.0,   7.0}};
    Eigen::MatrixXcd Om    {{1.0,   2.0}, {3.0,   4.0}};

    Eigen::MatrixXcd total_kron = kron(Lm, kron(Sitem, kron(Spinm, Om)));

    Eigen::MatrixXcd test_matrix = test_matrix_16();

    EXPECT_LT((total_kron - test_matrix).norm(), 1e-10);

    Dims dms;
    dms.num_layers = 2;
    dms.num_sites = 2;
    dms.num_spin = 2;
    dms.num_orbitals = 2;

    // Check View
    basis_matrix bm(dms);
    bm.dat = total_kron;

    Eigen::MatrixXcd ls_view = bm.site_block(0, 0);
    Eigen::MatrixXcd ls_view_test {
    { 2.5*I,  5.0*I,  3.0*I,  6.0*I},
    { 7.5*I, 10.0*I,  9.0*I, 12.0*I},
    { 4.0*I,  8.0*I,  3.5*I,  7.0*I},
    {12.0*I, 16.0*I, 10.5*I, 14.0*I}};


    EXPECT_LT((ls_view - ls_view_test).norm(), 1e-10);

    // Another block
    Eigen::MatrixXcd M {
    { 3.0*I,   6.0*I,   3.6*I,   7.2*I },
    { 9.0*I,  12.0*I,  10.8*I,  14.4*I },
    { 4.8*I,   9.6*I,   4.2*I,   8.4*I },
    {14.4*I,  19.2*I,  12.6*I,  16.8*I }
    };
    EXPECT_LT((bm.site_block(1, 1, 0, 1) - M).norm(), 1e-10);

    // Another
    Eigen::MatrixXcd M2 {
    {1.0, 2.0, 1.2, 2.4},
    {3.0, 4.0, 3.6, 4.8},
    {1.6, 3.2, 1.4, 2.8},
    {4.8, 6.4, 4.2, 5.6}
    };
    EXPECT_LT((bm.site_block(0, 1, 1, 0) - M2).norm(), 1e-10);

    //Short hand works
    EXPECT_LT((bm.site_block(1, 1) - bm.site_block(1, 1, 1, 1)).norm(), 1e-10);

    // Setting data
    Eigen::MatrixXcd M9 {
    {9.0, 9.0, 9.2, 9.4},
    {9.0, 9.0, 9.6, 9.8},
    {9.6, 9.2, 9.4, 9.8},
    {9.8, 9.4, 9.2, 9.6}
    };

    bm.site_block(1, 1) = M9;
    EXPECT_LT((bm.site_block(1, 1)-M9).norm(), 1e-10);
}