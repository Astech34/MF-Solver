#pragma once

// Index ordering: (layer, site, spin, orbital), orbital fastest
struct Dims {
    int num_layers   = 0;
    int num_sites    = 0;
    int num_spin     = 2;
    int num_orbitals = 0;

    int total() const { return num_layers * num_sites * num_spin * num_orbitals; }
};
