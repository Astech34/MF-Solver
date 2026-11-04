#include "MF-Solver/interaction.hpp"

basis_matrix kanamori(const hopping_model& hopping, const scf_state& scf, const kfree_params& kfp) {
    const double U  = kfp.U;
    const double Up = kfp.U_prime;
    const double J  = kfp.J;

    // Kanamori is in orbital basis
    // (layer, site, spin, orbital)
    int nm = hopping.dims.num_orbitals;
    int dim_kanamori = 2 * hopping.dims.total(); // total dimension
    basis_matrix H(hopping.dims);

    // We need to iterate through each layer and each site
    // rho indexing is rho(layer, site, spin, orbital,  layer, site, spin, orbital)

    // Iterate through each layer and site
    for (int ly = 0; ly < hopping.dims.num_layers; ly++) {
        for (int st = 0; st < hopping.dims.num_sites; st++) {
            // Access the relevant part of the rho matrix for this layer and site
            // rho(layer, site, spin, orbital, layer, site, spin, orbital)

            //Kanamori MF Begin (spin, orbital)
            double n_up = 0.0, n_dn = 0.0;
            for (int m = 0; m < nm; m++) {
                n_up += scf.rho(ly, st, 0, m,   ly, st, 0, m  ).real();
                n_dn += scf.rho(ly, st, 1, m,   ly, st, 1, m  ).real();
            }

            // Diagonal terms
            for (int m = 0; m < nm; m++) {
                const double n_up_m = scf.rho(ly, st, 0, m,   ly, st, 0, m  ).real();
                const double n_dn_m = scf.rho(ly, st, 1, m,   ly, st, 1, m  ).real();

                // Up Diagonal
                H(ly, st, 0, m,   ly, st, 0, m  ) += U  * n_dn_m
                            + Up * (n_dn - n_dn_m);
                            //+ (Up - J) * (n_up - n_up_m);
                
                // Down Diagonal
                H(ly, st, 1, m, ly, st, 1, m) += U  * n_up_m
                            + Up * (n_up - n_up_m);
                            //+ (Up - J) * (n_dn - n_dn_m);
                
                // Hubbard U Off Diagonal Terms
                H(ly, st, 1, m, ly, st, 0, m) -= U * scf.rho(ly, st, 0, m, ly, st, 1, m); // -U <d†_{m↑} d_{m↓}> d†_{m↓} d_{m↑}
                H(ly, st, 0, m, ly, st, 1, m) -= U * scf.rho(ly, st, 1, m, ly, st, 0, m); // -U <d†_{m↓} d_{m↑}> d†_{m↑} d_{m↓}
            }

            // Off-diagonal terms: exchange (H_exc) and pair-hopping (H_ph)
            for (int m = 0; m < nm; m++) {
                for (int mp = 0; mp < nm; mp++) {
                    if (mp == m) continue;

                    // Diagonal U'-J
                    // Up Spin
                    H(ly, st, 0, m, ly, st, 0, m) += 0.5 * (Up - J) * scf.rho(ly, st, 0, mp, ly, st, 0, mp);
                    H(ly, st, 0, mp, ly, st, 0, mp) += 0.5 * (Up - J) * scf.rho(ly, st, 0, m, ly, st, 0, m);
                    // Down Spin
                    H(ly, st, 1, m, ly, st, 1, m) += 0.5 * (Up - J) * scf.rho(ly, st, 1, mp, ly, st, 1, mp);
                    H(ly, st, 1, mp, ly, st, 1, mp) += 0.5 * (Up - J) * scf.rho(ly, st, 1, m, ly, st, 1, m);

                    // U' off diagonal terms
                    H(ly, st, 1, mp, ly, st, 0, m) -= Up * scf.rho(ly, st, 0, m, ly, st, 1, mp);
                    H(ly, st, 0, m,  ly, st, 1, mp) -= Up * scf.rho(ly, st, 1, mp, ly, st, 0, m);

                    // U'-J off diagonal
                    // Up Spin
                    H(ly, st, 0, mp, ly, st, 0, m) -= 0.5 * (Up - J) * scf.rho(ly, st, 0, m, ly, st, 0, mp);
                    H(ly, st, 0, m, ly, st, 0, mp) -= 0.5 * (Up - J) * scf.rho(ly, st, 0, mp, ly, st, 0, m);
                    // Down Spin
                    H(ly, st, 1, mp, ly, st, 1, m) -= 0.5 * (Up - J) * scf.rho(ly, st, 1, m, ly, st, 1, mp);
                    H(ly, st, 1, m, ly, st, 1, mp) -= 0.5 * (Up - J) * scf.rho(ly, st, 1, mp, ly, st, 1, m);

                    // H_exc 1: <d†_{m↑} d_{m'↑}> d†_{m'↓} d_{m↓}
                    H(ly, st, 1, mp, ly, st, 1, m) += J * scf.rho(ly, st, 0, m, ly, st, 0, mp);
                    // H_ph  1: <d†_{m↑} d_{m'↑}> d†_{m↓} d_{m'↓}
                    H(ly, st, 1, m, ly, st, 1, mp) += J * scf.rho(ly, st, 0, m, ly, st, 0, mp);

                    // H_exc 2 + H_ph 2: (<d†_{m'↓}d_{m↓}> + <d†_{m↓}d_{m'↓}>) d†_{m↑}d_{m'↑}
                    H(ly, st, 0, m, ly, st, 0, mp) += J * (scf.rho(ly, st, 1, mp, ly, st, 1, m) + scf.rho(ly, st, 1, m, ly, st, 1, mp));

                    // H_exc 3: -<d†_{m↑} d_{m↓}> d†_{m'↓} d_{m'↑}
                    H(ly, st, 1,mp, ly, st, 0, mp) -= J * scf.rho(ly, st, 0, m, ly, st, 1, m);
                    // H_ph  3: -<d†_{m↑} d_{m'↓}> d†_{m↓} d_{m'↑}
                    H(ly, st, 1, m, ly, st, 0, mp) -= J * scf.rho(ly, st, 0, m, ly, st, 1, mp);

                    // H_exc 4: -<d†_{m'↓} d_{m'↑}> d†_{m↑} d_{m↓}
                    H(ly, st, 0, m, ly, st, 1, m) -= J * scf.rho(ly, st, 1, mp, ly, st, 0, mp);
                    // H_ph  4: -<d†_{m↓} d_{m'↑}> d†_{m↑} d_{m'↓}
                    H(ly, st, 0, m, ly, st, 1, mp) -= J * scf.rho(ly, st, 1, m, ly, st, 0, mp);
                }
            }
            
            // Kanamori MF End (spin, orbital)
        }
    }

    // Return the constructed Hamiltonian matrix H
    // Kanamori is in spin . orbital basis
    // (layer, site, spin, orbital)
    return H;
}