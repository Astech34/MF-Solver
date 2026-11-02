#include "MF-Solver/hamiltonian.hpp"
#include "MF-Solver/scf.hpp"

Eigen::MatrixXcd kanamori(const hopping_model& hopping, const scf_state& scf, const kfree_params& kfp) {
    const double U  = kfp.U;
    const double Up = kfp.U_prime;
    const double J  = kfp.J;

    // Kanamori is in orbital basis
    // (layer, site, spin, orbital)
    int nm = hopping.num_orbitals;
    int dim_kanamori = 2 * hopping.dim(); // total dimension
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(dim_kanamori, dim_kanamori);

    // We need to iterate through each layer and each site

    //Kanamori MF Begin
    double n_up = 0.0, n_dn = 0.0;
    for (int m = 0; m < nm; m++) {
        n_up += scf.rho(m,   m  ).real();
        n_dn += scf.rho(m+nm, m+nm).real();
    }

    // Diagonal terms
    for (int m = 0; m < nm; m++) {
        const double n_up_m = scf.rho(m,   m  ).real();
        const double n_dn_m = scf.rho(m+nm, m+nm).real();

        // Up Diagonal
        H(m,   m  ) += U  * n_dn_m
                     + Up * (n_dn - n_dn_m);
                     //+ (Up - J) * (n_up - n_up_m);
        
        // Down Diagonal
        H(m+nm, m+nm) += U  * n_up_m
                     + Up * (n_up - n_up_m);
                     //+ (Up - J) * (n_dn - n_dn_m);
        
        // Hubbard U Off Diagonal Terms
        H(m+nm, m) -= U * scf.rho(m, m+nm); // -U <d†_{m↑} d_{m↓}> d†_{m↓} d_{m↑}
        H(m, m+nm) -= U * scf.rho(m+nm, m); // -U <d†_{m↓} d_{m↑}> d†_{m↑} d_{m↓}
    }

    // Off-diagonal terms: exchange (H_exc) and pair-hopping (H_ph)
    for (int m = 0; m < nm; m++) {
        for (int mp = 0; mp < nm; mp++) {
            if (mp == m) continue;

            // Diagonal U'-J
            // Up Spin
            H(m,m) += 0.5 * (Up - J) * scf.rho(mp, mp);
            H(mp,mp) += 0.5 * (Up - J) * scf.rho(m, m);
            // Up Spin
            H(m+nm,m+nm) += 0.5 * (Up - J) * scf.rho(mp+nm, mp+nm);
            H(mp+nm,mp+nm) += 0.5 * (Up - J) * scf.rho(m+nm, m+nm);

            // U' off diagonal terms
            H(mp+nm, m) -= Up * scf.rho(m, mp+nm);
            H(m, mp+nm) -= Up * scf.rho(mp+nm, m);

            // U'-J off diagonal
            // Up Spin
            H(mp, m) -= 0.5 * (Up - J) * scf.rho(m, mp);
            H(m, mp) -= 0.5 * (Up - J) * scf.rho(mp, m);
            // Down Spin
            H(mp+nm, m+nm) -= 0.5 * (Up - J) * scf.rho(m+nm, mp+nm);
            H(m+nm, mp+nm) -= 0.5 * (Up - J) * scf.rho(mp+nm, m+nm);

            // H_exc 1: <d†_{m↑} d_{m'↑}> d†_{m'↓} d_{m↓}
            H(mp+nm, m+nm) += J * scf.rho(m, mp);
            // H_ph  1: <d†_{m↑} d_{m'↑}> d†_{m↓} d_{m'↓}
            H(m+nm, mp+nm) += J * scf.rho(m, mp);

            // H_exc 2 + H_ph 2: (<d†_{m'↓}d_{m↓}> + <d†_{m↓}d_{m'↓}>) d†_{m↑}d_{m'↑}
            H(m, mp) += J * (scf.rho(mp+nm, m+nm) + scf.rho(m+nm, mp+nm));

            // H_exc 3: -<d†_{m↑} d_{m↓}> d†_{m'↓} d_{m'↑}
            H(mp+nm, mp) -= J * scf.rho(m, m+nm);
            // H_ph  3: -<d†_{m↑} d_{m'↓}> d†_{m↓} d_{m'↑}
            H(m+nm, mp) -= J * scf.rho(m, mp+nm);

            // H_exc 4: -<d†_{m'↓} d_{m'↑}> d†_{m↑} d_{m↓}
            H(m, m+nm) -= J * scf.rho(mp+nm, mp);
            // H_ph  4: -<d†_{m↓} d_{m'↑}> d†_{m↑} d_{m'↓}
            H(m, mp+nm) -= J * scf.rho(m+nm, mp);
        }
    }
    
    // Kanamori MF End
    
    assert((H - H.adjoint()).norm() < 1e-10 && "H is not Hermitian!");

    // Return the constructed Hamiltonian matrix H
    // Kanamori is in spin . orbital basis
    // (layer, site, spin, orbital)
    return kron(hopping.ILayer(), kron(hopping.ISite(), H));
}