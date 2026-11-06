
double calculate_band_energy(const Eigensystem& sys, double mu, double T) {
    const int N_k     = static_cast<int>(sys.evals.size());
    const int N_bands = 12;

    double E = 0.0;
    for (const auto& ev : sys.evals) {
        for (int b = 0; b < N_bands; b++) {
            const double x = std::clamp((ev[b] - mu) / T, -500.0, 500.0);
            const double f = 1.0 / (std::exp(x) + 1.0);
            E += ev[b] * f;
        }
    }
    return E / static_cast<double>(N_k);
}

double kanamori_dc(const hopping_model& hopping, const scf_state& scf, const kfree_params& kfp) {
    const double U  = kp.U;
    const double Up = kp.U_prime;
    const double J  = kp.J;

    double dc = 0.0;

    // Also Sum over all layers and sites:
    // Kanamori is in orbital basis
    // (layer, site, spin, orbital)
    int nm = hopping.dims.num_orbitals;
    int dim_kanamori = 2 * hopping.dims.total(); // total dimension

    // Iterate through each layer and site
    for (int ly = 0; ly < hopping.dims.num_layers; ly++) {
        for (int st = 0; st < hopping.dims.num_sites; st++) {
            // Spin/Orbit Start Here
            // -U intraorbital:
            for (int m = 0; m < nm; m++){
                dc -= U * (rho(ly, st, 0, m,  ly, st, 1, m)*rho(ly, st, 1, m, ly, st, 0, m) - 
                            rho(ly, st, 0, m, ly, st, 0, m)*rho(ly, st, 1, m, ly, st, 1, m)).real();
            }

            // -U' interorbital opposite-spin: -U' sum_{m≠m'} <n_{m↑}><n_{m'↓}>
            for (int m = 0; m < nm; m++)
                for (int mp = 0; mp < nm; mp++)
                    if (mp != m)
                        dc -= Up * (rho(ly, st, 0, m, ly, st, 1, mp) * rho(ly, st, 1, mp, ly, st, 0, m) - 
                                    rho(ly, st, 0, m, ly, st, 0, m) * rho(ly, st, 1, mp, ly, st, 1, mp)).real();
            
            std::cout << "DC after -U and -U': " << dc << "\n";

            // -(U'-J) same-spin: -(U'-J) sum_{m<m', σ} <n_{mσ}><n_{m'σ}>
            for (int m = 0; m < nm; m++)
                for (int mp = m+1; mp < nm; mp++) {
                    dc -= (Up - J) * (rho(ly, st, 0, m, ly, st, 0, mp)*rho(ly, st, 0, mp, ly, st, 0, m)
                                    - rho(ly, st, 0, m, ly, st, 0, m)*rho(ly, st, 0, mp, ly, st, 0, mp)).real();  // σ=↑
                    dc -= (Up - J) * (rho(ly, st, 1, m, ly, st, 1, mp)*rho(ly, st, 1, mp, ly, st, 1, m)
                                    - rho(ly, st, 1, m, ly, st, 1, m)*rho(ly, st, 1,mp, ly, st, 1, mp)).real(); // σ=↓
                }

            // J exchange + pair-hopping DC: sum over all ordered pairs m≠m'
            for (int m = 0; m < nm; m++)
                for (int mp = 0; mp < nm; mp++) {
                    if (mp == m) continue;
                    // Exchange: <d†_{m↑}d_{m'↑}><d†_{m'↓}d_{m↓}> - <d†_{m↑}d_{m↓}><d†_{m'↓}d_{m'↑}>
                    dc += J * (rho(ly, st, 0, m, ly, st, 0, mp) * rho(ly, st, 1, mp, ly, st, 1, m)
                            - rho(ly, st, 0, m, ly, st, 1, m) * rho(ly, st, 1, mp, ly, st, 0, mp)).real();
                    // Pair-hopping: <d†_{m↑}d_{m'↑}><d†_{m↓}d_{m'↓}> - <d†_{m↑}d_{m'↓}><d†_{m↓}d_{m'↑}>
                    dc += J * (rho(ly, st, 0, m, ly, st, 0, mp) * rho(ly, st, 1, m, ly, st, 1, mp)
                            - rho(ly, st, 0, m, ly, st, 1, mp) * rho(ly, st, 1, m, ly, st, 0, mp)).real();
                }
            
            // Spin/Orbit End Here
        }
    }

    return dc;
}