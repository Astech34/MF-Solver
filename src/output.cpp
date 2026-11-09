#include "MF-Solver/output.hpp"

#include <iomanip>
#include <string>

namespace {

std::string spin_label(int spin, int num_spin) {
    if (num_spin == 2) return spin == 0 ? "up" : "dn";
    return "s" + std::to_string(spin);
}

} // namespace

void print_occupations(const Occupations& occ, std::ostream& os) {
    const Dims& d = occ.dims;
    const auto flags = os.flags();
    const auto prec  = os.precision();
    os << std::fixed << std::setprecision(4);

    // Header: orbital indices
    os << "Occupations\n" << std::setw(10) << "";
    for (int o = 0; o < d.num_orbitals; ++o) os << std::setw(9) << ("m" + std::to_string(o));
    os << std::setw(10) << "sum" << "\n";

    double n_total = 0.0;
    for (int l = 0; l < d.num_layers; ++l) {
        for (int s = 0; s < d.num_sites; ++s) {
            os << "  Layer " << l << ", Site " << s << "\n";

            double n_site = 0.0;
            for (int sp = 0; sp < d.num_spin; ++sp) {
                double n_spin = 0.0;
                os << "    " << std::setw(4) << std::left << spin_label(sp, d.num_spin) << std::right << "  ";
                for (int o = 0; o < d.num_orbitals; ++o) {
                    os << std::setw(9) << occ(l, s, sp, o);
                    n_spin += occ(l, s, sp, o);
                }
                os << std::setw(10) << n_spin << "\n";
                n_site += n_spin;
            }
            os << "    " << std::setw(6 + 9 * d.num_orbitals) << std::left << "total"
               << std::right << std::setw(10) << n_site << "\n";
            n_total += n_site;
        }
    }
    os << "  N_total = " << n_total << "\n";

    os.flags(flags);
    os.precision(prec);
}

void print_moments(const Moments& m, const std::string& title, std::ostream& os) {
    const auto flags = os.flags();
    const auto prec  = os.precision();
    os << std::fixed << std::setprecision(4);

    auto print_row = [&](const std::string& label, const Eigen::Vector3d& v) {
        os << "  " << std::setw(12) << std::left << label << std::right;
        for (int i = 0; i < 3; ++i) os << std::setw(10) << v(i);
        os << std::setw(10) << v.norm() << "\n";
    };

    os << title << "\n" << std::setw(14) << ""
       << std::setw(10) << "x" << std::setw(10) << "y" << std::setw(10) << "z"
       << std::setw(10) << "|m|" << "\n";

    Eigen::Vector3d total = Eigen::Vector3d::Zero();
    for (int l = 0; l < m.nlayer; ++l)
        for (int s = 0; s < m.nsite; ++s) {
            print_row("L" + std::to_string(l) + " S" + std::to_string(s), m(l, s));
            total += m(l, s);
        }
    print_row("total", total);

    os.flags(flags);
    os.precision(prec);
}
