// Regularized incomplete gamma functions P(a, x) and Q(a, x).
//
// Three-path evaluation strategy (numerical best practice):
//   • a > IGAM_LARGE_A     → Wilson–Hilferty normal approximation  (error O(1/a²))
//   • x <  a + 1           → series expansion for P(a, x)
//   • x >= a + 1           → modified Lentz continued fraction for Q(a, x)
//
// Tolerance is near machine epsilon for double precision.

#include "math_utils.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

constexpr double IGAM_EPS      = 3.0e-15;   // convergence tolerance
constexpr double IGAM_FPMIN    = 1.0e-300;  // guard against divide-by-zero in Lentz
constexpr int    IGAM_MAXIT    = 1000;
constexpr double IGAM_LARGE_A  = 500.0;     // switch to WH above this

// Wilson–Hilferty cube-root approximation for large a.
// Q(a, x) ≈ erfc(z/√2) / 2, where z is the transformed argument.
double igamc_wilson_hilferty(double a, double x)
{
    const double cbrt_ratio = std::cbrt(x / a);
    const double rcp9a      = 1.0 / (9.0 * a);
    const double z          = (cbrt_ratio - (1.0 - rcp9a)) / std::sqrt(rcp9a);
    return 0.5 * std::erfc(z * M_SQRT1_2);
}

// Series expansion for P(a, x). Converges quickly for x < a + 1.
double igam_series(double a, double x)
{
    if (x < 0.0)
        throw std::domain_error("igam_series: x must be >= 0");

    double ap  = a;
    double sum = 1.0 / a;
    double del = sum;

    for (int n = 0; n < IGAM_MAXIT; ++n) {
        ++ap;
        del *= x / ap;
        sum += del;
        if (std::fabs(del) < std::fabs(sum) * IGAM_EPS)
            break;
    }
    return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

// Modified Lentz continued fraction for Q(a, x). Converges quickly for x >= a + 1.
double igamc_continued_fraction(double a, double x)
{
    double b = x + 1.0 - a;
    double c = 1.0 / IGAM_FPMIN;
    double d = 1.0 / b;
    double h = d;

    for (int i = 1; i <= IGAM_MAXIT; ++i) {
        const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;

        d = an * d + b;
        if (std::fabs(d) < IGAM_FPMIN) d = IGAM_FPMIN;
        c = b + an / c;
        if (std::fabs(c) < IGAM_FPMIN) c = IGAM_FPMIN;
        d = 1.0 / d;

        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < IGAM_EPS) break;
    }
    return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

}  // namespace

// ─── Public API ─────────────────────────────────────────────────

double igam(double a, double x)
{
    if (a <= 0.0 || x < 0.0)
        throw std::domain_error("igam: requires a > 0 and x >= 0");
    if (x == 0.0)
        return 0.0;
    if (a > IGAM_LARGE_A)
        return 1.0 - igamc_wilson_hilferty(a, x);
    if (x < a + 1.0)
        return igam_series(a, x);
    return 1.0 - igamc_continued_fraction(a, x);
}

double igamc(double a, double x)
{
    if (a <= 0.0 || x < 0.0)
        throw std::domain_error("igamc: requires a > 0 and x >= 0");
    if (x == 0.0)
        return 1.0;
    if (a > IGAM_LARGE_A)
        return igamc_wilson_hilferty(a, x);
    if (x < a + 1.0)
        return 1.0 - igam_series(a, x);
    return igamc_continued_fraction(a, x);
}
