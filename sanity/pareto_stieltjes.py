import numpy as np
from power_law_ridge_regression import pareto_pdf, m, mprime, PowerLawDataGenerator
from scipy.integrate import quad
from functools import partial
import matplotlib.pyplot as plt


def stieltjes(p, z, x_min):
    def integrand(l):
        return p(l) / (l - z)

    return quad(integrand, x_min, np.inf)[0]


def sanity_check_stieltjes():
    for x_min in [1.0, 0.1]:
        for alpha in [1.0, 2.0, 3.0]:
            p = partial(pareto_pdf, alpha_=alpha, x_min_=x_min)
            zs = np.linspace(x_min - 4, x_min, 100)[:-1]

            numerical_s = np.array([stieltjes(p, z, x_min) for z in zs])
            analytic_s = m(alpha, x_min, zs)
            x, *_ = PowerLawDataGenerator(1000, alpha, x_min).generate_data(10000)
            cov_x = np.cov(x.T)
            trace_inv = [np.trace(np.linalg.inv(cov_x - z * np.eye(1000))) / 1000 for z in zs]

            h = plt.plot(zs, analytic_s, "-", label=f"Analytic (alpha={alpha}, xm={x_min})")
            plt.plot(
                zs,
                numerical_s,
                "+",
                label=f"Numerical (alpha={alpha}, xm={x_min})",
                color=h[0].get_color(),
            )
            plt.plot(
                zs,
                trace_inv,
                ".",
                label=f"Tr((Σ - zI)^-1)/n (alpha={alpha}, xm={x_min})",
                color=h[0].get_color(),
            )
            plt.legend()
        plt.xlabel("z")
        plt.ylabel("S(z)")
        plt.title(r"Stieltjes transform S(z) of Pareto distribution (parameter $\alpha$)")
    plt.yscale("log")
    plt.show()


def sanity_check_derivative():
    dz = 1e-6
    for x_min in [1.0, 0.1]:
        for alpha in [1.0, 2.0, 3.0]:
            zs = np.linspace(x_min - 4, x_min, 100)[:-1]
            analytic_sprime = mprime(alpha, x_min, zs)
            numerical_sprime = (m(alpha, x_min, zs + dz) - m(alpha, x_min, zs - dz)) / (2 * dz)

            h = plt.plot(zs, analytic_sprime, "-", label=f"Analytic (alpha={alpha}, xm={x_min})")
            plt.plot(
                zs,
                numerical_sprime,
                "+",
                label=f"Numerical (alpha={alpha}, xm={x_min})",
                color=h[0].get_color(),
            )
            plt.legend()
        plt.xlabel("z")
        plt.ylabel("dS/dz")
        plt.title(r"Check the gradient")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    sanity_check_stieltjes()
    sanity_check_derivative()
