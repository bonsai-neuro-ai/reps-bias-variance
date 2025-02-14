import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hyp2f1
from utils import try_plot


class PowerLawDataGenerator:
    def __init__(self, n, alpha, eig_min):
        self.n_x = n
        self.alpha = alpha
        self.eig_min = eig_min

        # Generate variances
        u = np.random.uniform(0, 1, n)
        self.variances = eig_min * (1 - u) ** (-1 / alpha)

        # Generate random rotation matrix Q
        self.rot, _ = np.linalg.qr(np.random.randn(n, n))

        # Generate response variable weights
        self.w = np.random.randn(n) / np.sqrt(n)

    def generate_data(self, m, sanity: bool = False):
        # Generate data such that the spectrum of cov(x) is a power-law
        X = np.random.randn(m, self.n_x)
        X = X * np.sqrt(self.variances)
        Xrot = X @ self.rot

        if sanity:
            empirical_var_x = np.var(X, axis=0)
            empirical_eigs = np.linalg.eigvalsh(X.T @ X / (m - 1))

            bins = np.logspace(np.log10(self.eig_min), 2, 51)
            bin_centers = np.sqrt(bins[:-1] * bins[1:])
            empirical_var_pmf = np.histogram(empirical_var_x, bins=bins)[0]
            empirical_spectrum_pmf = np.histogram(empirical_eigs, bins=bins)[0]
            log_scale_pmf = pareto_pdf(bin_centers, self.alpha, self.eig_min) * bin_centers
            h = plt.plot(
                bin_centers,
                empirical_var_pmf / empirical_var_pmf.sum(),
                linestyle="none",
                marker="+",
            )
            plt.plot(
                bin_centers,
                empirical_spectrum_pmf / empirical_spectrum_pmf.sum(),
                linestyle="none",
                marker="x",
                color=h[0].get_color(),
            )
            plt.plot(
                bin_centers, log_scale_pmf / log_scale_pmf.sum(), marker=".", color=h[0].get_color()
            )
            plt.vlines(self.eig_min, *plt.ylim(), color="black", linestyle="--")
            plt.xscale("log")
            plt.yscale("log")

        # Generate the response variable
        y = Xrot @ self.w + np.random.randn(m)
        return Xrot, y


def ridge_regression(X, y, lam):
    m, n = X.shape
    return np.linalg.solve(X.T @ X + m * lam * np.eye(n), X.T @ y)


def pareto_pdf(x, alpha, x_min):
    return alpha * x_min**alpha / x ** (alpha + 1)


def m(alpha, eig_min, z):
    """Stieltjes transform of Pareto distribution"""
    return alpha / (eig_min * (alpha + 1)) * hyp2f1(1, alpha + 1, alpha + 2, z / eig_min)


def mprime(alpha, eig_min, z):
    """Derivative of m(..., z) with respect to z"""
    return alpha / (eig_min**2 * (alpha + 2)) * hyp2f1(2, alpha + 2, alpha + 3, z / eig_min)


def v(gamma, alpha, eig_min, z):
    """Stieltjes transform of the asymptotic covariance matrix"""
    return gamma * (m(alpha, eig_min, z) + 1 / z) - 1 / z


def vprime(gamma, alpha, eig_min, z):
    """Derivative of v(..., z) with respect to z"""
    return gamma * (mprime(alpha, eig_min, z) - 1 / z**2) + 1 / z**2


def asymptotic_mse(gamma, alpha, lam, eig_min):
    v_ = v(gamma, alpha, eig_min, -lam)
    vp_ = vprime(gamma, alpha, eig_min, -lam)
    return (1 + (lam / gamma - 1) * (1 - lam * vp_ / v_)) / lam / v_


def run_experiment_get_mse(m, gamma, alpha, lam, eig_min, run=0):
    np.random.seed(hash((m, gamma, alpha, lam, eig_min, run)) % 2**32)
    n = int(m * gamma)
    generator = PowerLawDataGenerator(n, alpha, eig_min)
    X_train, y_train = generator.generate_data(m)
    w_hat = ridge_regression(X_train, y_train, lam)
    pred_y = X_train @ w_hat
    mse_train = np.mean((pred_y - y_train) ** 2)

    w_error = np.sum((w_hat - generator.w) ** 2)

    X_test, y_test = generator.generate_data(m)
    pred_y = X_test @ w_hat
    mse_test = np.mean((pred_y - y_test) ** 2)

    return {
        "m": m,
        "n": n,
        "γ": gamma,
        "α": alpha,
        "ω": np.linalg.norm(generator.w) ** 2,
        "λ": lam,
        "eig_min": eig_min,
        "run": run,
        "train mse": mse_train,
        "test mse": mse_test,
        "w_error": w_error,
    }


if __name__ == "__main__":
    import argparse
    from joblib import Parallel, delayed, Memory
    import pandas as pd
    import seaborn as sns

    sns.set_theme(font_scale=0.5)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alphas",
        type=float,
        default=[1.5],
        nargs="+",
        help="Values of alpha (power law exponent) to run",
    )
    parser.add_argument("--ms", type=int, default=[10000], nargs="+", help="Values of m to run")
    parser.add_argument(
        "--lambdas", type=float, default=[0.1], nargs="+", help="Values of λ to run"
    )
    parser.add_argument("--gamma", type=float, default=0.1, help="Asymptotic n/m ratio")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per experiment")
    parser.add_argument("--eig_min", type=float, default=1e-1, help="Minimum eigenvalue of cov(x)")
    parser.add_argument("--pool", type=int, default=8, help="Number of parallel jobs")
    args = parser.parse_args()

    # Sanity-check that the calculations are not exploding or hitting any errors before launching
    # the joblib pool
    assert np.isfinite(
        run_experiment_get_mse(
            args.ms[0], args.gamma, args.alphas[0], args.lambdas[0], args.eig_min
        )["train mse"]
    )
    assert np.isfinite(asymptotic_mse(args.gamma, args.alphas[0], args.lambdas[0], args.eig_min))

    memory = Memory(location="results_power_law_ridge_regression", verbose=0)
    run_experiment_get_mse = memory.cache(run_experiment_get_mse)

    pool = Parallel(n_jobs=8, verbose=1)
    results = pool(
        delayed(run_experiment_get_mse)(m, args.gamma, a, lam, args.eig_min, run)
        for m in args.ms
        for a in args.alphas
        for lam in args.lambdas
        for run in range(args.runs)
    )
    results = pd.DataFrame(results)

    asymptotics = pd.DataFrame(
        [
            {
                "γ": args.gamma,
                "α": a,
                "λ": lam,
                "eig_min": args.eig_min,
                "mse": asymptotic_mse(args.gamma, a, lam, args.eig_min),
            }
            for a in args.alphas
            for lam in args.lambdas
        ]
    )

    # Plot theory
    plt.figure(figsize=(3, 2))
    sns.lineplot(data=asymptotics, x="α", y="mse", hue="λ")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("plots/powerlaw_ridge_regression_theory.png")
    try_plot()

    # Plot empirical MSE vs M for each lambda, broken down by alpha
    for group, df in results.groupby(["α"]):
        plt.figure(figsize=(3, 2))
        h = sns.lineplot(data=df, x="m", y="test mse", hue="λ")
        asymp = asymptotics.query(f"α == {group[0]}").sort_values("λ")["mse"]
        plt.hlines(
            asymp,
            *plt.xlim(),
            colors=[l.get_color() for l in h.lines],
            linestyle="--",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.title(f"α={group[0]}")
        plt.tight_layout()
        plt.savefig(f"plots/powerlaw_ridge_regression_alpha{group[0]:.2f}.png")
        try_plot()
