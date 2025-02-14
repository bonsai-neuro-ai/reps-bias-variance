import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hyp2f1


class PowerLawDataGenerator:
    def __init__(self, n_, alpha_, eig_min_):
        self.n_x = n_
        self.alpha_ = alpha_
        self.eig_min_ = eig_min_

        # Generate variances
        u = np.random.uniform(0, 1, n_)
        self.variances = eig_min_ * (1 - u) ** (-1 / alpha_)

        # Generate random rotation matrix Q
        self.rot, _ = np.linalg.qr(np.random.randn(n_, n_))

        # Generate response variable weights
        self.w = np.random.randn(n_) / np.sqrt(n_)

    def generate_data(self, m_, sanity: bool = False):
        # Generate data such that the spectrum of cov(x) is a power-law
        X = np.random.randn(m_, self.n_x)
        X = X * np.sqrt(self.variances)
        Xrot = X @ self.rot

        if sanity:
            empirical_var_x = np.var(X, axis=0)
            empirical_eigs = np.linalg.eigvalsh(X.T @ X / (m_ - 1))

            bins = np.logspace(np.log10(self.eig_min_), 2, 51)
            bin_centers = np.sqrt(bins[:-1] * bins[1:])
            empirical_var_pmf = np.histogram(empirical_var_x, bins=bins)[0]
            empirical_spectrum_pmf = np.histogram(empirical_eigs, bins=bins)[0]
            log_scale_pmf = pareto_pdf(bin_centers, self.alpha_, self.eig_min_) * bin_centers
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
            plt.vlines(self.eig_min_, *plt.ylim(), color="black", linestyle="--")
            plt.xscale("log")
            plt.yscale("log")

        # Generate the response variable
        y = Xrot @ self.w + np.random.randn(m_)
        return Xrot, y


def ridge_regression(X, y, lambda_):
    m, n = X.shape
    return np.linalg.solve(X.T @ X + m * lambda_ * np.eye(n), X.T @ y)


def pareto_pdf(x, alpha_, x_min_):
    return alpha_ * x_min_**alpha_ / x ** (alpha_ + 1)


def m(alpha_, eig_min_, z):
    """Stieltjes transform of Pareto distribution"""
    return alpha_ / (eig_min_ * (alpha_ + 1)) * hyp2f1(1, alpha_ + 1, alpha_ + 2, z / eig_min_)


def mprime(alpha_, eig_min_, z):
    """Derivative of m(..., z) with respect to z"""
    return alpha_ / (eig_min_**2 * (alpha_ + 2)) * hyp2f1(2, alpha_ + 2, alpha_ + 3, z / eig_min_)


def v(gamma_, alpha_, eig_min_, z):
    """Stieltjes transform of the asymptotic covariance matrix"""
    return gamma_ * (m(alpha_, eig_min_, z) + 1 / z) - 1 / z


def vprime(gamma_, alpha_, eig_min_, z):
    """Derivative of v(..., z) with respect to z"""
    return gamma_ * (mprime(alpha_, eig_min_, z) - 1 / z**2) + 1 / z**2


def asymptotic_mse(gamma_, alpha_, lambda_, eig_min_):
    v_ = v(gamma_, alpha_, eig_min_, -lambda_)
    vp_ = vprime(gamma_, alpha_, eig_min_, -lambda_)
    return (1 + (lambda_ / gamma_ - 1) * (1 - lambda_ * vp_ / v_)) / lambda_ / v_


def run_experiment_get_mse(m_, gamma_, alpha_, lambda_, eig_min_, run_=0):
    np.random.seed(hash((m_, gamma_, alpha_, lambda_, eig_min_, run_)) % 2**32)
    n = int(m_ * gamma_)
    generator = PowerLawDataGenerator(n, alpha_, eig_min_)
    X_train, y_train = generator.generate_data(m_)
    w_hat = ridge_regression(X_train, y_train, lambda_)
    pred_y = X_train @ w_hat
    mse_train = np.mean((pred_y - y_train) ** 2)

    w_error = np.sum((w_hat - generator.w) ** 2)

    X_test, y_test = generator.generate_data(m_)
    pred_y = X_test @ w_hat
    mse_test = np.mean((pred_y - y_test) ** 2)

    return {
        "m": m_,
        "n": n,
        "gamma": gamma_,
        "alpha": alpha_,
        "signal": np.linalg.norm(generator.w) ** 2,
        "lambda": lambda_,
        "eig_min": eig_min_,
        "run": run_,
        "train mse": mse_train,
        "test mse": mse_test,
        "w_error": w_error,
    }


if __name__ == "__main__":
    from joblib import Parallel, delayed, Memory
    import pandas as pd
    import seaborn as sns

    power_law_alpha = 1.5
    gamma = 0.1  # asymptotic n/m ratio
    runs = 5
    eig_min = 1e-1

    # for m in [100, 500, 1000, 5000, 10000, 50000]:
    #     generator = PowerLawDataGenerator(int(gamma*m), power_law_alpha, eig_min)
    #     generator.generate_data(m, sanity=True)
    # plt.show()
    # exit()

    ms = np.logspace(1, 4, 10).astype(int)
    lambda_opt = gamma
    lambdas = [.001, .01, .1, 1.0]

    # Sanity-check that the calculations are not exploding or hitting any errors
    assert np.isfinite(
        run_experiment_get_mse(
            ms[0], gamma, power_law_alpha, lambdas[0], eig_min
        )["train mse"]
    )
    assert np.isfinite(
        asymptotic_mse(gamma, power_law_alpha, lambdas[0], eig_min)
    )

    memory = Memory(location="results_power_law_ridge_regression", verbose=0)
    run_experiment_get_mse = memory.cache(run_experiment_get_mse)

    pool = Parallel(n_jobs=8, verbose=1)
    results = pool(
        delayed(run_experiment_get_mse)(
            m, gamma, power_law_alpha, lam, eig_min, run
        )
        for m in ms
        for lam in lambdas
        for run in range(runs)
    )
    results = pd.DataFrame(results)

    # plt.figure()
    # sns.lineplot(data=results, x="m", y="signal")
    # plt.xscale("log")
    # plt.ylim(0, 2)
    # plt.show()

    plt.figure()
    # handles = sns.lineplot(data=results, x="m", y="train mse", hue="lambda")
    handles = sns.lineplot(data=results, x="m", y="test mse", hue="lambda")

    # Per lambda, compute the asymptotic MSE and plot with hlines
    xl = plt.xlim()
    ana_mses = []
    for lam in lambdas:
        ana_mses.append(
            asymptotic_mse(gamma, power_law_alpha, lam, eig_min)
        )
    # Plot using colors from the handles
    plt.hlines(
        ana_mses, xl[0], xl[1], linestyle="--", colors=[h.get_color() for h in handles.lines]
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()
