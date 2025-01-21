import argparse
from functools import partial

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from comparators import procrustes, cka, regression_mse
from data_gen import PowerLawFourierSynthesisNeurons, RadialBasisNeurons

comparators = {
    "procrustes": procrustes,
    "linear_cka": partial(cka, debias="none", kernel="linear"),
    "debiased_linear_cka": partial(cka, debias="song", kernel="linear"),
    "brownian_cka": partial(cka, debias="none", kernel="brownian"),
    "debiased_brownian_cka": partial(cka, debias="song", kernel="brownian"),
    "regression": regression_mse,
}


def run_job(cls, run_id, d, m, n, comparator, **constructor_kwargs):
    x_fn = cls(latent_dim=d, num_neurons=n, **constructor_kwargs)
    y_fn = cls(latent_dim=d, num_neurons=n, **constructor_kwargs)
    z = np.random.uniform(-1, 1, size=(m, d))
    x = x_fn(z)
    y = y_fn(z)

    return dict(d=d, m=m, n=n, value=comparator(x, y), run=run_id, **constructor_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Shared arguments
    parser.add_argument("--comparator", choices=comparators.keys(), required=True)
    parser.add_argument("--m", type=int, default=[1000], nargs="+", help="Number of trials")
    parser.add_argument("--n", type=int, default=[1000], nargs="+", help="Number of neurons")
    parser.add_argument("--d", type=int, default=1, help="Latent dimension of z")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--pool", type=int, default=-1)
    parser.add_argument("--plot", action="store_true")

    # Power Law subparser
    power_law_parser = subparsers.add_parser("power_law")
    power_law_parser.add_argument("--alpha", type=float, default=[3.0], nargs="+")

    # Radial Basis subparser
    radial_basis_parser = subparsers.add_parser("radial_basis")
    radial_basis_parser.add_argument("--bandwidth", type=float, default=[0.5], nargs="+")

    args = parser.parse_args()

    assert (
        min(len(args.n), len(args.m)) == 1
    ), "Must supply either --m or --n as a single scalar and vary the other"

    comp_fn = comparators[args.comparator]
    comp_name = args.comparator.replace("_", " ")

    if args.mode == "power_law":
        memory = joblib.Memory(location="results_power_law", verbose=0)
        data_gen_class = PowerLawFourierSynthesisNeurons
        kwargs = [{"alpha": a} for a in args.alpha]
    elif args.mode == "radial_basis":
        memory = joblib.Memory(location="results_radial_basis", verbose=0)
        data_gen_class = RadialBasisNeurons
        kwargs = [{"bandwidth": bw} for bw in args.bandwidth]
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    # Ignore the cls arg when caching; the different modes will be cached in different directories
    run_job = memory.cache(run_job, ignore=["cls"])

    if args.plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for kw in kwargs:
            gen = data_gen_class(latent_dim=args.d, num_neurons=500, **kw)
            z = np.zeros((1000, args.d))
            z[:, 0] = np.linspace(-1, 1, 1000)
            x = gen(z)
            ax[0].plot(z, x[:, 0], label=str(kw))

            cov = np.cov(x.T)
            descending_eigs = np.linalg.eigvalsh(cov)[::-1]
            ax[1].plot(descending_eigs, label=str(kw))
        ax[0].set_xlabel("z")
        ax[0].set_ylabel("response")
        ax[0].set_title("Example neurons' tuning")
        ax[0].legend()

        ax[1].set_xlabel("eigenvalue index")
        ax[1].set_ylabel("eigenvalue")
        ax[1].set_yscale("log")
        ax[1].set_xscale("log")
        ax[1].set_title("Population covariance spectrum")
        ax[1].legend()

        plt.show()

    # Sanity-check before running in parallel
    run_job(cls=data_gen_class, run_id=0, d=args.d, m=args.m[0], n=args.n[0], comparator=comp_fn, **kwargs[0])

    # Dispatch compute jobs with 1 job per parameter setting
    pool = joblib.Parallel(n_jobs=args.pool, verbose=10)
    results = pool(
        joblib.delayed(run_job)(cls=data_gen_class, run_id=r, d=args.d, m=m, n=n, comparator=comp_fn, **kw)
        for r in range(args.repeats)
        for m in args.m
        for n in args.n
        for kw in kwargs
    )

    results_df = pd.DataFrame(results)

    # Rename the 'values' column to the comparator name
    results_df[comp_name] = results_df.pop("value")

    x_axis = "m" if len(args.m) > 1 else "n"

    if args.plot:
        hue_by = list(kwargs[0].keys())[0]

        plt.figure()
        sns.lineplot(data=results_df, x=x_axis, y=comp_name, hue=hue_by)
        plt.title(f"{comp_name} vs {x_axis}, latent dim={args.d}")
        plt.xscale("log")
        plt.show()
