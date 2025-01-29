import argparse
from functools import partial

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from comparators import procrustes, cka, regression_mse
from data_gen import PowerLawFourierSynthesisNeurons, RadialBasisNeurons, DataGenBase
from utils import unique_keys, dict_difference, update_dict

comparators = {
    "procrustes": procrustes,
    "linear_cka": partial(cka, debias="none", kernel="linear"),
    "debiased_linear_cka": partial(cka, debias="song", kernel="linear"),
    "brownian_cka": partial(cka, debias="none", kernel="brownian"),
    "debiased_brownian_cka": partial(cka, debias="song", kernel="brownian"),
    "regression": regression_mse,
}


def run_job(cls, run_id, comparator, m, **constructor_kwargs):
    x_fn = cls(**constructor_kwargs)
    y_fn = cls(**constructor_kwargs)
    z = np.random.uniform(-1, 1, size=(m, constructor_kwargs["latent_dim"]))
    x = x_fn(z)
    y = y_fn(z)

    return dict(m=m, value=comparator(x, y), run=run_id, **constructor_kwargs)


def plot_tuning(gen: DataGenBase, ax=None, num_neurons: int = 1, label=None):
    ax = ax or plt.gca()

    is_noisy = gen.poisson_scale is not None and gen.poisson_scale > 0.0

    z = np.zeros((1000, gen.latent_dim))
    z[:, 0] = np.linspace(-1, 1, 1000)
    activity = gen(z)
    ax.plot(z, activity[:, :num_neurons], "." if is_noisy else "-", label=label)

    ax.set_xlabel("z")
    ax.set_ylabel("response")
    ax.set_title("Example neurons' tuning")
    ax.legend()

    return z, activity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Shared arguments
    parser.add_argument("--comparator", choices=comparators.keys(), required=True)
    parser.add_argument("--m", type=int, default=[1000], nargs="+", help="Number of trials")
    parser.add_argument("--n", type=int, default=[1000], nargs="+", help="Number of neurons")
    parser.add_argument("--d", type=int, default=1, help="Latent dimension of z")
    parser.add_argument(
        "--poisson-scale",
        type=float,
        default=0.0,
        help="Flag to add Poisson noise. If 0.0, no noise added. "
        "Otherwise, neural activity is multiplied by poisson_scale and sampled from a poisson.",
    )
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

    kwargs_base = {
        "latent_dim": args.d,
        "num_neurons": args.n[0],
        "poisson_scale": args.poisson_scale,
    }

    if args.mode == "power_law":
        memory = joblib.Memory(location="results_power_law", verbose=0)
        data_gen_class = PowerLawFourierSynthesisNeurons
        kwargs = [{"alpha": a, **kwargs_base} for a in args.alpha]
    elif args.mode == "radial_basis":
        memory = joblib.Memory(location="results_radial_basis", verbose=0)
        data_gen_class = RadialBasisNeurons
        kwargs = [{"bandwidth": bw, **kwargs_base} for bw in args.bandwidth]
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    del kwargs_base

    if args.plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        for kw in kwargs:
            data_gen = data_gen_class(**kw)
            z, x = plot_tuning(data_gen, ax=axs[0], label=str(unique_keys(kw, kwargs)))
            cov = np.cov(x.T)
            descending_eigs = np.linalg.eigvalsh(cov)[::-1]
            axs[1].plot(descending_eigs, label=str(unique_keys(kw, kwargs)))

        axs[1].set_xlabel("eigenvalue index")
        axs[1].set_ylabel("eigenvalue")
        axs[1].set_yscale("log")
        axs[1].set_xscale("log")
        axs[1].set_title("Population covariance spectrum")
        axs[1].legend()

        plt.savefig(f"plots/example_neurons_{args.mode}_{args.d}d_noise{args.poisson_scale}.png")
        plt.show()

    # Sanity-check a mini job before running in parallel
    run_job(
        data_gen_class,
        run_id=0,
        comparator=comp_fn,
        m=args.m[0],
        **update_dict(kwargs[0], num_neurons=args.n[0]),
    )

    # Create a joblib file cache to load precomputed results. Ignore the cls arg when caching;
    # the different modes will be cached in different directories
    run_job = memory.cache(run_job, ignore=["cls"])

    # Dispatch compute jobs with 1 job per parameter setting
    pool = joblib.Parallel(n_jobs=args.pool, verbose=10)
    results = pool(
        joblib.delayed(run_job)(
            data_gen_class,
            run_id=r,
            comparator=comp_fn,
            m=m,
            **update_dict(kw, num_neurons=n),
        )
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
        hue_by = ", ".join(dict_difference(*kwargs)[0].keys())

        plt.figure()
        sns.lineplot(data=results_df, x=x_axis, y=comp_name, hue=hue_by)
        plt.title(f"{comp_name} vs {x_axis}, latent dim={args.d}")
        plt.xscale("log")
        plt.savefig(
            f"plots/{args.mode}_{args.comparator}_vs_{x_axis}_{args.d}d_noise{args.poisson_scale}.png"
        )
        plt.show()
