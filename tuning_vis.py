import torch
import matplotlib.pyplot as plt
from data_gen import RadialBasisNeurons


@torch.no_grad()
def main(n_neurons, d, bandwidth, ax3d: bool = False, seed=178948):
    z = torch.zeros(1000, d)
    z[:, 0] = torch.linspace(-1, 1, 1000)

    pop = RadialBasisNeurons(num_neurons=n_neurons, bandwidth=bandwidth, latent_dim=d, seed=seed)

    responses = pop(z)
    cov = responses.T @ (responses - responses.mean(0, keepdim=True)) / responses.shape[0]
    _, u = torch.linalg.eigh(cov + 1e-6 * torch.eye(cov.shape[0]))

    pc1 = responses @ u[:, -1]
    pc2 = responses @ u[:, -2]
    pc3 = responses @ u[:, -3]

    fig = plt.figure(figsize=(3, 3))
    if ax3d:
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(pc1, pc2, pc3)
    else:
        ax = fig.add_subplot(111)
        ax.plot(pc1, pc2)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if ax3d:
        ax.set_zlabel("PC3")

    # Equal axis sizes
    ax.set_aspect("equal", adjustable="datalim")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neurons", type=int, default=1000)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--bandwidth", type=float, default=0.1)
    parser.add_argument("--ax3d", action="store_true")

    args = parser.parse_args()

    main(args.n_neurons, args.d, args.bandwidth, args.ax3d)
    plt.title(f"N={args.n_neurons}, d={args.d}, bandwidth={args.bandwidth}")
    plt.show()