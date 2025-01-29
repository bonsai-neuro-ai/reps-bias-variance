import torch


def prep_reps(rep1: torch.Tensor, rep2: torch.Tensor, center=True, scale=True):
    if rep1.ndim != 2 or rep2.ndim != 2 or rep2.size(0) != rep1.size(0):
        raise ValueError("rep1 and rep2 must have shapes (m, n1) and (m, n2)")

    rep1 = rep1.clone()
    rep2 = rep2.clone()

    if center:
        rep1 = rep1 - torch.mean(rep1, dim=0, keepdim=True)
        rep2 = rep2 - torch.mean(rep2, dim=0, keepdim=True)

    if scale:
        # Make the full data matrix unit Frobenius norm
        rep1 = rep1 / torch.sqrt(torch.sum(rep1**2))
        rep2 = rep2 / torch.sqrt(torch.sum(rep2**2))

    return rep1, rep2


def double_center(k):
    m = k.size(0)
    h = torch.eye(m, dtype=k.dtype, device=k.device) - torch.ones_like(k) / m
    return h @ k @ h


def hsic(x, y, debias="none", kernel="linear"):
    if kernel == "linear":
        x, y = prep_reps(x, y, center=True, scale=False)
        # Compute (Linear) HSIC
        kx = torch.einsum("if,jf->ij", x, x)
        ky = torch.einsum("if,jf->ij", y, y)
    elif kernel == "brownian":
        assert x.ndim == 2 and y.ndim == 2, "x and y must be 2D"
        norm_x = torch.linalg.norm(x, dim=-1)
        norm_y = torch.linalg.norm(y, dim=-1)
        dist_x = rdm(x, q=1.0)
        dist_y = rdm(y, q=1.0)
        kx = 1 / 2 * (norm_x[:, None] + norm_x[None, :] - dist_x)
        ky = 1 / 2 * (norm_y[:, None] + norm_y[None, :] - dist_y)
    else:
        raise ValueError("Unknown kernel")

    kx = double_center(kx)
    ky = double_center(ky)

    m = x.size(0)

    if debias == "none":
        return torch.sum(kx * ky) / (m * (m - 1))
    elif debias == "lange":
        kx, ky = torch.triu(kx, diagonal=1), torch.triu(ky, diagonal=1)
        return torch.sum(kx * ky) * 2 / (m * (m - 3))
    elif debias == "song":
        # Zero the diagonal
        kx[torch.arange(m), torch.arange(m)] = 0
        ky[torch.arange(m), torch.arange(m)] = 0
        return (
            torch.sum(kx * ky)
            + torch.sum(kx) * torch.sum(ky) / ((m - 1) * (m - 2) - 2 / (m - 2) * torch.sum(kx @ ky))
        ) / (m * (m - 3))
    else:
        raise ValueError("Unknown debias")


def cka(rep1: torch.Tensor, rep2: torch.Tensor, debias="none", kernel="linear"):
    hsic12 = hsic(rep1, rep2, debias=debias, kernel=kernel)
    hsic11 = hsic(rep1, rep1, debias=debias, kernel=kernel)
    hsic22 = hsic(rep2, rep2, debias=debias, kernel=kernel)
    return hsic12 / torch.sqrt(hsic11 * hsic22)


def procrustes(rep1: torch.Tensor, rep2: torch.Tensor):
    rep1, rep2 = prep_reps(rep1, rep2, center=True, scale=True)
    # Compute Procrustes similarity using nuclear norm trick. Note that norm on the covariance (
    # e.g. 1/m or 1/(m-1)) is irrelevant because canceled in the last line.
    cov_xy = torch.einsum("mi,mj->ij", rep1, rep2)
    trace_cov_x = torch.sum(rep1**2)
    trace_cov_y = torch.sum(rep2**2)
    return torch.linalg.matrix_norm(cov_xy, ord="nuc") / torch.sqrt(trace_cov_x * trace_cov_y)


def sqrtm(mat):
    e, v = torch.linalg.eigh(mat, )
    return v @ torch.diag(torch.sqrt(torch.clip(e, 0, None))) @ v.T


def fidelity(kx, ky):
    kx_half = sqrtm(kx)
    return torch.trace(sqrtm(kx_half @ ky @ kx_half))


def bures(rep1: torch.Tensor, rep2: torch.Tensor):
    rep1, rep2 = prep_reps(rep1, rep2, center=True, scale=True)
    # Compute (Linear) kernels
    kx = torch.einsum("if,jf->ij", rep1, rep1)
    ky = torch.einsum("if,jf->ij", rep2, rep2)
    # Compute Bures similarity
    norm_x = torch.trace(kx)
    norm_y = torch.trace(ky)
    return fidelity(kx, ky) / torch.sqrt(norm_x * norm_y)


def rdm(rep, q: float = 1.0):
    """Pairwise euclidean distance raised to the power q"""
    xxT = torch.einsum("if,jf->ij", rep, rep)
    d = torch.diag(xxT)
    sq_dist = torch.clip(d[None, :] + d[:, None] - 2 * xxT, 0.0, None)
    return sq_dist ** (q / 2.0)


def rsa_cosine(rep1: torch.Tensor, rep2: torch.Tensor, q: float = 1.0, center=False):
    rdm1 = rdm(rep1, q=q)
    rdm2 = rdm(rep2, q=q)
    if center:
        rdm1 = double_center(rdm1)
        rdm2 = double_center(rdm2)
    rdm1 = rdm1 / torch.sqrt(torch.sum(rdm1**2))
    rdm2 = rdm2 / torch.sqrt(torch.sum(rdm2**2))
    return torch.sum(rdm1 * rdm2)


def regression_mse(x, y, bias=True, procrustes=False):
    """Solve for w such that y ≈ x @ w, then return average error per m per n. If procrustes=True,
    w is constrained to be orthonormal.
    """
    if bias:
        x, y = prep_reps(x, y, center=True, scale=False)

    if procrustes:
        # Procrustes solution (orthonormal w) maximizes Tr(y.T @ x @ w). Since Tr(A.T @ A) is the
        # Frobenius inner-product, this is equivalent to finding w that maximally aligns with x.T@y,
        # i.e. the UV components of the SVD of x.T @ y.
        u, _, vh = torch.linalg.svd(x.T @ y)
        w = u @ vh
    else:
        w = torch.linalg.lstsq(x, y, rcond=-1)[0]

    y_pred = x @ w
    return torch.mean((y - y_pred) ** 2)
