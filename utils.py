import torch
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.utils import make_grid
import torch.nn as nn
import os
import math
from typing import Union, Optional
import numpy as np
from functools import partial
import ot as pot

def plot_distribution(x: Tensor, title: str) -> plt.Figure:
    x = x.detach().cpu()
    batch_size = int(x.shape[0])

    fig, axes = plt.subplots(2, max(batch_size // 2, 1), constrained_layout=True)
    axes = axes.flatten()

    for b in range(batch_size):
        hist, edges = torch.histogram(x[b], density=True)
        axes[b].plot(edges[:-1], hist)

    mean, std = x.mean(), x.std()
    fig.suptitle(f"{title} | Mean: {mean:.4f} Std: {std:.4f}")
    fig.supxlabel("X")
    fig.supylabel("Density")

    return fig


def log_images(
    logger: TensorBoardLogger, images: Tensor, title: str, global_step: int
) -> None:
    grid = make_grid(images.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True)
    logger.experiment.add_image(title, grid, global_step)


def log_distribution(
    logger: TensorBoardLogger, x: Tensor, title: str, global_step: int
) -> None:
    figure = plot_distribution(x, title)
    logger.experiment.add_figure(title, figure, global_step)


def log_samples(
    logger: TensorBoardLogger,
    samples: Tensor,
    tag: str,
    global_step: int,
) -> None:
    log_images(logger, samples, f"images/{tag}", global_step)
    log_distribution(logger, samples, f"distribution/{tag}", global_step)


def save_model_ckpt(model: nn.Module, ckpt_path: str) -> None:
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)


def load_model_ckpt(
    model: nn.Module, ckpt_path: str, device: torch.device
) -> nn.Module:
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")))
    return model.to(device)


def plot_pred_true(batch_pred, batch_ture, logger: TensorBoardLogger,global_step: int)-> None:
    fig, axes = plt.subplots() 
    x_pred = batch_pred[:, 0].cpu().numpy()
    y_pred = batch_pred[:, 1].cpu().numpy()

    x_true = batch_ture[:, 0].cpu().numpy()
    y_true = batch_ture[:, 1].cpu().numpy()

    # 绘制散点图
    axes.scatter(x_true, y_true, c='red', marker='x', label='True')
    axes.scatter(x_pred, y_pred, c='blue', marker='o', label='Predicted')

    # 设置图表标题和轴标签
    plt.title("Scatter Plot")
    plt.xlabel("X")
    plt.ylabel("Y")

    # 显示图例
    plt.legend()
    logger.experiment.add_figure('pred%true', fig, global_step)





min_var_est = 1e-8


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list):
    assert X.size(0) == Y.size(0)
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    if biased:
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = Kt_XX_sum / (m * (m - 1)) + Kt_YY_sum / (m * (m - 1)) - 2.0 * K_XY_sum / (m * m)

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased
    )
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)  # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX**2).sum() - sum_diag2_X  # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY**2).sum() - sum_diag2_Y  # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum = (K_XY**2).sum()  # \| K_{XY} \|_F^2

    if biased:
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = Kt_XX_sum / (m * (m - 1)) + Kt_YY_sum / (m * (m - 1)) - 2.0 * K_XY_sum / (m * m)

    var_est = (
        2.0
        / (m**2 * (m - 1.0) ** 2)
        * (
            2 * Kt_XX_sums.dot(Kt_XX_sums)
            - Kt_XX_2_sum
            + 2 * Kt_YY_sums.dot(Kt_YY_sums)
            - Kt_YY_2_sum
        )
        - (4.0 * m - 6.0) / (m**3 * (m - 1.0) ** 3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0
        * (m - 2.0)
        / (m**3 * (m - 1.0) ** 2)
        * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0 * (m - 3.0) / (m**3 * (m - 1.0) ** 2) * (K_XY_2_sum)
        - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0
        / (m**3 * (m - 1.0))
        * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0)
        )
    )
    return mmd2, var_est


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret


def compute_distances(pred, true):
    """computes distances between vectors."""
    mse = torch.nn.functional.mse_loss(pred, true).item()
    me = math.sqrt(mse)
    mae = torch.mean(torch.abs(pred - true)).item()
    return mse, me, mae


def compute_distribution_distances(pred: torch.Tensor, true: Union[torch.Tensor, list]):
    """computes distances between distributions.
    pred: [batch, times, dims] tensor
    true: [batch, times, dims] tensor or list[batch[i], dims] of length times

    This handles jagged times as a list of tensors.
    """
    NAMES = [
        "1-Wasserstein",
        "2-Wasserstein",
        "Linear_MMD",
        "Poly_MMD",
        "RBF_MMD",
        "Mean_MSE",
        "Mean_L2",
        "Mean_L1",
        "Median_MSE",
        "Median_L2",
        "Median_L1",
    ]
    is_jagged = isinstance(true, list)
    pred_is_jagged = isinstance(pred, list)
    dists = []
    to_return = []
    names = []
    filtered_names = [name for name in NAMES if not is_jagged or not name.endswith("MMD")]
    ts = len(pred) if pred_is_jagged else pred.shape[1]
    for t in np.arange(ts):
        if pred_is_jagged:
            a = pred[t]
        else:
            a = pred[:, t, :]
        if is_jagged:
            b = true[t]
        else:
            b = true[:, t, :]
        w1 = wasserstein(a, b, power=1)
        w2 = wasserstein(a, b, power=2)
        if not pred_is_jagged and not is_jagged:
            mmd_linear = linear_mmd2(a, b).item()
            mmd_poly = poly_mmd2(a, b, d=2, alpha=1.0, c=2.0).item()
            mmd_rbf = mix_rbf_mmd2(a, b, sigma_list=[0.01, 0.1, 1, 10, 100]).item()
        mean_dists = compute_distances(torch.mean(a, dim=0), torch.mean(b, dim=0))
        median_dists = compute_distances(torch.median(a, dim=0)[0], torch.median(b, dim=0)[0])
        if pred_is_jagged or is_jagged:
            dists.append((w1, w2, *mean_dists, *median_dists))
        else:
            dists.append((w1, w2, mmd_linear, mmd_poly, mmd_rbf, *mean_dists, *median_dists))
        # For multipoint datasets add timepoint specific distances
        if ts > 1:
            names.extend([f"t{t+1}/{name}" for name in filtered_names])
            to_return.extend(dists[-1])

    to_return.extend(np.array(dists).mean(axis=0))
    names.extend(filtered_names)
    return names, to_return


def rand_hyperplane(shape, device='cuda:0'):

    # Uniformly sample the angle direction
    N = torch.prod(torch.tensor(shape[1:])) 

    samples_norm = np.random.beta(a=N / 2., b=128 / 2.,
                                 size=shape[0]).astype(np.double)
    inverse_beta = samples_norm / (1 - samples_norm +1e-8)
    inverse_beta = torch.from_numpy(inverse_beta).to(device).double()
    # Sampling from p_r(R) by change-of-variable (c.f. Appendix B)
    samples_norm = (torch.sqrt(inverse_beta +1e-8)).view(len(samples_norm), -1)
    
    gaussian = torch.randn(shape[0], N, device=device)
    unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)

    perturbation_x = (unit_gaussian * samples_norm).float()
    
    return perturbation_x.view(shape)