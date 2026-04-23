import os
import math
import itertools
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.stats import pearsonr, linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable

from vb_prism_1 import VariationalPrismNeuralGLM, run_variational_e_step


def _find_best_tau(A_hat_block, A_true_block):
    """Adaptive threshold: scan quantiles of |A_hat| to maximise F1 on this block."""
    abs_hat = np.abs(A_hat_block).ravel()
    nz = abs_hat[abs_hat > 1e-12]
    if nz.size == 0:
        return 0.0
    candidates = np.unique(np.quantile(nz, np.linspace(0.70, 0.995, 60)))
    candidates = candidates[candidates > 1e-12]
    if candidates.size == 0:
        return float(np.min(nz))
    best_f1, best_tau = -1.0, float(candidates[0])
    for tau in candidates:
        tm = np.abs(A_true_block) > tau
        hm = np.abs(A_hat_block) > tau
        tp = int((tm & hm).sum())
        fp = int(((~tm) & hm).sum())
        fn = int((tm & (~hm)).sum())
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_tau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Metrics helpers
# -------------------------
def _block_metrics(A_hat, A_true, tau_exc, tau_inh, n_exc):
    N = A_true.shape[0]
    true_mask = np.zeros((N, N), dtype=bool)
    hat_mask  = np.zeros((N, N), dtype=bool)
    true_mask[:n_exc, :] = np.abs(A_true[:n_exc, :]) > tau_exc
    true_mask[n_exc:, :] = np.abs(A_true[n_exc:, :]) > tau_inh
    hat_mask[:n_exc, :]  = np.abs(A_hat[:n_exc, :])  > tau_exc
    hat_mask[n_exc:, :]  = np.abs(A_hat[n_exc:, :])  > tau_inh

    def prf(t, h):
        tp = int((t & h).sum()); fp = int((~t & h).sum()); fn = int((t & ~h).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return tp, fp, fn, p, r, f

    tp, fp, fn, prec, rec, f1 = prf(true_mask, hat_mask)
    tn = int((~true_mask & ~hat_mask).sum())
    tp_e, fp_e, fn_e, p_e, r_e, f_e = prf(true_mask[:n_exc, :], hat_mask[:n_exc, :])
    tp_i, fp_i, fn_i, p_i, r_i, f_i = prf(true_mask[n_exc:, :], hat_mask[n_exc:, :])
    tp_es, fp_es, fn_es, p_es, r_es, f_es = prf(
        A_true[:n_exc, :] > tau_exc, A_hat[:n_exc, :] > tau_exc)
    tp_is, fp_is, fn_is, p_is, r_is, f_is = prf(
        A_true[n_exc:, :] < -tau_inh, A_hat[n_exc:, :] < -tau_inh)

    return dict(
        true_mask=true_mask, hat_mask=hat_mask,
        tp=tp, fp=fp, fn=fn, tn=tn, prec=prec, rec=rec, f1=f1,
        acc=(tp + tn) / (N * N),
        p_e=p_e, r_e=r_e, f_e=f_e,
        p_i=p_i, r_i=r_i, f_i=f_i,
        p_es=p_es, r_es=r_es, f_es=f_es,
        p_is=p_is, r_is=r_is, f_is=f_is,
    )


def _draw_confusion_panel(ax, fig, conf_image, cmap_conf, title):
    im = ax.imshow(conf_image, cmap=cmap_conf)
    ax.set_title(title)
    ax.set_xlabel("Presynaptic neuron index")
    ax.set_ylabel("Postsynaptic neuron index")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar.ax.set_yticklabels(['TN', 'FN', 'FP', 'TP'])


def _draw_stats_panel(ax, m, tau_exc, tau_inh, label=""):
    tp, fp, fn = m['tp'], m['fp'], m['fn']
    bars = ax.bar(
        ['True Positive\n(TP)', 'False Positive\n(FP)', 'False Negative\n(FN)'],
        [tp, fp, fn], color=['#38761d', '#ed2f20', '#df20df']
    )
    ax.set_title(f"Edge classification counts ({label})" if label else "Edge classification counts", fontsize=10)
    ax.set_xlabel("Prediction outcome", fontsize=8)
    ax.set_ylabel("Number of off-diagonal edges", fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, [tp, fp, fn]):
        ax.text(bar.get_x() + bar.get_width() / 2, v / 2, str(int(v)),
                ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        ax.text(bar.get_x() + bar.get_width() / 2, v, str(int(v)),
                ha='center', va='bottom', fontweight='bold', fontsize=18)
    stats_text = (
        f"precision = {m['prec']:.3f}\n"
        f"recall    = {m['rec']:.3f}\n"
        f"F1        = {m['f1']:.3f}\n"
        f"accuracy  = {m['acc']:.3f}\n"
        f"τ_exc     = {tau_exc:.4f}\n"
        f"τ_inh     = {tau_inh:.4f}"
    )
    block_text = (
        f"Excitatory block (abs):  p={m['p_e']:.3f}  r={m['r_e']:.3f}  F1={m['f_e']:.3f}\n"
        f"Inhibitory block (abs):  p={m['p_i']:.3f}  r={m['r_i']:.3f}  F1={m['f_i']:.3f}\n"
        f"Excitatory block (sign): p={m['p_es']:.3f}  r={m['r_es']:.3f}  F1={m['f_es']:.3f}\n"
        f"Inhibitory block (sign): p={m['p_is']:.3f}  r={m['r_is']:.3f}  F1={m['f_is']:.3f}"
    )
    ax.text(0.02, 0.98, stats_text, fontsize=9, verticalalignment='top',
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.text(0.02, 0.52, block_text, fontsize=8, verticalalignment='top',
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8))


def _best_state_permutation_by_correlation(B_true, B_hat):
    K = B_true.shape[0]
    best_r_sum, best_perm = -np.inf, tuple(range(K))
    for perm in itertools.permutations(range(K)):
        r_sum = sum(abs(pearsonr(B_true[k], B_hat[perm[k]])[0]) for k in range(K))
        if r_sum > best_r_sum:
            best_r_sum, best_perm = r_sum, perm
    return list(best_perm)


def _plot_bias_correlation_panel(ax, B_true, B_hat, state_idx, color):
    r_val, _ = pearsonr(B_true, B_hat)
    b_min = min(B_true.min(), B_hat.min())
    b_max = max(B_true.max(), B_hat.max())
    ax.scatter(B_true, B_hat, color=color, alpha=0.5)
    ax.plot([b_min, b_max], [b_min, b_max], 'k--', alpha=0.7)
    ax.set_xlim(b_min, b_max); ax.set_ylim(b_min, b_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Log-baseline correlation, state {state_idx} (R={r_val:.3f})")
    ax.set_xlabel(f"True $B_{{true}}$ (state {state_idx})")
    ax.set_ylabel(f"Inferred $\\hat{{B}}$ (state {state_idx})")


def _smooth(x, w=20):
    x = np.array(x, dtype=float)
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    padded = np.concatenate([np.full(w - 1, x[0]), x])
    return np.convolve(padded, kernel, mode='valid')


# -------------------------
# Ground truth vs recovered
# -------------------------
def plot_ground_truth_vs_recovered(A_true, A_hat, dataset_name, out_path, suffix=""):
    N = A_true.shape[0]
    n_exc = int(0.8 * N)

    tau_exc = _find_best_tau(A_hat[:n_exc, :], A_true[:n_exc, :])
    tau_inh = _find_best_tau(A_hat[n_exc:, :], A_true[n_exc:, :])
    m = _block_metrics(A_hat, A_true, tau_exc, tau_inh, n_exc)

    mask_off = ~np.eye(N, dtype=bool)
    r_val, _ = pearsonr(A_true[mask_off], A_hat[mask_off])
    vabs = min(max(np.abs(A_true).max(), np.abs(A_hat).max()) * 0.9, 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"Ground truth vs recovered: {dataset_name}\n"
        f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  F1={m['f1']:.3f}  R={r_val:.3f}",
        fontsize=12,
    )
    tick_positions = list(range(0, N + 1, 20))
    for ax, mat, title in zip(axes, [A_true, A_hat],
                               ["Ground truth $A_{true}$", "Recovered $\\hat{A}$"]):
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-vabs, vmax=vabs, aspect='equal')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Presynaptic neuron index", fontsize=10)
        ax.set_ylabel("Postsynaptic neuron index", fontsize=10)
        ax.set_xticks(tick_positions); ax.set_yticks(tick_positions)
        ax.tick_params(labelsize=8)
        ax.invert_yaxis()
        ax.axhline(n_exc - 0.5, color='black', lw=1.0, ls='--', alpha=0.6)
        ax.axvline(n_exc - 0.5, color='black', lw=1.0, ls='--', alpha=0.6)

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im, cax=cax, label="Synaptic weight")
    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    save_path = f"{out_path}/{dataset_name}_ground_truth_vs_recovered_{suffix}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Ground truth vs recovered plot saved to: {save_path}")


# -------------------------
# Full connectivity validation
# -------------------------
def plot_connectivity_validation(model, A_true, B_true, dataset_name, out_path, suffix=""):
    A_hat = model.A.detach().cpu().numpy()
    B_hat = model.B.detach().cpu().numpy()
    N = model.N
    n_exc = int(0.8 * N)

    tau_exc = _find_best_tau(A_hat[:n_exc, :], A_true[:n_exc, :])
    tau_inh = _find_best_tau(A_hat[n_exc:, :], A_true[n_exc:, :])
    m = _block_metrics(A_hat, A_true, tau_exc, tau_inh, n_exc)

    conf_img = np.zeros_like(A_hat)
    conf_img[m['true_mask'] & m['hat_mask']]   = 3  # TP
    conf_img[~m['true_mask'] & m['hat_mask']]  = 2  # FP
    conf_img[m['true_mask'] & ~m['hat_mask']]  = 1  # FN
    cmap_conf = mcolors.ListedColormap(['white', '#df20df', '#ed2f20', '#38761d'])

    K_b = B_true.shape[0]
    n_bias_cols = min(3, max(1, K_b))
    n_bias_rows = int(np.ceil(K_b / n_bias_cols))
    fig = plt.figure(figsize=(18, 5 + 4 * n_bias_rows))
    gs = fig.add_gridspec(1 + n_bias_rows, 3, height_ratios=[1.0] + [1.0] * n_bias_rows)
    axes_top = np.array([fig.add_subplot(gs[0, col]) for col in range(3)])
    plt.suptitle(f"VB-PRISM Fit Validation: {dataset_name}", fontsize=16)

    tick_positions = list(range(0, N + 1, 20))
    _draw_stats_panel(axes_top[0], m, tau_exc, tau_inh, label="")

    _draw_confusion_panel(axes_top[1], fig, conf_img, cmap_conf,
                          f"Edge prediction confusion map  (F1={m['f1']:.3f})")
    axes_top[1].set_xticks(tick_positions); axes_top[1].set_yticks(tick_positions)
    axes_top[1].invert_yaxis()

    mask_off = ~np.eye(N, dtype=bool)
    slope, intercept, r_val, _, _ = linregress(A_true[mask_off], A_hat[mask_off])
    x_line = np.array([-0.4, 0.4])
    axes_top[2].scatter(A_true[mask_off], A_hat[mask_off], alpha=0.3, s=5)
    axes_top[2].plot(x_line, x_line, 'r--', label='slope=1')
    axes_top[2].plot(x_line, slope * x_line + intercept, 'b-', lw=1.2,
                     label=f'fit slope={slope:.3f}')
    axes_top[2].legend(fontsize=8)
    axes_top[2].set_title(f"Weight correlation  R={r_val:.3f}  slope={slope:.3f}")
    axes_top[2].set_xlabel("True weight $A_{true}$ (off-diagonal)")
    axes_top[2].set_ylabel("Inferred weight $\\hat{A}$ (off-diagonal)")

    best_b_perm = _best_state_permutation_by_correlation(B_true, B_hat)
    b_colors = ['blue', 'red', 'green', 'purple', 'orange']
    bias_axes = [fig.add_subplot(gs[row + 1, col])
                 for row in range(n_bias_rows) for col in range(3)]
    for k in range(K_b):
        _plot_bias_correlation_panel(bias_axes[k], B_true[k],
                                     B_hat[best_b_perm[k]], state_idx=k,
                                     color=b_colors[k % len(b_colors)])
    for ax in bias_axes[K_b:]:
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    save_path = f"{out_path}/{dataset_name}_connectivity_validation_{suffix}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Connectivity validation plot saved to: {save_path}")


# -------------------------
# Residuals
# -------------------------
def plot_residuals(model, A_true, B_true, dataset_name, out_path, suffix=""):
    A_hat = model.A.detach().cpu().numpy()
    B_hat = model.B.detach().cpu().numpy()
    N = model.N
    n_exc = int(0.8 * N)
    mask_off = ~np.eye(N, dtype=bool)
    thresh = 1e-6

    inh_true_flat = A_true[n_exc:][mask_off[n_exc:]]
    inh_hat_flat  = A_hat[n_exc:][mask_off[n_exc:]]
    inh_nz   = np.abs(inh_true_flat) > thresh
    inh_resid = inh_hat_flat[inh_nz] - inh_true_flat[inh_nz]

    exc_true_flat = A_true[:n_exc][mask_off[:n_exc]]
    exc_hat_flat  = A_hat[:n_exc][mask_off[:n_exc]]
    exc_nz   = np.abs(exc_true_flat) > thresh
    exc_resid = exc_hat_flat[exc_nz] - exc_true_flat[exc_nz]

    diag_idx = np.arange(N)
    diag_resid = A_hat[diag_idx, diag_idx] - A_true[diag_idx, diag_idx]

    tau_exc = _find_best_tau(A_hat[:n_exc, :], A_true[:n_exc, :])
    tau_inh = _find_best_tau(A_hat[n_exc:, :], A_true[n_exc:, :])
    m = _block_metrics(A_hat, A_true, tau_exc, tau_inh, n_exc)

    best_b_perm = _best_state_permutation_by_correlation(B_true, B_hat)
    K_b = B_true.shape[0]
    B_hat_aligned = np.array([B_hat[best_b_perm[k]] for k in range(K_b)])
    b_resid_all = (B_hat_aligned - B_true).ravel()
    b_true_flat = B_true.ravel()
    b_split = np.median(b_true_flat)
    b_resid_low  = b_resid_all[b_true_flat <  b_split]
    b_resid_high = b_resid_all[b_true_flat >= b_split]

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    fig.suptitle(f"Edge & Weight Residuals: {dataset_name}", fontsize=13)

    def _hist_panel(ax, data, title, color):
        if data.size == 0:
            ax.set_title(title, fontsize=9)
            ax.text(0.5, 0.5, "no data", ha='center', va='center', transform=ax.transAxes)
            return
        med, std = np.median(data), np.std(data)
        ax.hist(data, bins=30, color=color, alpha=0.65, edgecolor='none')
        ax.axvline(0, color='k', ls='--', lw=0.9, alpha=0.6)
        mid_y = ax.get_ylim()[1] * 0.55
        ax.plot(med, mid_y, 'ko', markersize=7, zorder=5)
        ax.errorbar(med, mid_y, xerr=std, color='k', lw=1.5, capsize=4, zorder=5)
        ax.text(0.04, 0.97, f"med={med:.3f}\nstd={std:.3f}", transform=ax.transAxes,
                fontsize=8, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Residual (fit\u2212true)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)

    _hist_panel(axes[0], inh_resid,  "A_off (Inhib) resid", 'steelblue')
    _hist_panel(axes[1], exc_resid,  "A_off (Excit) resid", 'salmon')
    _hist_panel(axes[2], diag_resid, "A_diag residual",     'lightsalmon')

    tp, fp, fn = m['tp'], m['fp'], m['fn']
    bars = axes[3].bar(['TP', 'FP', 'FN'], [tp, fp, fn],
                       color=['#38761d', '#ed2f20', '#df20df'])
    for bar, v in zip(bars, [tp, fp, fn]):
        axes[3].text(bar.get_x() + bar.get_width() / 2, v, str(int(v)),
                     ha='center', va='bottom', fontweight='bold', fontsize=14)
    axes[3].set_title("A_off Edge Accuracy", fontsize=9)
    axes[3].set_xlabel("Prediction class", fontsize=8)
    axes[3].grid(axis='y', alpha=0.3)

    _hist_panel(axes[4], b_resid_low,  f"B (Low < {b_split:.2f}) resid",     'steelblue')
    _hist_panel(axes[5], b_resid_high, f"B (High \u2265 {b_split:.2f}) resid", 'salmon')

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    save_path = f"{out_path}/{dataset_name}_residuals_{suffix}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Residuals plot saved to: {save_path}")


# -------------------------
# Spectral radius
# -------------------------
def plot_spectral_radius(model, dataset_name, out_path, suffix="", metrics=None):
    A_np = model.A.detach().cpu().numpy()
    eigvals = np.linalg.eigvals(A_np)
    rho_final = float(np.max(np.abs(eigvals)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Spectral radius monitor: {dataset_name}\n"
        f"Final rho(A) = {rho_final:.4f}  "
        f"({'STABLE' if rho_final < 1.0 else 'UNSTABLE'})",
        fontsize=11,
    )

    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, alpha=0.5, label='|z|=1')
    ax.plot(0.92 * np.cos(theta), 0.92 * np.sin(theta), 'r:', lw=0.8, alpha=0.6, label='|z|=0.92')
    re, im = eigvals.real, eigvals.imag
    ax.scatter(re, im, s=18, color='steelblue', alpha=0.75, zorder=3,
               label=f'Eigenvalues (N={len(eigvals)})')
    idx_max = np.argmax(np.abs(eigvals))
    ax.scatter(re[idx_max], im[idx_max], s=80, color='red', zorder=4,
               label=f'Max |λ| = {rho_final:.4f}')
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.4, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.4, alpha=0.5)
    ax.set_title(f"Eigenvalue spectrum  ρ(A) = {rho_final:.4f}", fontsize=10)
    ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)")
    ax.legend(fontsize=8, loc='upper left')

    ax2 = axes[1]
    has_hist = (metrics is not None and 'spectral_radius' in metrics
                and len(metrics.get('spectral_radius', [])) > 0
                and 'epochs' in metrics
                and len(metrics['epochs']) == len(metrics['spectral_radius']))
    if has_hist:
        epochs  = np.array(metrics['epochs'], dtype=float)
        rho_raw = np.array(metrics['spectral_radius'], dtype=float)
        w = max(1, len(epochs) // 30)
        ax2.plot(epochs, _smooth(rho_raw, w), color='green', lw=1.5, label='ρ(A) (smoothed)')
        ax2.scatter(epochs[-1], rho_final, s=80, color='red', zorder=4,
                    label=f'Final ρ = {rho_final:.4f}')
        ax2.set_xlabel("M-step epoch", fontsize=9)
        ax2.set_title("ρ(A) training trajectory", fontsize=10)
    else:
        ax2.bar(['Final ρ(A)'], [rho_final], color='steelblue', width=0.4)
        ax2.text(0, rho_final + 0.005, f'{rho_final:.4f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax2.set_title("Final spectral radius (no per-epoch history)", fontsize=10)
        ax2.set_ylim(0, max(1.1, rho_final * 1.15))
    ax2.axhline(1.0,  color='orange', ls=':', lw=1.0, label='ρ=1.0')
    ax2.axhline(0.92, color='red',    ls='--', lw=1.2, label='ρ_max=0.92')
    ax2.set_ylabel("Spectral radius ρ(A)")
    ax2.legend(fontsize=8); ax2.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    save_path = f"{out_path}/{dataset_name}_spectral_radius_{suffix}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Spectral radius plot saved to: {save_path}")


# -------------------------
# Training monitor
# -------------------------
def plot_training_monitor(metrics, dataset_name, out_path, suffix=""):
    if not metrics or 'epochs' not in metrics:
        print("No training metrics to plot (model was saved without metrics dict).")
        return

    epochs = np.array(metrics['epochs'])
    w = max(1, len(epochs) // 30)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    plt.suptitle(f"VB-PRISM Training Monitor: {dataset_name} {suffix}", fontsize=14)

    if 'em_iters' in metrics and 'nll_bin' in metrics:
        axes[0].plot(metrics['em_iters'], metrics['nll_bin'], marker='o', color='tab:blue')
    axes[0].set_title("E-step NLL per bin"); axes[0].set_xlabel("EM iteration")
    axes[0].set_ylabel("Neg. log-likelihood / bin")

    rho_raw = np.array(metrics.get('spectral_radius', []), dtype=float)
    if rho_raw.size > 0 and rho_raw.size == epochs.size:
        axes[1].plot(epochs, _smooth(rho_raw, w), color='green', lw=1.8, label='ρ(A)')
        axes[1].axhline(1.0,  color='orange', ls=':', lw=1.0, label='ρ=1.0')
        axes[1].axhline(0.92, color='red',    ls='--', lw=1.2, label='ρ_max=0.92')
        axes[1].set_ylim(bottom=0); axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "spectral_radius not logged\n(add to metrics in train_vb_prism)",
                     ha='center', va='center', transform=axes[1].transAxes, fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1].set_title("Spectral radius ρ(A)"); axes[1].set_xlabel("M-step epoch")

    nz_raw = np.array(metrics.get('nz_edges', []), dtype=float)
    if nz_raw.size == epochs.size:
        axes[2].plot(epochs, _smooth(nz_raw, w), color='purple', lw=1.8)
    axes[2].set_title("Connectivity support"); axes[2].set_xlabel("M-step epoch")
    axes[2].set_ylabel("Nonzero edges (|Aᵢⱼ| > 0.01)")

    nll_raw = np.array(metrics.get('nll_bin_M', []), dtype=float)
    l1_raw  = np.array(metrics.get('l1_norm', []),   dtype=float)
    if nll_raw.size == epochs.size:
        ax4r = axes[3].twinx()
        axes[3].plot(epochs, _smooth(nll_raw, w), color='tab:blue', lw=1.8, label='NLL')
        if l1_raw.size == epochs.size:
            ax4r.plot(epochs, _smooth(l1_raw, w), color='tab:red', lw=1.8, ls='--', label='L1')
        axes[3].set_ylabel("Poisson NLL / bin", color='tab:blue')
        ax4r.set_ylabel("L1 norm ‖A‖₁", color='tab:red')
    axes[3].set_title("M-step NLL and L1 norm"); axes[3].set_xlabel("M-step epoch")

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    save_path = f"{out_path}/{dataset_name}_training_monitor_{suffix}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training monitor plot saved to: {save_path}")


# -------------------------
# State prediction (full)
# -------------------------
def _encode_labels_zero_based(z, K):
    z = np.asarray(z).astype(int)
    uniq = np.unique(z)
    if uniq.size != K:
        return z
    mapping = {lab: i for i, lab in enumerate(sorted(uniq.tolist()))}
    return np.array([mapping[v] for v in z], dtype=int)


def _align_states_to_truth(gamma, z_true):
    T, K = gamma.shape
    z_true_enc = _encode_labels_zero_based(z_true, K)
    z_hat = np.argmax(gamma, axis=1)
    best_acc, best_perm = -1.0, tuple(range(K))
    for perm in itertools.permutations(range(K)):
        perm_arr = np.array(perm, dtype=int)
        acc = np.mean(perm_arr[z_hat] == z_true_enc)
        if acc > best_acc:
            best_acc, best_perm = acc, tuple(perm_arr.tolist())
    best_perm = np.array(best_perm, dtype=int)
    gamma_aligned  = gamma[:, best_perm]
    z_hat_aligned  = best_perm[z_hat]
    per_state_acc = [
        np.mean(z_hat_aligned[z_true_enc == k] == k) if np.any(z_true_enc == k) else np.nan
        for k in range(K)
    ]
    return gamma_aligned, z_true_enc, z_hat_aligned, best_acc, per_state_acc


def plot_state_prediction(gamma, z_true, dataset_name, out_path,
                          time_step=0.01, max_time_s=None):
    T, K = gamma.shape
    t = np.arange(T) * time_step
    if max_time_s is None:
        max_time_s = min(t[-1], 1000.0)
    t_mask = t <= max_time_s

    if z_true is not None:
        gamma, z_true_aligned, z_hat, acc, state_acc = _align_states_to_truth(gamma, z_true)
        z_true_plot = z_true_aligned
    else:
        z_hat = np.argmax(gamma, axis=1)
        acc, state_acc, z_true_plot = np.nan, [np.nan] * K, None
    conf = np.max(gamma, axis=1)

    fig = plt.figure(figsize=(30, 9), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 3, height_ratios=[1, 2.3], hspace=0.28, wspace=0.30)
    fig.suptitle(f"VB-PRISM State Prediction: {dataset_name}", fontsize=11)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(np.arange(K), np.mean(gamma, axis=0), color='steelblue', alpha=0.75)
    ax1.set_xticks(range(K))
    ax1.set_xticklabels([f"State {k}" for k in range(K)], fontsize=8)
    ax1.set_title("Mean occupancy"); ax1.set_ylabel("Mean posterior")

    ax2 = fig.add_subplot(gs[0, 1])
    vals = [0.0 if np.isnan(v) else v for v in state_acc]
    ax2.bar(np.arange(K), vals, color='seagreen', alpha=0.75)
    ax2.set_xticks(range(K))
    ax2.set_xticklabels([f"State {k}" for k in range(K)], fontsize=8)
    for i, v in enumerate(state_acc):
        if not np.isnan(v):
            ax2.text(i, v + 0.03, f"{v:.2f}", ha='center', fontsize=8)
    ax2.set_ylim(0, 1.05)
    ttl = f"State accuracy  avg={acc:.3f}" if not np.isnan(acc) else "State accuracy (no truth)"
    ax2.set_title(ttl); ax2.set_ylabel("Accuracy")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(conf, bins=40, color='mediumpurple', alpha=0.7)
    ax3.set_title(f"Confidence (acc={acc:.3f})" if not np.isnan(acc) else "Confidence")
    ax3.set_xlabel("max_k γ(t,k)"); ax3.set_ylabel("count")

    ax4 = fig.add_subplot(gs[1, :])
    if z_true_plot is not None:
        ax4.step(t[t_mask], z_true_plot[t_mask], where='post', color='black', lw=1.6,
                 label=r'True state $S_t$')
    ax4.step(t[t_mask], z_hat[t_mask], where='post', color='tab:orange', alpha=0.85, lw=1.2,
             label=r'Inferred state $\hat{S}_t$ (argmax posterior)')
    ax4.set_title(f"State fit (acc={acc:.3f})" if not np.isnan(acc) else "State fit")
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Latent state")
    ax4.set_yticks(list(range(K)))
    ax4.set_yticklabels([f"State {k}" for k in range(K)], fontsize=8)
    ax4.legend(loc='upper right', fontsize=8)
    if np.any(t_mask):
        ax4.set_xlim(t[t_mask][0], t[t_mask][-1])
        ax4.set_xticks(np.linspace(t[t_mask][0], t[t_mask][-1], 11))

    os.makedirs(out_path, exist_ok=True)
    save_path = f"{out_path}/{dataset_name}_state_prediction_{int(max_time_s)}s.png"
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"State prediction plot saved to: {save_path}")


# -------------------------
# Main entry point
# -------------------------
def run_plots(dataset, data_path, model_path, out_path):
    os.makedirs(out_path, exist_ok=True)

    print(f"Loading VB-PRISM model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    data_load  = np.load(f"{data_path}/{dataset}.spikes.npz")
    truth_load = np.load(f"{data_path}/{dataset}.prismTruth.npz")
    A_true = truth_load['A_true']
    B_true = truth_load['B_true']
    Y_np   = data_load['spikes'].astype(np.float32)
    z_true = truth_load.get('S_true', None)
    N = A_true.shape[0]

    Y_torch = torch.from_numpy(Y_np).to(device)
    X_torch = torch.zeros_like(Y_torch)
    X_torch[1:] = Y_torch[:-1]

    # Reconstruct model from checkpoint
    # Support both plain state_dict saves and full checkpoint dicts
    if isinstance(checkpoint, dict) and 'B' in checkpoint:
        K = checkpoint['B'].shape[0]
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        K = state_dict['B'].shape[0]
    else:
        raise ValueError("Unrecognised checkpoint format.")

    model = VariationalPrismNeuralGLM(n_states=K, n_neurons=N).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print("Running E-step for visualization...")
    with torch.no_grad():
        gamma, _, _ = run_variational_e_step(model, Y_torch, X_torch, delta_t=0.01)
        gamma_np = gamma.cpu().numpy()

    A_hat = model.A.detach().cpu().numpy()
    metrics = checkpoint.get('metrics', {}) if isinstance(checkpoint, dict) else {}

    print("Generating ground truth vs recovered...")
    plot_ground_truth_vs_recovered(A_true, A_hat, dataset, out_path, suffix="VB")

    print("Generating full connectivity validation...")
    plot_connectivity_validation(model, A_true, B_true, dataset, out_path, suffix="VB")

    print("Generating residuals...")
    plot_residuals(model, A_true, B_true, dataset, out_path, suffix="VB")

    print("Generating spectral radius...")
    plot_spectral_radius(model, dataset, out_path, suffix="VB", metrics=metrics)

    print("Generating training monitor...")
    plot_training_monitor(metrics, dataset, out_path, suffix="VB")

    print("Generating state prediction...")
    plot_state_prediction(gamma_np, z_true, dataset, out_path, time_step=0.01, max_time_s=10.0)

    print(f"Done. All plots saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str, default="daleN100_f7d754_ce2202")
    parser.add_argument("--data-path",  type=str, default="/pscratch/sd/s/sanjitr/causal_net_temp/spikesData")
    parser.add_argument("--model-path", type=str, default="vb_prism_best_model.pt")
    parser.add_argument("--out-path",   type=str, default="./plots/vb_prism_plots")
    args = parser.parse_args()
    run_plots(args.dataset, args.data_path, args.model_path, args.out_path)
