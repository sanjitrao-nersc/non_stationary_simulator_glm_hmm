import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from ashwood_ns.ashwood_ns_8 import PrismNeuralGLMHMM, compute_expectations, find_best_tau_block

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from scipy.stats import linregress

import matplotlib.colors as mcolors
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable

def _block_metrics(A_eval, A_true, tau_exc, tau_inh, n_exc):
    """Compute abs and sign-aware block metrics. Returns a dict."""
    N = A_true.shape[0]
    true_mask = np.zeros((N, N), dtype=bool)
    hat_mask = np.zeros((N, N), dtype=bool)
    true_mask[:n_exc, :] = np.abs(A_true[:n_exc, :]) > tau_exc
    true_mask[n_exc:, :] = np.abs(A_true[n_exc:, :]) > tau_inh
    hat_mask[:n_exc, :] = np.abs(A_eval[:n_exc, :]) > tau_exc
    hat_mask[n_exc:, :] = np.abs(A_eval[n_exc:, :]) > tau_inh

    def prf(t, h):
        tp = int((t & h).sum())
        fp = int((~t & h).sum())
        fn = int((t & ~h).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return tp, fp, fn, p, r, f

    tp, fp, fn, prec, rec, f1 = prf(true_mask, hat_mask)
    tn = int((~true_mask & ~hat_mask).sum())
    tp_e, fp_e, fn_e, p_e, r_e, f_e = prf(true_mask[:n_exc, :], hat_mask[:n_exc, :])
    tp_i, fp_i, fn_i, p_i, r_i, f_i = prf(true_mask[n_exc:, :], hat_mask[n_exc:, :])

    # Sign-aware (positive exc, negative inh)
    tp_es, fp_es, fn_es, p_es, r_es, f_es = prf(
        A_true[:n_exc, :] > tau_exc, A_eval[:n_exc, :] > tau_exc)
    tp_is, fp_is, fn_is, p_is, r_is, f_is = prf(
        A_true[n_exc:, :] < -tau_inh, A_eval[n_exc:, :] < -tau_inh)

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


def _draw_stats_panel(ax, m, tau_exc, tau_inh, label):
    tp, fp, fn = m['tp'], m['fp'], m['fn']
    bars = ax.bar(
        ['True Positive\n(TP)', 'False Positive\n(FP)', 'False Negative\n(FN)'],
        [tp, fp, fn],
        color=['#38761d', '#ed2f20', '#df20df']
    )
    title = f"Edge classification counts ({label})" if label else "Edge classification counts"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Prediction outcome\n(TP: correct edge, FP: spurious edge, FN: missed edge)", fontsize=8)
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
    """Resolve label switching by maximizing the sum of per-state |Pearson r|."""
    K = B_true.shape[0]
    best_r_sum = -np.inf
    best_perm = tuple(range(K))
    for perm in itertools.permutations(range(K)):
        r_sum = sum(abs(pearsonr(B_true[k], B_hat[perm[k]])[0]) for k in range(K))
        if r_sum > best_r_sum:
            best_r_sum = r_sum
            best_perm = perm
    return list(best_perm)


def _plot_bias_correlation_panel(ax, B_true, B_hat, state_idx, color):
    r_val, _ = pearsonr(B_true, B_hat)
    b_min = min(B_true.min(), B_hat.min())
    b_max = max(B_true.max(), B_hat.max())
    ax.scatter(B_true, B_hat, color=color, alpha=0.5)
    ax.plot([b_min, b_max], [b_min, b_max], 'k--', alpha=0.7)
    ax.set_xlim(b_min, b_max)
    ax.set_ylim(b_min, b_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Log-baseline bias correlation, state {state_idx} (R={r_val:.3f})")
    ax.set_xlabel(f"True log-baseline $B_{{true}}$ (state {state_idx})")
    ax.set_ylabel(f"Inferred log-baseline $\\hat{{B}}$ (state {state_idx})")


def plot_residuals(model, A_true, B_true, dataset_name, out_path, suffix="", metrics=None):
    """
    6-panel residual diagnostics:
      (0) A_off Inhib residual histogram
      (1) A_off Excit residual histogram
      (2) A_diag residual histogram
      (3) Edge accuracy bar chart (TP / FP / FN)
      (4) B residual for low-baseline neurons (B_true < median)
      (5) B residual for high-baseline neurons (B_true >= median)
    All residuals are signed: fit - true.
    """
    A_hat = model.A.detach().cpu().numpy()[:, :model.N]
    B_hat = model.B.detach().cpu().numpy()
    N = model.N
    n_exc = int(0.8 * N)
    mask_off = ~np.eye(N, dtype=bool)
    thresh = 1e-6

    # Inhibitory off-diagonal: true-nonzero entries only
    inh_true_flat = A_true[n_exc:][mask_off[n_exc:]]
    inh_hat_flat  = A_hat[n_exc:][mask_off[n_exc:]]
    inh_nz = np.abs(inh_true_flat) > thresh
    inh_resid = inh_hat_flat[inh_nz] - inh_true_flat[inh_nz]

    # Excitatory off-diagonal: true-nonzero entries only
    exc_true_flat = A_true[:n_exc][mask_off[:n_exc]]
    exc_hat_flat  = A_hat[:n_exc][mask_off[:n_exc]]
    exc_nz = np.abs(exc_true_flat) > thresh
    exc_resid = exc_hat_flat[exc_nz] - exc_true_flat[exc_nz]

    # Diagonal: all N entries
    diag_idx = np.arange(N)
    diag_resid = A_hat[diag_idx, diag_idx] - A_true[diag_idx, diag_idx]

    # Edge accuracy metrics
    if metrics is not None and 'tau_exc' in metrics and 'tau_inh' in metrics:
        tau_exc = float(metrics['tau_exc'])
        tau_inh = float(metrics['tau_inh'])
    else:
        tau_exc = find_best_tau_block(A_hat[:n_exc], A_true[:n_exc])
        tau_inh = find_best_tau_block(A_hat[n_exc:], A_true[n_exc:])
    m = _block_metrics(A_hat, A_true, tau_exc, tau_inh, n_exc)

    # B residuals split by B_true threshold
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
        med = np.median(data)
        std = np.std(data)
        ax.hist(data, bins=30, color=color, alpha=0.65, edgecolor='none')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.9, alpha=0.6)
        ylims = ax.get_ylim()
        mid_y = ylims[1] * 0.55
        ax.plot(med, mid_y, 'ko', markersize=7, zorder=5)
        ax.errorbar(med, mid_y, xerr=std, color='k', linewidth=1.5, capsize=4, zorder=5)
        ax.text(0.04, 0.97, f"med={med:.3f}\nstd={std:.3f}", transform=ax.transAxes,
                fontsize=8, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Residual (fit\u2212true)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)

    _hist_panel(axes[0], inh_resid,    "A_off (Inhib) resid",  color='steelblue')
    _hist_panel(axes[1], exc_resid,    "A_off (Excit) resid",  color='salmon')
    _hist_panel(axes[2], diag_resid,   "A_diag residual",      color='lightsalmon')

    # Panel 3: edge accuracy bar chart
    tp, fp, fn = m['tp'], m['fp'], m['fn']
    bars = axes[3].bar(['TP', 'FP', 'FN'], [tp, fp, fn],
                       color=['#38761d', '#ed2f20', '#df20df'])
    for bar, v in zip(bars, [tp, fp, fn]):
        axes[3].text(bar.get_x() + bar.get_width() / 2, v, str(int(v)),
                     ha='center', va='bottom', fontweight='bold', fontsize=14)
    axes[3].set_title("A_off Edge Accuracy", fontsize=9)
    axes[3].set_xlabel("Prediction class", fontsize=8)
    axes[3].set_ylabel("Count", fontsize=8)
    axes[3].grid(axis='y', alpha=0.3)

    _hist_panel(axes[4], b_resid_low,  f"B (Low < {b_split:.2f}) resid",   color='steelblue')
    _hist_panel(axes[5], b_resid_high, f"B (High \u2265 {b_split:.2f}) resid", color='salmon')

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    fname_parts = [dataset_name, "residuals"]
    if suffix:
        fname_parts.append(suffix)
    save_path = f"{out_path}/{'_'.join(fname_parts)}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Residuals plot saved to: {save_path}")


def plot_connectivity_validation(
    model,
    A_true,
    B_true,
    dataset_name,
    out_path,
    suffix="",
    metrics=None,
):
    """Validation plot with fixed connectivity diagnostics and one bias-correlation panel per state."""
    A_hat = model.A.detach().cpu().numpy()[:, :model.N]   # lag-1 block only
    B_hat = model.B.detach().cpu().numpy()
    N = model.N
    n_exc = int(0.8 * N)

    if metrics is not None and ('tau_exc' in metrics) and ('tau_inh' in metrics):
        tau_exc = float(metrics['tau_exc'])
        tau_inh = float(metrics['tau_inh'])
    else:
        tau_exc = find_best_tau_block(A_hat[:n_exc, :], A_true[:n_exc, :])
        tau_inh = find_best_tau_block(A_hat[n_exc:, :], A_true[n_exc:, :])

    m = _block_metrics(A_hat, A_true, tau_exc, tau_inh, n_exc)

    conf_img = np.zeros_like(A_hat)
    conf_img[m['true_mask'] & m['hat_mask']] = 3   # TP
    conf_img[~m['true_mask'] & m['hat_mask']] = 2  # FP
    conf_img[m['true_mask'] & ~m['hat_mask']] = 1  # FN

    cmap_conf = mcolors.ListedColormap(['white', '#df20df', '#ed2f20', '#38761d'])

    K_b = B_true.shape[0]
    n_bias_cols = min(3, max(1, K_b))
    n_bias_rows = int(np.ceil(K_b / n_bias_cols))
    fig = plt.figure(figsize=(18, 5 + 4 * n_bias_rows))
    gs = fig.add_gridspec(1 + n_bias_rows, 3, height_ratios=[1.0] + [1.0] * n_bias_rows)
    axes_top = np.array([fig.add_subplot(gs[0, col]) for col in range(3)])
    plt.suptitle(f"PRISM Fit Validation: {dataset_name}", fontsize=16)

    tick_positions = list(range(0, N + 1, 20))

    # (0,0) TP/FP/FN stats panel
    _draw_stats_panel(axes_top[0], m, tau_exc, tau_inh, label="")

    # (0,1) Confusion map
    _draw_confusion_panel(axes_top[1], fig, conf_img, cmap_conf,
                          f"Edge prediction confusion map  (F1={m['f1']:.3f})")
    axes_top[1].set_xlabel("Presynaptic neurons")
    axes_top[1].set_ylabel("Postsynaptic neurons")
    axes_top[1].set_xticks(tick_positions)
    axes_top[1].set_yticks(tick_positions)
    axes_top[1].invert_yaxis()

    # (0,2) A_true vs A_hat weight scatter
    mask_off = ~np.eye(N, dtype=bool)
    slope, intercept, r_val, _, _ = linregress(A_true[mask_off], A_hat[mask_off])
    x_line = np.array([-0.4, 0.4])
    axes_top[2].scatter(A_true[mask_off], A_hat[mask_off], alpha=0.3, s=5)
    axes_top[2].plot(x_line, x_line, 'r--', label='slope=1')
    axes_top[2].plot(x_line, slope * x_line + intercept, 'b-', linewidth=1.2,
                    label=f'fit slope={slope:.3f}')
    axes_top[2].legend(fontsize=8)
    axes_top[2].set_title(f"Connectivity weight correlation  R={r_val:.3f}  slope={slope:.3f}")
    axes_top[2].set_xlabel("True weight $A_{true}$ (off-diagonal)")
    axes_top[2].set_ylabel("Inferred weight $\\hat{A}$ (off-diagonal)")

    # Align B_hat states to B_true by best permutation (resolve label-switching)
    best_b_perm = _best_state_permutation_by_correlation(B_true, B_hat)
    b_colors = ['blue', 'red', 'green', 'purple', 'orange']
    bias_axes = []
    for row in range(n_bias_rows):
        for col in range(3):
            bias_axes.append(fig.add_subplot(gs[row + 1, col]))

    for k in range(K_b):
        _plot_bias_correlation_panel(
            bias_axes[k],
            B_true[k],
            B_hat[best_b_perm[k]],
            state_idx=k,
            color=b_colors[k % len(b_colors)],
        )

    for ax in bias_axes[K_b:]:
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(f"{out_path}/{dataset_name}_connectivity_validation_{suffix}.png", dpi=150)
    plt.close(fig)


def _smooth(x, w=20):
    """Rolling mean with window w. Pads edges so output length matches input."""
    x = np.array(x, dtype=float)
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    padded = np.concatenate([np.full(w - 1, x[0]), x])
    return np.convolve(padded, kernel, mode='valid')


def plot_training_monitor(metrics, dataset_name, out_path, suffix=""):
    """4-panel training monitor: E-step NLL, spectral radius, edge support, M-step NLL + L1."""
    epochs = np.array(metrics['epochs'])
    w = max(1, len(epochs) // 30)   # window ≈ 3% of total epochs

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    plt.suptitle(f"PRISM Training Monitor: {dataset_name} {suffix}", fontsize=14)

    # 1. E-step NLL — one point per EM iteration, already clean
    axes[0].plot(metrics['em_iters'], metrics['nll_bin'], marker='o', color='tab:blue')
    axes[0].set_title("E-step: neg. log-likelihood per bin")
    axes[0].set_xlabel("EM iteration (outer loop)")
    axes[0].set_ylabel("Neg. log-likelihood / bin")

    # 2. Spectral radius — trend only (guard: key may be absent if not logged)
    rho_raw = np.array(metrics.get('spectral_radius', []), dtype=float)
    if rho_raw.size > 0 and rho_raw.size == epochs.size:
        rho_sm = _smooth(rho_raw, w)
        axes[1].plot(epochs, rho_sm, color='green', linewidth=1.8, label='ρ(A) (smoothed)')
        axes[1].axhline(1.0,  color='orange', linestyle=':',  linewidth=1.0,
                        label='ρ = 1.0 (marginal stability)')
        axes[1].axhline(0.92, color='red',    linestyle='--', linewidth=1.2,
                        label='ρ_max = 0.92 (enforced ceiling)')
        axes[1].set_ylim(bottom=0)
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5,
                     "spectral_radius not logged\n(see plot_spectral_radius())",
                     ha='center', va='center', transform=axes[1].transAxes,
                     fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1].set_title("Lag-1 spectral radius ρ(A)")
    axes[1].set_xlabel("M-step epoch (global)")
    axes[1].set_ylabel("Spectral radius ρ(A)")

    # 3. Nonzero edges + adaptive tau schedule (secondary axis)
    nz_raw = np.array(metrics['nz_edges'], dtype=float)
    nz_sm  = _smooth(nz_raw, w)
    l3b, = axes[2].plot(epochs, nz_sm,  color='purple', linewidth=1.8, label='nonzero edges (smoothed)')
    axes[2].set_title("Connectivity support")
    axes[2].set_xlabel("M-step epoch (global)")
    axes[2].set_ylabel("Nonzero edges (|Aᵢⱼ| > 0.01)", color='purple')
    axes[2].legend(handles=[l3b], loc='upper left', fontsize=8)

    # 4. M-step NLL (left) + L1 norm (right) — smooth trend + faint raw
    nll_raw = np.array(metrics['nll_bin_M'], dtype=float)
    l1_raw  = np.array(metrics['l1_norm'],   dtype=float)
    nll_sm  = _smooth(nll_raw, w)
    l1_sm   = _smooth(l1_raw,  w)

    ax4r = axes[3].twinx()
    axes[3].plot(epochs, nll_sm,  color='tab:blue', linewidth=1.8, label='NLL (smoothed)')
    ax4r.plot(epochs, l1_sm,  color='tab:red', linewidth=1.8, linestyle='--', label='L1 (smoothed)')
    axes[3].set_title("M-step: Poisson NLL and L1 norm dynamics")
    axes[3].set_xlabel("M-step epoch (global)")
    axes[3].set_ylabel("Poisson NLL / bin", color='tab:blue')
    ax4r.set_ylabel("L1 norm ‖A‖₁ (off-diagonal)", color='tab:red')

    # Phase 2 boundary marker (epoch where phase 2 begins, if recorded)
    p2_start = metrics.get('phase2_start_epoch', None)
    if p2_start is not None:
        for ax in axes[1:]:
            ax.axvline(p2_start, color='black', linestyle=':', alpha=0.6, label='Phase 2')

    lns = axes[3].get_lines()[-1:] + ax4r.get_lines()[-1:]
    axes[3].legend(lns, [l.get_label() for l in lns], loc='upper right', fontsize=8)

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(f"{out_path}/{dataset_name}_training_monitor_{suffix}.png", dpi=150)
    plt.close(fig)

def _encode_labels_zero_based(z, K):
    """Map arbitrary labels to 0..K-1 where possible."""
    z = np.asarray(z).astype(int)
    uniq = np.unique(z)
    if uniq.size != K:
        return z
    mapping = {lab: i for i, lab in enumerate(sorted(uniq.tolist()))}
    return np.array([mapping[v] for v in z], dtype=int)


def _align_states_to_truth(gamma, z_true):
    """
    Resolve label-switching by permuting inferred states to maximize accuracy vs truth.
    Returns aligned_gamma, aligned_z_hat, best_acc, per_state_acc.
    """
    T, K = gamma.shape
    z_true_enc = _encode_labels_zero_based(z_true, K)
    z_hat = np.argmax(gamma, axis=1)

    best_acc = -1.0
    best_perm = tuple(range(K))
    for perm in itertools.permutations(range(K)):
        perm = np.array(perm, dtype=int)
        z_hat_p = perm[z_hat]
        acc = np.mean(z_hat_p == z_true_enc)
        if acc > best_acc:
            best_acc = acc
            best_perm = tuple(perm.tolist())

    best_perm = np.array(best_perm, dtype=int)
    gamma_aligned = gamma[:, best_perm]
    z_hat_aligned = best_perm[z_hat]
    per_state_acc = [
        np.mean(z_hat_aligned[z_true_enc == k] == k) if np.any(z_true_enc == k) else np.nan
        for k in range(K)
    ]
    return gamma_aligned, z_true_enc, z_hat_aligned, best_acc, per_state_acc


def plot_state_prediction(gamma, z_true, dataset_name, out_path, time_step=0.1, max_time_s=1000.0):
    """
    Jan-style state prediction panel:
    mean occupancy, per-state accuracy, confidence histogram, fit trace, truth trace.
    """
    T, K = gamma.shape
    t = np.arange(T) * time_step
    if max_time_s is None:
        max_time_s = 1000.0
    max_time_s = min(max_time_s, t[-1] if T > 1 else time_step)
    t_mask = t <= max_time_s

    if z_true is not None:
        gamma, z_true_aligned, z_hat, acc, state_acc = _align_states_to_truth(gamma, z_true)
        z_true_plot = z_true_aligned
    else:
        z_hat = np.argmax(gamma, axis=1)
        acc = np.nan
        state_acc = [np.nan] * K
        z_true_plot = None
    conf = np.max(gamma, axis=1)

    fig = plt.figure(figsize=(30, 9), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 2.3], hspace=0.28, wspace=0.30)
    fig.suptitle(f"Dataset: {dataset_name}", fontsize=11)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(np.arange(K), np.mean(gamma, axis=0), color='steelblue', alpha=0.75)
    ax1.set_xticks(range(K))
    ax1.set_xticklabels([f"State {k}" for k in range(K)], fontsize=8)
    ax1.set_title("Mean occupancy", fontsize=9)
    ax1.set_xlabel("State", fontsize=8)
    ax1.set_ylabel("Mean posterior", fontsize=8)
    ax1.tick_params(labelsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    vals = [0.0 if np.isnan(v) else v for v in state_acc]
    ax2.bar(np.arange(K), vals, color='seagreen', alpha=0.75)
    ax2.set_xticks(range(K))
    ax2.set_xticklabels([f"State {k}" for k in range(K)], fontsize=8)
    for i, v in enumerate(state_acc):
        if not np.isnan(v):
            ax2.text(i, v + 0.03, f"{v:.2f}", ha='center', fontsize=8)
    ax2.set_ylim(0, 1.05)
    ttl = f"State accuracy, avr={acc:.3f}" if not np.isnan(acc) else "State accuracy (truth unavailable)"
    ax2.set_title(ttl, fontsize=9)
    ax2.set_xlabel("State", fontsize=8)
    ax2.set_ylabel("Accuracy", fontsize=8)
    ax2.tick_params(labelsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(conf, bins=40, color='mediumpurple', alpha=0.7)
    ttl = f"Confidence (acc={acc:.3f})" if not np.isnan(acc) else "Confidence"
    ax3.set_title(ttl, fontsize=9)
    ax3.set_xlabel("S_hat_CL", fontsize=8)
    ax3.set_ylabel("count", fontsize=8)
    ax3.tick_params(labelsize=8)

    # Single, stretched overlay panel for visual comparison.
    ax4 = fig.add_subplot(gs[1, :])
    if z_true_plot is not None:
        ax4.step(t[t_mask], z_true_plot[t_mask], where='post', color='black', linewidth=1.6,
                 label=r'True state $S_t$')
    ax4.step(t[t_mask], z_hat[t_mask], where='post', color='tab:orange', alpha=0.85, linewidth=1.2,
             label=r'Inferred state $\hat{S}_t$ (argmax posterior)')
    fit_ttl = f"State fit (acc={acc:.3f})" if not np.isnan(acc) else "State fit"
    ax4.set_title(fit_ttl, fontsize=9)
    ax4.set_xlabel("Time (s)", fontsize=8)
    ax4.set_ylabel("Latent state", fontsize=8)
    K_plot = gamma.shape[1]
    ax4.set_yticks(list(range(K_plot)))
    ax4.set_yticklabels([f"State {k}" for k in range(K_plot)], fontsize=8)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.tick_params(labelsize=8)
    if np.any(t_mask):
        ax4.set_xlim(t[t_mask][0], t[t_mask][-1])
        ax4.set_xticks(np.linspace(t[t_mask][0], t[t_mask][-1], 11))

    # Use constrained_layout (set on figure creation) instead of tight_layout
    # to avoid layout warnings with this GridSpec-based panel figure.
    suffix = f"{int(max_time_s)}s"
    save_path = f"{out_path}/{dataset_name}_state_prediction_{suffix}.png"
    plt.savefig(save_path, dpi=180)
    print(f"State prediction plot saved to: {save_path}")



def _reorder_inh_top(A, n_exc):
    """Reorder rows and cols so inhibitory block (rows n_exc:) appears on top."""
    idx = list(range(n_exc, A.shape[0])) + list(range(n_exc))
    return A[np.ix_(idx, idx)]


def plot_ground_truth_vs_recovered(A_true, A_hat, dataset_name, out_path, suffix=""):
    """Side-by-side A_true vs A_hat in original neuron order, y-axis 0 at bottom."""
    N = A_true.shape[0]
    n_exc = int(0.8 * N)

    # Compute summary stats
    tau_exc = find_best_tau_block(A_hat[:n_exc, :], A_true[:n_exc, :])
    tau_inh = find_best_tau_block(A_hat[n_exc:, :], A_true[n_exc:, :])
    m = _block_metrics(A_hat, A_true, tau_exc, tau_inh, n_exc)
    mask_off = ~np.eye(N, dtype=bool)
    r_val, _ = pearsonr(A_true[mask_off], A_hat[mask_off])

    vabs = max(np.abs(A_true).max(), np.abs(A_hat).max()) * 0.9
    vabs = min(vabs, 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"Ground truth vs recovered connectivity: {dataset_name}\n"
        f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  "
        f"F1={m['f1']:.3f}  R={r_val:.3f}",
        fontsize=12,
    )

    tick_positions = list(range(0, N + 1, 20))

    for ax, mat, title in zip(axes, [A_true, A_hat],
                               ["Ground truth $A_{true}$", "Recovered $\\hat{A}$"]):
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-vabs, vmax=vabs, aspect='equal')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Presynaptic neuron index", fontsize=10)
        ax.set_ylabel("Postsynaptic neuron index", fontsize=10)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.tick_params(labelsize=8)
        ax.invert_yaxis()
        # Block boundary between exc (0–79) and inh (80–99)
        ax.axhline(n_exc - 0.5, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
        ax.axvline(n_exc - 0.5, color='black', linewidth=1.0, linestyle='--', alpha=0.6)

    # Anchor colorbar to right panel only so it doesn't push the two matrices apart
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im, cax=cax, label="Synaptic weight")

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    save_path = f"{out_path}/{dataset_name}_ground_truth_vs_recovered_{suffix}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Ground truth vs recovered plot saved to: {save_path}")



def plot_spectral_radius(model, dataset_name, out_path, suffix="", metrics=None):
    """
    Compute rho(A) = max|eigenvalue| of the fitted lag-1 connectivity matrix
    directly from the model object, and plot it alongside any logged history.

    This always works regardless of whether spectral_radius was logged during
    training, because the final value is recomputed from the actual weight matrix.

    Parameters
    ----------
    model   : PrismNeuralGLMHMM  - fitted model with .A parameter [N, L*N]
    metrics : dict, optional      - training metrics dict; if 'spectral_radius'
                                    is present its trajectory is overlaid
    """
    A_np = model.A.detach().cpu().numpy()[:, :model.N]   # lag-1 block [N, N]
    eigvals = np.linalg.eigvals(A_np)
    rho_final = float(np.max(np.abs(eigvals)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Spectral radius monitor: {dataset_name}\n"
        f"Final rho(A) = {rho_final:.4f}  "
        f"({'STABLE' if rho_final < 1.0 else 'UNSTABLE -- rho >= 1'})  |  "
        f"Lag-1 block shape: {A_np.shape}",
        fontsize=11,
    )

    # ---- Panel 1: eigenvalue spectrum in the complex plane ----
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta),
            'k--', lw=0.8, alpha=0.5, label='|z|=1 (stability boundary)')
    ax.plot(0.92 * np.cos(theta), 0.92 * np.sin(theta),
            'r:', lw=0.8, alpha=0.6, label='|z|=0.92 (enforced ceiling)')
    re, im = eigvals.real, eigvals.imag
    ax.scatter(re, im, s=18, color='steelblue', alpha=0.75, zorder=3,
               label=f'Eigenvalues (N={len(eigvals)})')
    idx_max = np.argmax(np.abs(eigvals))
    ax.scatter(re[idx_max], im[idx_max], s=80, color='red', zorder=4,
               label=f'Max |lambda| = {rho_final:.4f}')
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.4, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.4, alpha=0.5)
    ax.set_title(f"Eigenvalue spectrum of A (lag-1 block)\nrho(A) = {rho_final:.4f}", fontsize=10)
    ax.set_xlabel("Re(lambda)", fontsize=9)
    ax.set_ylabel("Im(lambda)", fontsize=9)
    ax.legend(fontsize=8, loc='upper left')

    # ---- Panel 2: training trajectory (if logged) or single-point bar ----
    ax2 = axes[1]
    has_hist = (
        metrics is not None
        and 'spectral_radius' in metrics
        and len(metrics.get('spectral_radius', [])) > 0
        and 'epochs' in metrics
        and len(metrics['epochs']) == len(metrics['spectral_radius'])
    )
    if has_hist:
        epochs  = np.array(metrics['epochs'], dtype=float)
        rho_raw = np.array(metrics['spectral_radius'], dtype=float)
        w = max(1, len(epochs) // 30)
        rho_sm  = _smooth(rho_raw, w)
        ax2.plot(epochs, rho_sm, color='green', lw=1.5,
                 label='rho(A) training trajectory (smoothed)')
        ax2.scatter(epochs[-1], rho_final, s=80, color='red', zorder=4,
                    label=f'Final rho = {rho_final:.4f}')
        ax2.set_xlabel("M-step epoch (global)", fontsize=9)
        ax2.set_title("rho(A) training trajectory", fontsize=10)
    else:
        ax2.bar(['Final rho(A)'], [rho_final], color='steelblue', width=0.4)
        ax2.text(0, rho_final + 0.005, f'{rho_final:.4f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax2.set_xlabel("(No per-epoch history logged)", fontsize=9)
        ax2.set_title("Final spectral radius", fontsize=10)
        ax2.set_ylim(0, max(1.1, rho_final * 1.15))

    ax2.axhline(1.0,  color='orange', linestyle=':', lw=1.0,
                label='rho = 1.0 (marginal stability)')
    ax2.axhline(0.92, color='red', linestyle='--', lw=1.2,
                label='rho_max = 0.92 (enforced ceiling)')
    ax2.set_ylabel("Spectral radius rho(A)  =  max|eigenvalue|", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    fname_parts = [dataset_name, "spectral_radius"]
    if suffix:
        fname_parts.append(suffix)
    save_path = f"{out_path}/{'_'.join(fname_parts)}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Spectral radius plot saved to: {save_path}")


def run_saved_model_plots(dataset, data_path, model_path, out_path):
    # 1. Path Configuration
    os.makedirs(out_path, exist_ok=True)

    print(f"Loading saved model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['hyperparameters']
    print(f"Checkpoint hyperparameters: {config}")
    
    # 2. Load Data for Inference
    data_load = np.load(f"{data_path}/{dataset}.spikes.npz")
    truth_load = np.load(f"{data_path}/{dataset}.prismTruth.npz")
    
    A_true = truth_load['A_true']
    N = A_true.shape[0]
    Y_np = data_load['spikes'].reshape(-1, N).astype(np.float32)
    B_true = truth_load['B_true']
    z_true = truth_load['S_true'] if 'S_true' in truth_load else None
    
    n_lags = int(config.get('n_lags', 1))
    lags = []
    for l in range(1, n_lags + 1):
        lag = np.roll(Y_np, l, axis=0).astype(np.float32)
        lag[:l, :] = 0.0
        lags.append(lag)
    X_np = np.concatenate(lags, axis=1)
    Y_full = torch.tensor(Y_np, device=device)
    X_full = torch.tensor(X_np, device=device)

    # 3. Reconstruct Model
    n_states = int(config.get('n_states', 2))
    model = PrismNeuralGLMHMM(n_states=n_states, n_neurons=N, n_lags=n_lags).to(device)
    
    # Load weights (unwrapping DataParallel if needed)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. Final E-Step (Inference)
    print("Running final inference for latent states...")
    with torch.no_grad():
        gamma, _ = compute_expectations(
            model,
            Y_full,
            X_full,
            delta_t=float(config.get('delta_t', 1.0)),
            gamma_temp=1.0,
            gamma_floor=0.0,
        )
        z_hat = gamma.argmax(dim=1).cpu().numpy()
        gamma_np = gamma.cpu().numpy()

    # 5. Connectivity and Training Monitor (Slide 6 & 7)
    print("Generating connectivity and monitor plots...")
    plot_connectivity_validation(
        model,
        A_true,
        B_true,
        dataset,
        out_path,
        suffix="inference",
        metrics=checkpoint.get('metrics', {}),
    )
    plot_training_monitor(checkpoint['metrics'], dataset, out_path, suffix="inference")

    # 5b. Spectral radius monitor
    print("Generating spectral radius plot...")
    plot_spectral_radius(model, dataset, out_path, suffix="inference",
                         metrics=checkpoint.get('metrics', {}))

    # 6. Jan-style state prediction plot (no toolbox dependency)
    print("Generating Jan-style state prediction plot...")
    plot_state_prediction(gamma_np, z_true, dataset, out_path, time_step=0.1, max_time_s=None)

    # 7. Ground truth vs recovered A side-by-side
    print("Generating ground truth vs recovered A plot...")
    A_hat_np = model.A.detach().cpu().numpy()[:, :N]
    plot_ground_truth_vs_recovered(A_true, A_hat_np, dataset, out_path, suffix="inference")

    # 8. Residuals diagnostics
    print("Generating residuals plot...")
    plot_residuals(model, A_true, B_true, dataset, out_path,
                   suffix="inference", metrics=checkpoint.get('metrics', {}))

    print("Success. All plots saved to output directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PRISM plots from a saved checkpoint.")
    parser.add_argument("--dataset", type=str, default="daleN100_b1036a", help="Dataset stem (without extensions).")
    parser.add_argument(
        "--data-path",
        type=str,
        default="/pscratch/sd/s/sanjitr/causal_net_temp/spikesData",
        help="Directory containing .spikes.npz and .prismTruth.npz files.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["ns7", "ns8", "ns9", "ns10", "auto"],
        default="auto",
        help="Model variant for default path resolution. Use 'auto' to pick the newest available checkpoint.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, derived from --model-root, --dataset, and --variant.",
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default="/pscratch/sd/s/sanjitr/causal_net_temp/models",
        help="Root directory for checkpoints when --model-path is not set.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Directory to save generated plots. If omitted, uses plots/ns7_plots, plots/ns8_plots, or plots/ns9_plots by variant.",
    )
    args = parser.parse_args()

    model_path = args.model_path
    resolved_variant = args.variant
    if model_path is None:
        ns7_path = f"{args.model_root}/{args.dataset}_best_model.pt"
        ns8_path = f"{args.model_root}/{args.dataset}_best_model_ns8.pt"
        ns9_path = f"{args.model_root}/{args.dataset}_best_model_ns9.pt"
        ns10_path = f"{args.model_root}/{args.dataset}_best_model_ns10.pt"

        if args.variant == "ns7":
            model_path = ns7_path
            resolved_variant = "ns7"
        elif args.variant == "ns8":
            model_path = ns8_path
            resolved_variant = "ns8"
        elif args.variant == "ns9":
            model_path = ns9_path
            resolved_variant = "ns9"
        elif args.variant == "ns10":
            model_path = ns10_path
            resolved_variant = "ns10"
        else:
            # Auto: pick newest existing checkpoint among ns7, ns8, ns9.
            candidates = []
            if os.path.exists(ns7_path):
                candidates.append(("ns7", ns7_path, os.path.getmtime(ns7_path)))
            if os.path.exists(ns8_path):
                candidates.append(("ns8", ns8_path, os.path.getmtime(ns8_path)))
            if os.path.exists(ns9_path):
                candidates.append(("ns9", ns9_path, os.path.getmtime(ns9_path)))
            if os.path.exists(ns10_path):
                candidates.append(("ns10", ns10_path, os.path.getmtime(ns9_path)))
            if not candidates:
                raise FileNotFoundError(
                    f"No checkpoint found. Tried:\n  {ns7_path}\n  {ns8_path}\n  {ns9_path}\n {ns10_path}"
                )
            candidates.sort(key=lambda x: x[2], reverse=True)
            resolved_variant, model_path, _ = candidates[0]
    else:
        # Infer variant from explicit path for output-folder defaulting.
        if model_path.endswith("_best_model_ns10.pt"):
            resolved_variant = "ns10"
        if model_path.endswith("_best_model_ns9.pt"):
            resolved_variant = "ns9"
        elif model_path.endswith("_best_model_ns8.pt"):
            resolved_variant = "ns8"
        elif model_path.endswith("_best_model.pt"):
            resolved_variant = "ns7"
        elif resolved_variant == "auto":
            resolved_variant = "ns7"

    out_path = args.out_path
    if out_path is None:
        if resolved_variant == "ns10":
            out_path = "/pscratch/sd/s/sanjitr/causal_net_temp/plots/ns10_plots"
        elif resolved_variant == "ns9":
            out_path = "/pscratch/sd/s/sanjitr/causal_net_temp/plots/ns9_plots"
        elif resolved_variant == "ns8":
            out_path = "/pscratch/sd/s/sanjitr/causal_net_temp/plots/ns8_plots"
        else:
            out_path = "/pscratch/sd/s/sanjitr/causal_net_temp/plots/ns7_plots"

    print(f"Resolved variant: {resolved_variant}")
    print(f"Resolved checkpoint: {model_path}")
    print(f"Resolved output path: {out_path}")

    run_saved_model_plots(
        dataset=args.dataset,
        data_path=args.data_path,
        model_path=model_path,
        out_path=out_path,
    )