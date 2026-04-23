import torch
import numpy as np
import os
import argparse
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr, linregress
from vb_sssm_1 import VB_SSSM, variational_e_step, tridiagonal_precision, load_checkpoint

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Shared helpers
# -------------------------

def _active_mask(model, threshold=1e-5):
    """ARD criterion from paper §4: prune if <z_m^n> < 1e-5 for all m."""
    return (model.gamma_nm.cpu().numpy().max(axis=1) > threshold)

def _dominant_state(model):
    """ñ_m = argmax_n <z_m^n> — the most likely label at each coarse bin."""
    return np.argmax(model.gamma_nm.cpu().numpy(), axis=0)  # shape (M,)

def _logit_to_hz(x, delta_ms=1.0):
    """
    Invert the logit transform (paper §2.1): λ_m = sigmoid(2·x_m) / Δ.
    delta_ms : bin width in ms (Δ in the paper).  Returns rate in Hz (spikes/s).
    """
    return (1.0 / (1.0 + np.exp(-2.0 * x))) / (delta_ms * 1e-3)

def _dominant_firing_rate_hz(model, delta_ms=1.0):
    """x̃_m = <x_m^{ñ_m}> converted to Hz (paper §4, 'estimated firing rate')."""
    mu   = model.mu_hat.cpu().numpy()   # (N, M)
    best = _dominant_state(model)       # (M,)
    x_dominant = mu[best, np.arange(model.M)]
    return _logit_to_hz(x_dominant, delta_ms)

# -------------------------
# 1. Firing rate trajectories (FIXED: logit → Hz, uses mu_hat)
# -------------------------

def plot_firing_rate_trajectories(model, C=10, delta_ms=1.0, true_rates_hz=None,
                                  dataset_name="", out_path="."):
    """
    Per-label firing rate trajectories in Hz (paper Fig. 3c).
    Converts logit-domain mu_hat to actual rates via λ_m = sigmoid(2·x_m) / Δ.
    Also plots the dominant-state composite estimate x̃_m (paper Fig. 3a).
    """
    mu        = model.mu_hat.cpu().numpy()       # (N, M)
    gamma     = model.gamma_nm.cpu().numpy()     # (N, M)
    active    = np.where(_active_mask(model))[0]
    M         = model.M
    bins      = np.arange(M) * C * delta_ms      # time axis in ms

    fig, axes = plt.subplots(len(active) + 1, 1,
                             figsize=(14, 3 * (len(active) + 1)), sharex=True)
    fig.suptitle(f"VB-SSSM Firing Rate Trajectories — {dataset_name}", fontsize=13)

    # Top panel: composite dominant-state estimate (paper Fig. 3a)
    ax0 = axes[0]
    rate_dom = _dominant_firing_rate_hz(model, delta_ms)
    ax0.plot(bins, rate_dom, color='tab:red', lw=1.5, label="Dominant-state estimate x̃_m")
    if true_rates_hz is not None:
        ax0.plot(bins, true_rates_hz, 'k--', alpha=0.6, lw=1.2, label="True rate")
    ax0.set_ylabel("Firing rate (Hz)")
    ax0.set_title("Composite estimate")
    ax0.legend(fontsize=8)

    # One panel per active label (paper Fig. 3c)
    for i, n in enumerate(active):
        ax = axes[i + 1]
        rate_n = _logit_to_hz(mu[n], delta_ms)
        ax.plot(bins, rate_n, lw=1.5, label=f"Label {n}")
        ax.fill_between(bins, 0, gamma[n] * rate_n.max(), alpha=0.15, label="⟨z_m^n⟩ (scaled)")
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_title(f"Label {n}  (mean occupancy {gamma[n].mean():.3f})")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(f"{out_path}/{dataset_name}_rate_trajectories.png", dpi=150)
    plt.close()

# -------------------------
# 2. State occupancy heatmap + transition matrix
# -------------------------

def plot_state_dynamics_monitor(model, dataset_name, out_path):
    """
    Soft state responsibilities ⟨z_m^n⟩ and inferred transition matrix (paper Fig. 2b/d).
    """
    gamma = model.gamma_nm.cpu().numpy()

    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    ax1 = fig.add_subplot(gs[0])
    im  = ax1.imshow(gamma, aspect='auto', cmap='magma', interpolation='none')
    ax1.set_title(r"Posterior state responsibilities $\langle z_m^n \rangle$")
    ax1.set_xlabel("Time (coarse bins)")
    ax1.set_ylabel("Label index")
    plt.colorbar(im, ax=ax1, label="Responsibility")

    ax2  = fig.add_subplot(gs[1])
    A_eff = np.exp(model.get_log_A().cpu().numpy())
    im2  = ax2.imshow(A_eff, cmap='Blues', vmin=0, vmax=1)
    ax2.set_title("Estimated transition matrix")
    ax2.set_xlabel("State t+1")
    ax2.set_ylabel("State t")
    for i in range(A_eff.shape[0]):
        for j in range(A_eff.shape[1]):
            ax2.text(j, i, f"{A_eff[i,j]:.2f}", ha='center', va='center', fontsize=7)
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(f"{out_path}/{dataset_name}_state_monitor.png", dpi=150)
    plt.close()

# -------------------------
# 3. Temporal correlation diagnostics (existing, kept)
# -------------------------

def plot_correlation_diagnostics(model, dataset_name, out_path):
    """β^n precision profile — higher β means smoother (more correlated) firing rates."""
    betas  = model.beta_n.cpu().numpy()
    active = _active_mask(model)

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(betas)), betas,
            color=['tab:green' if a else 'tab:gray' for a in active])
    plt.axhline(1.0, color='k', linestyle='--', alpha=0.5, label="β=1 baseline")
    plt.title(f"Temporal correlation precision β^n — {dataset_name}")
    plt.xlabel("Label index")
    plt.ylabel("β^n")
    plt.xticks(range(len(betas)))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_path}/{dataset_name}_correlation_profile.png", dpi=150)
    plt.close()

# -------------------------
# 4. Hard state sequence + spike raster  [NEW]
# -------------------------

def plot_state_sequence_with_spikes(model, Y_np, C=10, delta_ms=1.0,
                                    dataset_name="", out_path="."):
    """
    Two-panel figure (paper Fig. 2b/d style):
      Top:    colour-coded hard state assignment ñ_m = argmax_n ⟨z_m^n⟩ over time.
      Bottom: spike raster (one row per neuron, or population PSTH for many neurons).
    """
    M         = model.M
    best      = _dominant_state(model)          # (M,)
    active    = np.where(_active_mask(model))[0]
    N_active  = len(active)
    cmap      = plt.cm.get_cmap('tab10', model.N)
    bins      = np.arange(M) * C * delta_ms     # ms

    fig, (ax_state, ax_spikes) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True,
        gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle(f"State sequence & spikes — {dataset_name}", fontsize=13)

    # State colour bar
    for m in range(M - 1):
        ax_state.axvspan(bins[m], bins[m + 1],
                         color=cmap(best[m]), alpha=0.8)
    from matplotlib.patches import Patch
    handles = [Patch(color=cmap(n), label=f"Label {n}") for n in active]
    ax_state.legend(handles=handles, loc='upper right', fontsize=8, ncol=N_active)
    ax_state.set_yticks([])
    ax_state.set_ylabel("State")

    # Spike raster / PSTH
    T_used = M * C
    Y_clip = Y_np[:T_used]                     # (T_used, N_neurons)
    if Y_clip.shape[1] <= 30:
        # Raster for small neuron counts
        spike_times, neuron_ids = np.where(Y_clip > 0)
        t_ms = spike_times * delta_ms
        ax_spikes.scatter(t_ms, neuron_ids, s=0.5, c='black', alpha=0.4)
        ax_spikes.set_ylabel("Neuron")
    else:
        # Population PSTH
        psth = Y_clip.sum(axis=1)              # (T_used,)
        t_fine = np.arange(T_used) * delta_ms
        ax_spikes.plot(t_fine, psth, lw=0.6, color='black')
        ax_spikes.set_ylabel("Pop. spike count")

    ax_spikes.set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(f"{out_path}/{dataset_name}_state_sequence.png", dpi=150)
    plt.close()

# -------------------------
# 5. Change point detection  [NEW]
# -------------------------

def plot_change_points(model, C=10, delta_ms=1.0, true_change_pts_ms=None,
                       dataset_name="", out_path="."):
    """
    Overlays estimated change points (bins where ñ_m ≠ ñ_{m+1}) on the dominant
    firing rate in Hz (paper §4.1.1).  Optionally marks true change points.
    """
    best        = _dominant_state(model)
    rate_dom    = _dominant_firing_rate_hz(model, delta_ms)
    bins        = np.arange(model.M) * C * delta_ms

    # Change points: bins where dominant label switches
    cp_bins = np.where(np.diff(best) != 0)[0]          # coarse bin index
    cp_ms   = (cp_bins + 0.5) * C * delta_ms           # midpoint in ms

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(bins, rate_dom, color='tab:red', lw=1.5, label="Estimated rate")
    for t in cp_ms:
        ax.axvline(t, color='tab:blue', lw=1.2, linestyle='--', alpha=0.8)
    if true_change_pts_ms is not None:
        for t in true_change_pts_ms:
            ax.axvline(t, color='black', lw=1.2, linestyle=':', alpha=0.7)
        ax.plot([], [], 'k:', label="True change points")
    ax.plot([], [], color='tab:blue', linestyle='--', label=f"Estimated CPs (n={len(cp_ms)})")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(f"Change point detection — {dataset_name}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_path}/{dataset_name}_change_points.png", dpi=150)
    plt.close()

    return cp_ms  # return for downstream evaluation

# -------------------------
# 6. K-S goodness-of-fit  [NEW]
# -------------------------

def plot_ks_goodness_of_fit(model, Y_np, C=10, delta_ms=1.0,
                             dataset_name="", out_path="."):
    """
    Time-rescaling K-S test (Brown et al. 2002, paper Fig. 4a inset).

    Under a correct model, the rescaled ISI integrals
        τ_i = ∫_{t_{i-1}}^{t_i} λ(t) dt
    are i.i.d. Exp(1), so u_i = 1 − exp(−τ_i) ~ Uniform(0,1).
    Sorted u_i plotted vs. i/S should lie on the diagonal ± 1.36/√S bounds.

    Uses the dominant-state firing rate x̃_m at coarse-bin resolution.
    """
    M      = model.M
    T_used = M * C
    Y_clip = Y_np[:T_used].sum(axis=1)          # (T_used,) binary, summed over neurons

    # Dominant-state rate per fine bin (constant within each coarse bin)
    mu_dom  = model.mu_hat.cpu().numpy()[_dominant_state(model), np.arange(M)]
    # λ_k in spikes/ms for each fine bin k
    rate_fine = np.repeat(1.0 / (1.0 + np.exp(-2.0 * mu_dom)), C) / (delta_ms * 1e-3) * 1e-3
    # rate_fine in spikes/ms (probability per fine bin ≈ λ·Δ)
    prob_fine = np.clip(rate_fine * delta_ms * 1e-3, 0, 1 - 1e-8)

    # Rescale time between successive spikes
    spike_bins = np.where(Y_clip > 0)[0]
    if len(spike_bins) < 5:
        print(f"[KS] Too few spikes ({len(spike_bins)}) to plot KS.")
        return

    rescaled = []
    prev = 0
    for k in spike_bins:
        tau_i = prob_fine[prev:k + 1].sum()
        rescaled.append(1.0 - np.exp(-tau_i))
        prev = k + 1
    u = np.sort(np.array(rescaled))
    S = len(u)
    expected = np.arange(1, S + 1) / S
    bound    = 1.36 / np.sqrt(S)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(expected, u, 'b-', lw=1.2, label="K-S plot")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.fill_between([0, 1], [0 - bound, 1 - bound], [0 + bound, 1 + bound],
                    alpha=0.15, color='gray', label="95% CI")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Uniform quantile")
    ax.set_ylabel("Empirical CDF of rescaled ISIs")
    ax.set_title(f"K-S goodness-of-fit — {dataset_name}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_path}/{dataset_name}_ks_plot.png", dpi=150)
    plt.close()

# -------------------------
# 7. State parameter comparison (β and μ̄)  [NEW]
# -------------------------

def plot_state_parameter_comparison(model, C=10, delta_ms=1.0,
                                    dataset_name="", out_path="."):
    """
    Side-by-side bar plots of β^n (temporal correlation precision) and mean firing
    rate μ̄^n for each active label (paper Fig. 4d/e).

    Higher β → smoother/more correlated trajectory within that state.
    Different β with similar μ̄ → states distinguished by temporal structure,
    not mean rate (the key finding in the paper's real-data analysis).
    """
    active  = np.where(_active_mask(model))[0]
    betas   = model.beta_n.cpu().numpy()[active]
    mu_hat  = model.mu_hat.cpu().numpy()        # (N, M)
    gamma   = model.gamma_nm.cpu().numpy()      # (N, M)

    # Sojourn-weighted mean firing rate per label (paper §4.2 definition of ⟨d_μ⟩)
    # T_n = total time bins assigned to label n
    mu_bar = np.array([
        _logit_to_hz(mu_hat[n], delta_ms)[gamma[n] > 1e-5].mean()
        if (gamma[n] > 1e-5).any() else 0.0
        for n in active
    ])

    labels = [f"Label {n}" for n in active]
    x      = np.arange(len(active))

    fig, (ax_beta, ax_mu) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Per-state parameters — {dataset_name}", fontsize=12)

    ax_beta.bar(x, betas, color='tab:blue')
    ax_beta.set_xticks(x); ax_beta.set_xticklabels(labels)
    ax_beta.set_ylabel("β^n  (temporal correlation precision)")
    ax_beta.set_title("Temporal correlation")

    ax_mu.bar(x, mu_bar, color='tab:orange')
    ax_mu.set_xticks(x); ax_mu.set_xticklabels(labels)
    ax_mu.set_ylabel("Mean firing rate (Hz)")
    ax_mu.set_title("Mean firing rate μ̄^n")

    plt.tight_layout()
    plt.savefig(f"{out_path}/{dataset_name}_state_parameters.png", dpi=150)
    plt.close()

# =========================================================================
# Connectivity diagnostics  (migrated from PlotterAshwood.py)
# These functions accept A_hat / B_hat / A_true / B_true as numpy arrays
# and have no dependency on PrismNeuralGLMHMM or any ashwood_ns module.
# =========================================================================

def _find_best_tau(A_hat_block, A_true_block):
    """Threshold that maximises F1 on absolute-value edge detection."""
    true_nz = np.abs(A_true_block) > 1e-6
    candidates = np.percentile(np.abs(A_hat_block), np.arange(10, 95, 5))
    best_f1, best_tau = -1.0, float(np.median(np.abs(A_hat_block)))
    for tau in candidates:
        hat_nz = np.abs(A_hat_block) > tau
        tp = int((true_nz & hat_nz).sum())
        fp = int((~true_nz & hat_nz).sum())
        fn = int((true_nz & ~hat_nz).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return best_tau


def _block_metrics(A_eval, A_true, tau_exc, tau_inh, n_exc):
    """Abs and sign-aware TP/FP/FN metrics, split by excitatory/inhibitory block."""
    N = A_true.shape[0]
    true_mask = np.zeros((N, N), dtype=bool)
    hat_mask  = np.zeros((N, N), dtype=bool)
    true_mask[:n_exc, :] = np.abs(A_true[:n_exc, :]) > tau_exc
    true_mask[n_exc:, :] = np.abs(A_true[n_exc:, :]) > tau_inh
    hat_mask[:n_exc, :]  = np.abs(A_eval[:n_exc, :]) > tau_exc
    hat_mask[n_exc:, :]  = np.abs(A_eval[n_exc:, :]) > tau_inh

    def prf(t, h):
        tp = int((t & h).sum())
        fp = int((~t & h).sum())
        fn = int((t & ~h).sum())
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return tp, fp, fn, p, r, f

    tp, fp, fn, prec, rec, f1 = prf(true_mask, hat_mask)
    tn = int((~true_mask & ~hat_mask).sum())
    tp_e, fp_e, fn_e, p_e, r_e, f_e = prf(true_mask[:n_exc, :], hat_mask[:n_exc, :])
    tp_i, fp_i, fn_i, p_i, r_i, f_i = prf(true_mask[n_exc:, :], hat_mask[n_exc:, :])
    tp_es, fp_es, fn_es, p_es, r_es, f_es = prf(
        A_true[:n_exc, :] > tau_exc,  A_eval[:n_exc, :] > tau_exc)
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
    cax  = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar.ax.set_yticklabels(['TN', 'FN', 'FP', 'TP'])


def _draw_stats_panel(ax, m, tau_exc, tau_inh, label=""):
    tp, fp, fn = m['tp'], m['fp'], m['fn']
    bars = ax.bar(
        ['True Positive\n(TP)', 'False Positive\n(FP)', 'False Negative\n(FN)'],
        [tp, fp, fn],
        color=['#38761d', '#ed2f20', '#df20df'],
    )
    title = f"Edge classification counts ({label})" if label else "Edge classification counts"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Prediction outcome\n(TP: correct edge, FP: spurious, FN: missed)", fontsize=8)
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
        f"Excitatory (abs):  p={m['p_e']:.3f}  r={m['r_e']:.3f}  F1={m['f_e']:.3f}\n"
        f"Inhibitory (abs):  p={m['p_i']:.3f}  r={m['r_i']:.3f}  F1={m['f_i']:.3f}\n"
        f"Excitatory (sign): p={m['p_es']:.3f}  r={m['r_es']:.3f}  F1={m['f_es']:.3f}\n"
        f"Inhibitory (sign): p={m['p_is']:.3f}  r={m['r_is']:.3f}  F1={m['f_is']:.3f}"
    )
    ax.text(0.02, 0.98, stats_text, fontsize=9, verticalalignment='top',
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.text(0.02, 0.52, block_text, fontsize=8, verticalalignment='top',
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8))


def _best_state_permutation_by_correlation(B_true, B_hat):
    """Resolve label switching by maximising the sum of per-state |Pearson r|."""
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
    ax.set_xlim(b_min, b_max)
    ax.set_ylim(b_min, b_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Log-baseline bias correlation, state {state_idx} (R={r_val:.3f})")
    ax.set_xlabel(f"True log-baseline $B_{{true}}$ (state {state_idx})")
    ax.set_ylabel(f"Inferred log-baseline $\\hat{{B}}$ (state {state_idx})")


def _smooth(x, w=20):
    """Rolling mean with window w; pads so output length matches input."""
    x = np.array(x, dtype=float)
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    padded = np.concatenate([np.full(w - 1, x[0]), x])
    return np.convolve(padded, kernel, mode='valid')


def _encode_labels_zero_based(z, K):
    z = np.asarray(z).astype(int)
    uniq = np.unique(z)
    if uniq.size != K:
        return z
    mapping = {lab: i for i, lab in enumerate(sorted(uniq.tolist()))}
    return np.array([mapping[v] for v in z], dtype=int)


def _align_states_to_truth(gamma, z_true):
    """
    Resolve label-switching by permuting inferred states to maximise accuracy.
    Returns aligned_gamma, z_true_enc, z_hat_aligned, best_acc, per_state_acc.
    """
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
    gamma_aligned = gamma[:, best_perm]
    z_hat_aligned = best_perm[z_hat]
    per_state_acc = [
        np.mean(z_hat_aligned[z_true_enc == k] == k) if np.any(z_true_enc == k) else np.nan
        for k in range(K)
    ]
    return gamma_aligned, z_true_enc, z_hat_aligned, best_acc, per_state_acc


def _reorder_inh_top(A, n_exc):
    """Reorder rows/cols so the inhibitory block (rows n_exc:) appears on top."""
    idx = list(range(n_exc, A.shape[0])) + list(range(n_exc))
    return A[np.ix_(idx, idx)]


# -------------------------
# 8. Residuals  [migrated]
# -------------------------

def plot_residuals(A_hat, B_hat, A_true, B_true, dataset_name, out_path,
                   suffix="", n_exc=None):
    """
    6-panel residual diagnostics.  All inputs are numpy arrays.

    A_hat, A_true : (N, N)  lag-1 connectivity matrices
    B_hat, B_true : (K, N)  per-state log-baseline bias matrices
    n_exc         : number of excitatory neurons (default: 80 % of N)
    """
    N = A_true.shape[0]
    if n_exc is None:
        n_exc = int(0.8 * N)
    mask_off = ~np.eye(N, dtype=bool)
    thresh   = 1e-6

    inh_true_flat = A_true[n_exc:][mask_off[n_exc:]]
    inh_hat_flat  = A_hat[n_exc:][mask_off[n_exc:]]
    inh_nz    = np.abs(inh_true_flat) > thresh
    inh_resid = inh_hat_flat[inh_nz] - inh_true_flat[inh_nz]

    exc_true_flat = A_true[:n_exc][mask_off[:n_exc]]
    exc_hat_flat  = A_hat[:n_exc][mask_off[:n_exc]]
    exc_nz    = np.abs(exc_true_flat) > thresh
    exc_resid = exc_hat_flat[exc_nz] - exc_true_flat[exc_nz]

    diag_idx   = np.arange(N)
    diag_resid = A_hat[diag_idx, diag_idx] - A_true[diag_idx, diag_idx]

    tau_exc = _find_best_tau(A_hat[:n_exc], A_true[:n_exc])
    tau_inh = _find_best_tau(A_hat[n_exc:], A_true[n_exc:])
    m = _block_metrics(A_hat, A_true, tau_exc, tau_inh, n_exc)

    best_b_perm   = _best_state_permutation_by_correlation(B_true, B_hat)
    K_b           = B_true.shape[0]
    B_hat_aligned = np.array([B_hat[best_b_perm[k]] for k in range(K_b)])
    b_resid_all   = (B_hat_aligned - B_true).ravel()
    b_true_flat   = B_true.ravel()
    b_split       = np.median(b_true_flat)
    b_resid_low   = b_resid_all[b_true_flat <  b_split]
    b_resid_high  = b_resid_all[b_true_flat >= b_split]

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    fig.suptitle(f"Edge & Weight Residuals: {dataset_name}", fontsize=13)

    def _hist_panel(ax, data, title, color):
        if data.size == 0:
            ax.set_title(title, fontsize=9)
            ax.text(0.5, 0.5, "no data", ha='center', va='center', transform=ax.transAxes)
            return
        med, std = np.median(data), np.std(data)
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
        ax.set_xlabel("Residual (fit−true)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)

    _hist_panel(axes[0], inh_resid,  "A_off (Inhib) resid", color='steelblue')
    _hist_panel(axes[1], exc_resid,  "A_off (Excit) resid", color='salmon')
    _hist_panel(axes[2], diag_resid, "A_diag residual",     color='lightsalmon')

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

    _hist_panel(axes[4], b_resid_low,  f"B (Low < {b_split:.2f}) resid",  color='steelblue')
    _hist_panel(axes[5], b_resid_high, f"B (High ≥ {b_split:.2f}) resid", color='salmon')

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    parts = [dataset_name, "residuals"] + ([suffix] if suffix else [])
    save_path = f"{out_path}/{'_'.join(parts)}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Residuals plot saved to: {save_path}")


# -------------------------
# 9. Connectivity validation  [migrated]
# -------------------------

def plot_connectivity_validation(A_hat, B_hat, A_true, B_true,
                                 dataset_name, out_path, suffix="", n_exc=None):
    """
    3-panel top row (TP/FP/FN stats, confusion map, weight scatter) plus one
    bias-correlation panel per state.  All inputs are numpy arrays.
    """
    N = A_true.shape[0]
    if n_exc is None:
        n_exc = int(0.8 * N)

    tau_exc = _find_best_tau(A_hat[:n_exc, :], A_true[:n_exc, :])
    tau_inh = _find_best_tau(A_hat[n_exc:, :], A_true[n_exc:, :])
    m = _block_metrics(A_hat, A_true, tau_exc, tau_inh, n_exc)

    conf_img = np.zeros_like(A_hat)
    conf_img[m['true_mask'] & m['hat_mask']]   = 3   # TP
    conf_img[~m['true_mask'] & m['hat_mask']]  = 2   # FP
    conf_img[m['true_mask'] & ~m['hat_mask']]  = 1   # FN
    cmap_conf = mcolors.ListedColormap(['white', '#df20df', '#ed2f20', '#38761d'])

    K_b = B_true.shape[0]
    n_bias_cols = min(3, max(1, K_b))
    n_bias_rows = int(np.ceil(K_b / n_bias_cols))
    fig = plt.figure(figsize=(18, 5 + 4 * n_bias_rows))
    gs  = fig.add_gridspec(1 + n_bias_rows, 3,
                           height_ratios=[1.0] + [1.0] * n_bias_rows)
    axes_top = [fig.add_subplot(gs[0, col]) for col in range(3)]
    plt.suptitle(f"Connectivity Validation: {dataset_name}", fontsize=16)

    tick_positions = list(range(0, N + 1, 20))
    _draw_stats_panel(axes_top[0], m, tau_exc, tau_inh)
    _draw_confusion_panel(axes_top[1], fig, conf_img, cmap_conf,
                          f"Edge confusion map  (F1={m['f1']:.3f})")
    axes_top[1].set_xticks(tick_positions)
    axes_top[1].set_yticks(tick_positions)
    axes_top[1].invert_yaxis()

    mask_off = ~np.eye(N, dtype=bool)
    slope, intercept, r_val, _, _ = linregress(A_true[mask_off], A_hat[mask_off])
    x_line = np.array([-0.4, 0.4])
    axes_top[2].scatter(A_true[mask_off], A_hat[mask_off], alpha=0.3, s=5)
    axes_top[2].plot(x_line, x_line, 'r--', label='slope=1')
    axes_top[2].plot(x_line, slope * x_line + intercept, 'b-', linewidth=1.2,
                     label=f'fit slope={slope:.3f}')
    axes_top[2].legend(fontsize=8)
    axes_top[2].set_title(f"Weight correlation  R={r_val:.3f}  slope={slope:.3f}")
    axes_top[2].set_xlabel("True weight $A_{true}$ (off-diagonal)")
    axes_top[2].set_ylabel("Inferred weight $\\hat{A}$ (off-diagonal)")

    best_b_perm = _best_state_permutation_by_correlation(B_true, B_hat)
    b_colors    = ['blue', 'red', 'green', 'purple', 'orange']
    bias_axes   = [fig.add_subplot(gs[row + 1, col])
                   for row in range(n_bias_rows) for col in range(3)]
    for k in range(K_b):
        _plot_bias_correlation_panel(bias_axes[k], B_true[k], B_hat[best_b_perm[k]],
                                     state_idx=k, color=b_colors[k % len(b_colors)])
    for ax in bias_axes[K_b:]:
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(f"{out_path}/{dataset_name}_connectivity_validation_{suffix}.png", dpi=150)
    plt.close(fig)
    print(f"Connectivity validation plot saved to: {out_path}")


# -------------------------
# 10. Training monitor  [migrated]
# -------------------------

def plot_training_monitor(metrics, dataset_name, out_path, suffix=""):
    """4-panel training monitor: E-step NLL, spectral radius, edge support, M-step NLL+L1."""
    epochs = np.array(metrics['epochs'])
    w = max(1, len(epochs) // 30)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    plt.suptitle(f"Training Monitor: {dataset_name} {suffix}", fontsize=14)

    axes[0].plot(metrics['em_iters'], metrics['nll_bin'], marker='o', color='tab:blue')
    axes[0].set_title("E-step: neg. log-likelihood per bin")
    axes[0].set_xlabel("EM iteration")
    axes[0].set_ylabel("NLL / bin")

    rho_raw = np.array(metrics.get('spectral_radius', []), dtype=float)
    if rho_raw.size > 0 and rho_raw.size == epochs.size:
        rho_sm = _smooth(rho_raw, w)
        axes[1].plot(epochs, rho_sm, color='green', linewidth=1.8, label='ρ(A) smoothed')
        axes[1].axhline(1.0,  color='orange', linestyle=':', linewidth=1.0, label='ρ=1.0')
        axes[1].axhline(0.92, color='red',    linestyle='--', linewidth=1.2, label='ρ_max=0.92')
        axes[1].set_ylim(bottom=0)
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "spectral_radius not logged",
                     ha='center', va='center', transform=axes[1].transAxes, fontsize=9)
    axes[1].set_title("Lag-1 spectral radius ρ(A)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ρ(A)")

    nz_raw = np.array(metrics['nz_edges'], dtype=float)
    nz_sm  = _smooth(nz_raw, w)
    axes[2].plot(epochs, nz_sm, color='purple', linewidth=1.8, label='nonzero edges')
    axes[2].set_title("Connectivity support")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Nonzero edges (|Aᵢⱼ| > 0.01)")
    axes[2].legend(fontsize=8)

    nll_raw = np.array(metrics['nll_bin_M'], dtype=float)
    l1_raw  = np.array(metrics['l1_norm'],   dtype=float)
    ax4r    = axes[3].twinx()
    axes[3].plot(epochs, _smooth(nll_raw, w), color='tab:blue', linewidth=1.8, label='NLL smoothed')
    ax4r.plot(epochs, _smooth(l1_raw, w), color='tab:red', linewidth=1.8,
              linestyle='--', label='L1 smoothed')
    axes[3].set_title("M-step NLL and L1 norm")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("NLL / bin", color='tab:blue')
    ax4r.set_ylabel("‖A‖₁ (off-diag)", color='tab:red')
    lns = axes[3].get_lines()[-1:] + ax4r.get_lines()[-1:]
    axes[3].legend(lns, [l.get_label() for l in lns], loc='upper right', fontsize=8)

    p2 = metrics.get('phase2_start_epoch')
    if p2 is not None:
        for ax in axes[1:]:
            ax.axvline(p2, color='black', linestyle=':', alpha=0.6, label='Phase 2')

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(f"{out_path}/{dataset_name}_training_monitor_{suffix}.png", dpi=150)
    plt.close(fig)


# -------------------------
# 11. State prediction  [migrated]
# -------------------------

def plot_state_prediction(gamma, z_true, dataset_name, out_path,
                          time_step=0.1, max_time_s=1000.0):
    """
    Jan-style state prediction panel: mean occupancy, per-state accuracy,
    confidence histogram, fit trace vs. truth trace.

    gamma  : (T, K) numpy array of posterior state probabilities
    z_true : (T,) array of ground-truth labels, or None
    """
    T, K = gamma.shape
    t = np.arange(T) * time_step
    if max_time_s is None:
        max_time_s = 1000.0
    max_time_s = min(max_time_s, t[-1] if T > 1 else time_step)
    t_mask = t <= max_time_s

    if z_true is not None:
        gamma, z_true_enc, z_hat, acc, state_acc = _align_states_to_truth(gamma, z_true)
        z_true_plot = z_true_enc
    else:
        z_hat       = np.argmax(gamma, axis=1)
        acc         = np.nan
        state_acc   = [np.nan] * K
        z_true_plot = None
    conf = np.max(gamma, axis=1)

    fig = plt.figure(figsize=(30, 9), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 3, height_ratios=[1, 2.3], hspace=0.28, wspace=0.30)
    fig.suptitle(f"Dataset: {dataset_name}", fontsize=11)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(np.arange(K), np.mean(gamma, axis=0), color='steelblue', alpha=0.75)
    ax1.set_xticks(range(K))
    ax1.set_xticklabels([f"State {k}" for k in range(K)], fontsize=8)
    ax1.set_title("Mean occupancy", fontsize=9)
    ax1.set_xlabel("State", fontsize=8)
    ax1.set_ylabel("Mean posterior", fontsize=8)

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

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(conf, bins=40, color='mediumpurple', alpha=0.7)
    ttl = f"Confidence (acc={acc:.3f})" if not np.isnan(acc) else "Confidence"
    ax3.set_title(ttl, fontsize=9)
    ax3.set_xlabel("max posterior", fontsize=8)
    ax3.set_ylabel("count", fontsize=8)

    ax4 = fig.add_subplot(gs[1, :])
    if z_true_plot is not None:
        ax4.step(t[t_mask], z_true_plot[t_mask], where='post', color='black',
                 linewidth=1.6, label='True state')
    ax4.step(t[t_mask], z_hat[t_mask], where='post', color='tab:orange', alpha=0.85,
             linewidth=1.2, label='Inferred state (argmax posterior)')
    fit_ttl = f"State fit (acc={acc:.3f})" if not np.isnan(acc) else "State fit"
    ax4.set_title(fit_ttl, fontsize=9)
    ax4.set_xlabel("Time (s)", fontsize=8)
    ax4.set_ylabel("Latent state", fontsize=8)
    ax4.set_yticks(list(range(K)))
    ax4.set_yticklabels([f"State {k}" for k in range(K)], fontsize=8)
    ax4.legend(loc='upper right', fontsize=8)
    if np.any(t_mask):
        ax4.set_xlim(t[t_mask][0], t[t_mask][-1])

    os.makedirs(out_path, exist_ok=True)
    save_path = f"{out_path}/{dataset_name}_state_prediction_{int(max_time_s)}s.png"
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"State prediction plot saved to: {save_path}")


# -------------------------
# 12. Ground truth vs recovered  [migrated]
# -------------------------

def plot_ground_truth_vs_recovered(A_true, A_hat, dataset_name, out_path,
                                   suffix="", n_exc=None):
    """
    Side-by-side heatmaps of A_true and A_hat with F1 / R annotation.
    Both inputs are (N, N) numpy arrays.
    """
    N = A_true.shape[0]
    if n_exc is None:
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
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.invert_yaxis()
        ax.axhline(n_exc - 0.5, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
        ax.axvline(n_exc - 0.5, color='black', linewidth=1.0, linestyle='--', alpha=0.6)

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im, cax=cax, label="Synaptic weight")

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    parts = [dataset_name, "ground_truth_vs_recovered"] + ([suffix] if suffix else [])
    save_path = f"{out_path}/{'_'.join(parts)}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Ground truth vs recovered plot saved to: {save_path}")


# -------------------------
# 13. Spectral radius  [migrated]
# -------------------------

def plot_spectral_radius(A_hat_np, dataset_name, out_path, suffix="", metrics=None):
    """
    Eigenvalue spectrum of A_hat_np plus optional training trajectory.

    A_hat_np : (N, N) numpy array — lag-1 connectivity matrix
    metrics  : optional dict with keys 'epochs' and 'spectral_radius'
    """
    eigvals   = np.linalg.eigvals(A_hat_np)
    rho_final = float(np.max(np.abs(eigvals)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Spectral radius monitor: {dataset_name}\n"
        f"Final ρ(A) = {rho_final:.4f}  "
        f"({'STABLE' if rho_final < 1.0 else 'UNSTABLE — ρ ≥ 1'})",
        fontsize=11,
    )

    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, alpha=0.5, label='|z|=1')
    ax.plot(0.92 * np.cos(theta), 0.92 * np.sin(theta), 'r:', lw=0.8, alpha=0.6,
            label='|z|=0.92')
    re, im_part = eigvals.real, eigvals.imag
    ax.scatter(re, im_part, s=18, color='steelblue', alpha=0.75, zorder=3,
               label=f'Eigenvalues (N={len(eigvals)})')
    idx_max = np.argmax(np.abs(eigvals))
    ax.scatter(re[idx_max], im_part[idx_max], s=80, color='red', zorder=4,
               label=f'Max |λ| = {rho_final:.4f}')
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.4, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.4, alpha=0.5)
    ax.set_title(f"Eigenvalue spectrum\nρ(A) = {rho_final:.4f}", fontsize=10)
    ax.set_xlabel("Re(λ)", fontsize=9)
    ax.set_ylabel("Im(λ)", fontsize=9)
    ax.legend(fontsize=8, loc='upper left')

    ax2 = axes[1]
    has_hist = (metrics is not None
                and 'spectral_radius' in metrics
                and len(metrics.get('spectral_radius', [])) > 0
                and 'epochs' in metrics
                and len(metrics['epochs']) == len(metrics['spectral_radius']))
    if has_hist:
        epochs  = np.array(metrics['epochs'], dtype=float)
        rho_raw = np.array(metrics['spectral_radius'], dtype=float)
        rho_sm  = _smooth(rho_raw, max(1, len(epochs) // 30))
        ax2.plot(epochs, rho_sm, color='green', lw=1.5, label='ρ(A) smoothed')
        ax2.scatter(epochs[-1], rho_final, s=80, color='red', zorder=4,
                    label=f'Final ρ = {rho_final:.4f}')
        ax2.set_xlabel("Epoch", fontsize=9)
        ax2.set_title("ρ(A) training trajectory", fontsize=10)
    else:
        ax2.bar(['Final ρ(A)'], [rho_final], color='steelblue', width=0.4)
        ax2.text(0, rho_final + 0.005, f'{rho_final:.4f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax2.set_title("Final spectral radius", fontsize=10)
        ax2.set_ylim(0, max(1.1, rho_final * 1.15))
    ax2.axhline(1.0,  color='orange', linestyle=':', lw=1.0, label='ρ=1.0')
    ax2.axhline(0.92, color='red',    linestyle='--', lw=1.2, label='ρ_max=0.92')
    ax2.set_ylabel("ρ(A) = max|eigenvalue|", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    parts = [dataset_name, "spectral_radius"] + ([suffix] if suffix else [])
    save_path = f"{out_path}/{'_'.join(parts)}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Spectral radius plot saved to: {save_path}")


# -------------------------
# 14. VB-SSSM training monitor  [new — uses metrics logged by vb_sssm_1.py]
# -------------------------

def plot_vb_training_monitor(metrics, dataset_name, out_path):
    """
    3-panel training monitor for VB-SSSM using the metrics dict saved in the checkpoint.

    Panels:
      (0) Log-normaliser per bin (proxy ELBO likelihood term) — should rise then plateau.
      (1) Number of active states over training — tracks ARD pruning.
      (2) β^n mean and max over active states — should stay well below the cap (100).
    """
    iters    = np.array(metrics['iters'])
    log_norm = np.array(metrics['log_norm_per_bin'])
    active   = np.array(metrics['active_states'])
    b_mean   = np.array(metrics['beta_mean'])
    b_max    = np.array(metrics['beta_max'])
    tau      = np.array(metrics['temperature'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"VB-SSSM Training Monitor — {dataset_name}", fontsize=13)

    # Panel 0: log-normaliser (ELBO proxy)
    axes[0].plot(iters, log_norm, color='tab:blue', lw=1.5)
    # Mark end of annealing phase
    anneal_end = iters[tau >= 1.0][0] if np.any(tau >= 1.0) else iters[-1]
    axes[0].axvline(anneal_end, color='gray', linestyle='--', alpha=0.6,
                    label=f'Annealing ends (iter {anneal_end})')
    axes[0].set_title("Log-normaliser per bin (ELBO proxy)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("log p(y) / M  (higher = better)")
    axes[0].legend(fontsize=8)

    # Panel 1: active state count
    axes[1].step(iters, active, where='post', color='tab:green', lw=2.0)
    axes[1].set_title("Active states (ARD pruning)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Number of active states")
    axes[1].set_yticks(range(int(active.max()) + 2))
    axes[1].axvline(anneal_end, color='gray', linestyle='--', alpha=0.6)

    # Panel 2: β^n mean ± max across active states
    axes[2].plot(iters, b_mean, color='tab:orange', lw=1.5, label='β mean (active)')
    axes[2].plot(iters, b_max,  color='tab:red',    lw=1.2, linestyle='--',
                 label='β max (active)')
    axes[2].axhline(100.0, color='black', linestyle=':', lw=1.0, label='β cap = 100')
    axes[2].set_title("Temporal correlation precision β^n")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("β^n")
    axes[2].legend(fontsize=8)
    axes[2].axvline(anneal_end, color='gray', linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    save_path = f"{out_path}/{dataset_name}_vb_training_monitor.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"VB training monitor saved to: {save_path}")


# -------------------------
# Run all plots
# -------------------------

def run_sssm_plots(dataset, data_path, model_path, out_path,
                   C=10, delta_ms=1.0, true_change_pts_ms=None,
                   A_true=None, B_true=None, A_hat=None, B_hat=None,
                   z_true=None):
    os.makedirs(out_path, exist_ok=True)

    print(f"Loading VB-SSSM checkpoint: {model_path}")
    model, metrics = load_checkpoint(model_path, device)

    try:
        data_load = np.load(f"{data_path}/{dataset}.spikes.npz")
        Y_np = data_load["spikes"].astype(np.float32)
    except Exception:
        Y_np = None
        print("Warning: spike data not found — some plots will be skipped.")

    print("Firing rate trajectories...")
    plot_firing_rate_trajectories(model, C=C, delta_ms=delta_ms,
                                  dataset_name=dataset, out_path=out_path)

    print("State dynamics monitor...")
    plot_state_dynamics_monitor(model, dataset, out_path)

    print("Temporal correlation diagnostics...")
    plot_correlation_diagnostics(model, dataset, out_path)

    print("State sequence + spike raster...")
    if Y_np is not None:
        plot_state_sequence_with_spikes(model, Y_np, C=C, delta_ms=delta_ms,
                                        dataset_name=dataset, out_path=out_path)

    print("Change point overlay...")
    cp_ms = plot_change_points(model, C=C, delta_ms=delta_ms,
                               true_change_pts_ms=true_change_pts_ms,
                               dataset_name=dataset, out_path=out_path)
    print(f"  Detected change points (ms): {np.round(cp_ms, 1).tolist()}")

    print("K-S goodness-of-fit...")
    if Y_np is not None:
        plot_ks_goodness_of_fit(model, Y_np, C=C, delta_ms=delta_ms,
                                dataset_name=dataset, out_path=out_path)

    print("State parameter comparison (β, μ̄)...")
    plot_state_parameter_comparison(model, C=C, delta_ms=delta_ms,
                                    dataset_name=dataset, out_path=out_path)

    # --- Optional connectivity diagnostics (require ground-truth arrays) ---
    if A_true is not None and A_hat is not None:
        print("Ground truth vs recovered A...")
        plot_ground_truth_vs_recovered(A_true, A_hat, dataset, out_path)

        print("Connectivity validation...")
        B_hat_use  = B_hat  if B_hat  is not None else np.zeros((1, A_hat.shape[0]))
        B_true_use = B_true if B_true is not None else np.zeros((1, A_true.shape[0]))
        plot_connectivity_validation(A_hat, B_hat_use, A_true, B_true_use,
                                     dataset, out_path)

        print("Residuals diagnostics...")
        plot_residuals(A_hat, B_hat_use, A_true, B_true_use, dataset, out_path)

        print("Spectral radius...")
        plot_spectral_radius(A_hat, dataset, out_path)

    if z_true is not None:
        print("State prediction vs ground truth...")
        gamma_T = model.gamma_nm.cpu().numpy().T   # (M, K)
        plot_state_prediction(gamma_T, z_true, dataset, out_path,
                              time_step=C * delta_ms * 1e-3)

    if metrics and 'iters' in metrics:
        print("VB training monitor...")
        plot_vb_training_monitor(metrics, dataset, out_path)

    print(f"All diagnostics saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic suite for VB-SSSM models.")
    parser.add_argument("--dataset",    type=str, required=True)
    parser.add_argument("--data-path",  type=str,
                        default="/pscratch/sd/s/sanjitr/causal_net_temp/spikesData")
    parser.add_argument("--model-path", type=str,
                        default="/pscratch/sd/s/sanjitr/causal_net_temp/models")
    parser.add_argument("--out-path",   type=str,
                        default="/pscratch/sd/s/sanjitr/causal_net_temp/plots/vb_sssm")
    parser.add_argument("--C",          type=int,   default=10)
    parser.add_argument("--delta-ms",   type=float, default=1.0)
    parser.add_argument("--truth-path", type=str,   default=None,
                        help="Path to .SimTruth.npz / .prismTruth.npz for connectivity plots.")
    args = parser.parse_args()

    A_true = B_true = z_true = A_hat = None
    if args.truth_path is not None:
        truth = np.load(args.truth_path, allow_pickle=True)
        A_true = truth.get('A_true')
        B_true = truth.get('B_true')
        z_true = truth.get('S_true') if 'S_true' in truth else truth.get('z_true')

    run_sssm_plots(args.dataset, args.data_path, args.model_path, args.out_path,
                   C=args.C, delta_ms=args.delta_ms,
                   A_true=A_true, B_true=B_true, z_true=z_true)