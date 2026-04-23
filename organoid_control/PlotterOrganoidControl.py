import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import itertools
from scipy.stats import pearsonr, linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Note: Keeping the original imports; however, in a pipeline context, 
# ashwood_ns_8 components are typically passed in or re-implemented locally.
# If these imports fail in your specific environment, ensure ashwood_ns_8.py is present.
try:
    from ashwood_ns.ashwood_ns_8 import PrismNeuralGLMHMM, compute_expectations, find_best_tau_block
except ImportError:
    # Fallback to local definitions if necessary for standalone plotting
    pass

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Helper Utilities (Original)
# =============================================================================

def _smooth(x, w=20):
    """Rolling mean with window w. Pads edges so output length matches input."""
    x = np.array(x, dtype=float)
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    padded = np.concatenate([np.full(w - 1, x[0]), x])
    return np.convolve(padded, kernel, mode='valid')

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
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, [tp, fp, fn]):
        ax.text(bar.get_x() + bar.get_width() / 2, v / 2, str(int(v)), ha='center', va='center', fontweight='bold', color='white')
        ax.text(bar.get_x() + bar.get_width() / 2, v, str(int(v)), ha='center', va='bottom', fontweight='bold', fontsize=18)

    stats_text = f"precision = {m['prec']:.3f}\nrecall    = {m['rec']:.3f}\nF1        = {m['f1']:.3f}\naccuracy  = {m['acc']:.3f}\nτ_exc     = {tau_exc:.4f}\nτ_inh     = {tau_inh:.4f}"
    ax.text(0.02, 0.98, stats_text, fontsize=9, verticalalignment='top', transform=ax.transAxes, family='monospace', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

def _best_state_permutation_by_correlation(B_true, B_hat):
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
    b_min, b_max = min(B_true.min(), B_hat.min()), max(B_true.max(), B_hat.max())
    ax.scatter(B_true, B_hat, color=color, alpha=0.5)
    ax.plot([b_min, b_max], [b_min, b_max], 'k--', alpha=0.7)
    ax.set_title(f"State {state_idx} Bias (R={r_val:.3f})")

# =============================================================================
# NEW: Organoid Control Pipeline Plots
# =============================================================================

def plot_organoid_control_results(checkpoint, out_path):
    """Specific plotting for the 4-stage organoid control pipeline."""
    dataset = checkpoint.get('dataset', 'unknown')
    os.makedirs(out_path, exist_ok=True)
    
    # --- Stage 3: Adaptive Tracking Performance ---
    s3 = checkpoint.get('stage3', {})
    if s3:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        beta_hist = s3.get('beta_history', [])
        ax[0].plot(beta_hist, color='darkorange', linewidth=2)
        ax[0].set_title(f"Stage 3: Adaptive Forgetting Factor (β*)\nFinal β = {s3['summary']['beta_final']:.4f}")
        ax[0].set_xlabel("Online Bins")
        ax[0].set_ylabel("β")
        
        # Compare Errors (Adaptive vs Static)
        # Note: ashwood_organoid_control saves 'adaptive_se' and 'static_se' lists
        se_a = s3.get('adaptive_se', [])
        se_s = s3.get('static_se', [])
        if len(se_a) > 0:
            ax[1].plot(_smooth(se_a, 100), label='Adaptive (L3)', alpha=0.8)
            ax[1].plot(_smooth(se_s, 100), label='Static (L2)', alpha=0.8)
            ax[1].set_title(f"Tracking MSE (Smoothed)\nEV Lift: {s3['summary']['EV_lift']:+.4f}")
            ax[1].set_xlabel("Online Bins")
            ax[1].set_ylabel("Mean Squared Error")
            ax[1].legend()
        plt.tight_layout()
        plt.savefig(f"{out_path}/{dataset}_stage3_adaptive_tracking.png")
        plt.close()

    # --- Stage 4: Closed Loop Latency Breakdown ---
    s4 = checkpoint.get('stage4', {})
    if s4:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Latency Breakdown
        latencies = np.array(s4.get('latency_ms', [])) # [bins, 4]
        if latencies.size > 0:
            labels = ['Kalman', 'HMM', 'LQI', 'Safety']
            means = np.mean(latencies, axis=0)
            axes[0].bar(labels, means, color='teal')
            axes[0].axhline(10.0, color='red', linestyle='--', label='10ms Budget')
            axes[0].set_title("Mean Per-Bin Latency Breakdown")
            axes[0].set_ylabel("ms")
            axes[0].legend()
        
        # Control Signal Visualization
        u_cmd = np.array(s4.get('u_cmd', []))
        if u_cmd.size > 0:
            for i in range(min(5, u_cmd.shape[1])):
                axes[1].plot(u_cmd[:1000, i], alpha=0.5, label=f"Stim Ch {i}")
            axes[1].set_title("Control Signals u* (Dry Run - First 1000 bins)")
            axes[1].set_xlabel("Online Bins")
            axes[1].set_ylabel("u")
            axes[1].legend(fontsize=8, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"{out_path}/{dataset}_stage4_control_latency.png")
        plt.close()
    
    print(f"Organoid control pipeline plots saved to: {out_path}")

# =============================================================================
# Original Plotting Functions (Maintained)
# =============================================================================

def plot_residuals(model, A_true, B_true, dataset_name, out_path, suffix="", metrics=None):
    A_hat = model.A.detach().cpu().numpy()[:, :model.N]
    B_hat = model.B.detach().cpu().numpy()
    N = model.N
    n_exc = int(0.8 * N)
    mask_off = ~np.eye(N, dtype=bool)
    thresh = 1e-6

    # Data separation and residual calculation
    inh_true_flat = A_true[n_exc:][mask_off[n_exc:]]
    inh_hat_flat  = A_hat[n_exc:][mask_off[n_exc:]]
    inh_nz = np.abs(inh_true_flat) > thresh
    inh_resid = inh_hat_flat[inh_nz] - inh_true_flat[inh_nz]

    exc_true_flat = A_true[:n_exc][mask_off[:n_exc]]
    exc_hat_flat  = A_hat[:n_exc][mask_off[:n_exc]]
    exc_nz = np.abs(exc_true_flat) > thresh
    exc_resid = exc_hat_flat[exc_nz] - exc_true_flat[exc_nz]

    diag_resid = A_hat[np.arange(N), np.arange(N)] - A_true[np.arange(N), np.arange(N)]

    if metrics and 'tau_exc' in metrics:
        tau_exc, tau_inh = float(metrics['tau_exc']), float(metrics['tau_inh'])
    else:
        tau_exc = find_best_tau_block(A_hat[:n_exc], A_true[:n_exc])
        tau_inh = find_best_tau_block(A_hat[n_exc:], A_true[n_exc:])
    m = _block_metrics(A_hat, A_true, tau_exc, tau_inh, n_exc)

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    fig.suptitle(f"Edge & Weight Residuals: {dataset_name}", fontsize=13)

    def _hist_panel(ax, data, title, color):
        if data.size == 0: return
        ax.hist(data, bins=30, color=color, alpha=0.6)
        ax.axvline(0, color='k', linestyle='--')
        ax.set_title(title, fontsize=9)

    _hist_panel(axes[0], inh_resid, "A_off (Inhib) resid", 'steelblue')
    _hist_panel(axes[1], exc_resid, "A_off (Excit) resid", 'salmon')
    _hist_panel(axes[2], diag_resid, "A_diag residual", 'lightsalmon')

    tp, fp, fn = m['tp'], m['fp'], m['fn']
    axes[3].bar(['TP', 'FP', 'FN'], [tp, fp, fn], color=['green', 'red', 'purple'])
    axes[3].set_title("Edge Accuracy", fontsize=9)

    # B alignment
    best_b_perm = _best_state_permutation_by_correlation(B_true, B_hat)
    B_hat_aligned = np.array([B_hat[best_b_perm[k]] for k in range(B_true.shape[0])])
    b_resid = (B_hat_aligned - B_true).ravel()
    _hist_panel(axes[4], b_resid, "Baseline (B) resid", 'gray')

    plt.tight_layout()
    plt.savefig(f"{out_path}/{dataset_name}_residuals_{suffix}.png")
    plt.close()

def plot_connectivity_validation(model, A_true, B_true, dataset_name, out_path, suffix="", metrics=None):
    A_hat = model.A.detach().cpu().numpy()[:, :model.N]
    B_hat = model.B.detach().cpu().numpy()
    N = model.N
    n_exc = int(0.8 * N)

    if metrics and 'tau_exc' in metrics:
        tau_exc, tau_inh = float(metrics['tau_exc']), float(metrics['tau_inh'])
    else:
        tau_exc = find_best_tau_block(A_hat[:n_exc, :], A_true[:n_exc, :])
        tau_inh = find_best_tau_block(A_hat[n_exc:], A_true[n_exc:])
    
    m = _block_metrics(A_hat, A_true, tau_exc, tau_inh, n_exc)
    conf_img = np.zeros_like(A_hat); conf_img[m['true_mask'] & m['hat_mask']] = 3; conf_img[~m['true_mask'] & m['hat_mask']] = 2; conf_img[m['true_mask'] & ~m['hat_mask']] = 1
    cmap_conf = mcolors.ListedColormap(['white', '#df20df', '#ed2f20', '#38761d'])

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    ax_stats = fig.add_subplot(gs[0, 0]); _draw_stats_panel(ax_stats, m, tau_exc, tau_inh, "")
    ax_conf = fig.add_subplot(gs[0, 1]); _draw_confusion_panel(ax_conf, fig, conf_img, cmap_conf, "Confusion Map")
    
    ax_corr = fig.add_subplot(gs[0, 2])
    mask_off = ~np.eye(N, dtype=bool)
    ax_corr.scatter(A_true[mask_off], A_hat[mask_off], alpha=0.3, s=5)
    ax_corr.plot([-0.5, 0.5], [-0.5, 0.5], 'r--')
    ax_corr.set_title("Weight Correlation")

    best_perm = _best_state_permutation_by_correlation(B_true, B_hat)
    for k in range(min(B_true.shape[0], 3)):
        ax_b = fig.add_subplot(gs[1, k])
        _plot_bias_correlation_panel(ax_b, B_true[k], B_hat[best_perm[k]], k, 'blue')

    plt.tight_layout()
    plt.savefig(f"{out_path}/{dataset_name}_connectivity_validation_{suffix}.png")
    plt.close()

def plot_training_monitor(metrics, dataset_name, out_path, suffix=""):
    epochs = np.array(metrics['epochs'])
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].plot(metrics['em_iters'], metrics['nll_bin'], marker='o'); axes[0].set_title("E-step NLL")
    axes[1].plot(epochs, metrics['spectral_radius']); axes[1].set_title("Spectral Radius")
    axes[2].plot(epochs, metrics['nz_edges']); axes[2].set_title("Nonzero Edges")
    axes[3].plot(epochs, metrics['nll_bin_M']); axes[3].set_title("M-step NLL")
    plt.tight_layout()
    plt.savefig(f"{out_path}/{dataset_name}_training_monitor_{suffix}.png")
    plt.close()

def plot_state_prediction(gamma, z_true, dataset_name, out_path, time_step=0.1, max_time_s=1000.0):
    T, K = gamma.shape
    t = np.arange(T) * time_step
    z_hat = np.argmax(gamma, axis=1)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    axes[0].bar(range(K), np.mean(gamma, axis=0)); axes[0].set_title("State Occupancy")
    axes[1].plot(t[:1000], z_hat[:1000], label='Inferred'); axes[1].set_title("State Trace")
    if z_true is not None: axes[1].plot(t[:1000], z_true[:1000], '--', label='True')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{out_path}/{dataset_name}_state_prediction.png")
    plt.close()

def plot_ground_truth_vs_recovered(A_true, A_hat, dataset_name, out_path, suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(A_true, cmap='RdBu_r', vmin=-0.5, vmax=0.5); axes[0].set_title("Ground Truth A")
    axes[1].imshow(A_hat, cmap='RdBu_r', vmin=-0.5, vmax=0.5); axes[1].set_title("Recovered A")
    plt.savefig(f"{out_path}/{dataset_name}_A_comparison_{suffix}.png")
    plt.close()

# =============================================================================
# Execution Logic
# =============================================================================

def run_saved_model_plots(dataset, data_path, model_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 1. Detect if this is an organoid control checkpoint
    is_organoid = 'stage1' in checkpoint and 'stage3' in checkpoint
    
    if is_organoid:
        print("[Plotter] Detected Organoid Control Checkpoint Structure.")
        s1 = checkpoint['stage1']
        A_hat = s1['A_hat']
        # Reconstruct a light model for existing plotting functions
        class LightModel:
            def __init__(self, A, B, N):
                self.A = torch.tensor(A); self.B = torch.tensor(B); self.N = N
        
        # Load Truth Data
        truth_load = np.load(f"{data_path}/{dataset}.prismTruth.npz")
        A_true, B_true = truth_load['A_true'], truth_load['B_true']
        
        # Check for the correct key in the stage1 dictionary
        # In ashwood_organoid_control.py, B might be inside 'metrics' or 'state_dict'
        s1_metrics = s1.get('metrics', {})
        
        # Pull B from wherever it was saved (usually checkpointed in metrics for Stage 1)
        B_recovered = s1.get('B_hat', None) 
        if B_recovered is None:
            # Fallback: if B_hat isn't explicit, use the B from prism_state_dict if it exists
            B_recovered = s1.get('prism_state_dict', {}).get('B', np.zeros((3, A_true.shape[0])))

        model_wrap = LightModel(A_hat, B_recovered, A_true.shape[0])        
        # Original Characterization Plots (using Stage 1 results)
        plot_connectivity_validation(model_wrap, A_true, B_true, dataset, out_path, suffix="stage1", metrics=s1['metrics'])
        plot_training_monitor(s1['metrics'], dataset, out_path, suffix="stage1")
        plot_ground_truth_vs_recovered(A_true, A_hat, dataset, out_path, suffix="stage1")
        
        # New Pipeline Plots
        plot_organoid_control_results(che3ckpoint, out_path)
        
    else:
        # Standard PRISM Checkpoint (Original logic)
        config = checkpoint['hyperparameters']
        data_load = np.load(f"{data_path}/{dataset}.spikes.npz")
        truth_load = np.load(f"{data_path}/{dataset}.prismTruth.npz")
        A_true, B_true = truth_load['A_true'], truth_load['B_true']
        z_true = truth_load['S_true'] if 'S_true' in truth_load else None
        
        N = A_true.shape[0]
        model = PrismNeuralGLMHMM(n_states=int(config['n_states']), n_neurons=N, n_lags=int(config['n_lags'])).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Run Inference
        Y_np = data_load['spikes'].reshape(-1, N).astype(np.float32)
        X_np = np.roll(Y_np, 1, axis=0); X_np[0, :] = 0
        with torch.no_grad():
            gamma, _ = compute_expectations(model, torch.tensor(Y_np).to(device), torch.tensor(X_np).to(device), delta_t=float(config['delta_t']))
            gamma_np = gamma.cpu().numpy()

        plot_connectivity_validation(model, A_true, B_true, dataset, out_path, metrics=checkpoint.get('metrics', {}))
        plot_training_monitor(checkpoint['metrics'], dataset, out_path)
        plot_state_prediction(gamma_np, z_true, dataset, out_path)
        plot_ground_truth_vs_recovered(A_true, model.A.detach().cpu().numpy()[:, :N], dataset, out_path)
        plot_residuals(model, A_true, B_true, dataset, out_path, metrics=checkpoint.get('metrics', {}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    args = parser.parse_args()

    run_saved_model_plots(args.dataset, args.data_path, args.model_path, args.out_path)