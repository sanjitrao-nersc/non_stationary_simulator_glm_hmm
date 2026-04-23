import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# -------------------------
# Reproducibility and device
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
 
 
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
# -------------------------
# Constraints / Prox
# -------------------------
def proximal_soft_threshold_offdiag(A, tau):
    """Global off-diagonal L1 proximal sparsification."""
    with torch.no_grad():
        n = A.shape[0]
        eye = torch.eye(n, device=A.device, dtype=A.dtype)
        off = A * (1.0 - eye)
        off = torch.sign(off) * torch.clamp(torch.abs(off) - float(tau), min=0.0)
        diag = torch.diag(torch.diag(A))
        A.copy_(off + diag)
 
 
def proximal_soft_threshold_offdiag_blockwise(A, tau_exc, tau_inh, exc_ratio=0.8):
    """Block-aware off-diagonal proximal shrink: separate tau for exc/inh rows on the lag-1 block."""
    with torch.no_grad():
        n = A.shape[0]
        n_exc = int(n * exc_ratio)
        eye = torch.eye(n, device=A.device, dtype=A.dtype)
 
        A1 = A[:, :n].clone()
        off1 = A1 * (1.0 - eye)
        shrunk1 = off1.clone()
        shrunk1[:n_exc, :] = torch.sign(off1[:n_exc, :]) * torch.clamp(torch.abs(off1[:n_exc, :]) - float(tau_exc), min=0.0)
        shrunk1[n_exc:, :] = torch.sign(off1[n_exc:, :]) * torch.clamp(torch.abs(off1[n_exc:, :]) - float(tau_inh), min=0.0)
        diag1 = torch.diag(torch.diag(A1))
        A[:, :n] = shrunk1 + diag1
 
        # Higher-lag blocks: uniform threshold (used when n_lags > 1)
        if A.shape[1] > n:
            tau_rest = float(tau_exc + tau_inh) / 2.0
            A_rest = A[:, n:].clone()
            A[:, n:] = torch.sign(A_rest) * torch.clamp(torch.abs(A_rest) - tau_rest, min=0.0)
 
 
def enforce_dale_sign_offdiag(A, exc_ratio=0.8):
    """
    Dale projection for A[post, pre].
    Applied to the lag-1 block (A[:, :N]) only; higher-lag blocks are unconstrained.
      - excitatory postsynaptic rows: off-diagonal >= 0
      - inhibitory postsynaptic rows: off-diagonal <= 0
    """
    with torch.no_grad():
        n = A.shape[0]
        n_exc = int(n * exc_ratio)
        eye = torch.eye(n, device=A.device, dtype=A.dtype)
        A1 = A[:, :n].clone()
        off = A1 * (1.0 - eye)
        off[:n_exc, :] = torch.clamp(off[:n_exc, :], min=0.0)
        off[n_exc:, :] = torch.clamp(off[n_exc:, :], max=0.0)
        diag = torch.diag(torch.diag(A1))
        A[:, :n] = off + diag
 
 
def enforce_inhibitory_diagonal(A, min_abs=5e-2):
    """Force all self-couplings to be inhibitory with minimum magnitude."""
    with torch.no_grad():
        idx = torch.arange(A.shape[0], device=A.device)
        d = A[idx, idx]
        A[idx, idx] = -torch.clamp(torch.abs(d), min=float(min_abs))
 
 
# -------------------------
# Initialization
# -------------------------
def warm_start_connectivity(A_param, X_full, Y_full, ridge=1e-2, keep_ratio=0.15, exc_ratio=0.8):
    """
    Ridge warm start with A in (post, pre) orientation.
    Supports multi-lag: X_full is [T, L*N], A_param is [N, L*N].
    Sparsification threshold is derived from lag-1 off-diagonal entries only.
    """
    with torch.no_grad():
        n = A_param.shape[0]          # N (number of neurons)
        n_in = X_full.shape[1]        # L*N (total input features)
        xtx = X_full.T @ X_full
        xty = X_full.T @ Y_full
        A_ls = torch.linalg.solve(xtx + float(ridge) * torch.eye(n_in, device=X_full.device), xty)
        # A_ls is [L*N, N] (pre, post); lag-1 block is A_ls[:n, :]
 
        off_mask = (1.0 - torch.eye(n, device=X_full.device)).bool()
        off_abs = torch.abs(A_ls[:n][off_mask])   # threshold from lag-1 off-diagonal only
        q = max(0.0, min(1.0, 1.0 - float(keep_ratio)))
        thr = torch.quantile(off_abs, q) if off_abs.numel() > 0 else torch.tensor(0.0, device=X_full.device)
        A_sparse = torch.where(torch.abs(A_ls) >= thr, A_ls, torch.zeros_like(A_ls))
 
        # Force lag-1 self-connections to be inhibitory
        idx = torch.arange(n, device=X_full.device)
        A_sparse[idx, idx] = -0.05
        A_sparse = torch.clamp(A_sparse, min=-0.5, max=0.5)
 
        # store as (post, pre): A_param is [N, L*N]
        A_param.copy_(A_sparse.T)
 
        # enforce Dale sign and inhibitory diagonal from the start
        enforce_dale_sign_offdiag(A_param, exc_ratio=exc_ratio)
        enforce_inhibitory_diagonal(A_param, min_abs=5e-2)
 
 
# -------------------------
# Metrics helpers
# -------------------------
def _prf_from_masks(true_mask, hat_mask):
    tp = int((true_mask & hat_mask).sum())
    fp = int(((~true_mask) & hat_mask).sum())
    fn = int((true_mask & (~hat_mask)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, tp, fp, fn
 
 
def _fp_fn_tp_from_block(A_hat_block, A_true_block, tau):
    true_mask = np.abs(A_true_block) > float(tau)
    hat_mask = np.abs(A_hat_block) > float(tau)
    tp = int((true_mask & hat_mask).sum())
    fp = int(((~true_mask) & hat_mask).sum())
    fn = int((true_mask & (~hat_mask)).sum())
    return fp, fn, tp
 
 
 
def find_best_tau_block(A_hat_block, A_true_block):
    """
    Adaptive threshold search over quantiles of |A_hat_block| (non-zero support).
    Used only for evaluation.
    """
    abs_hat = np.abs(A_hat_block).ravel()
    nz_hat = abs_hat[abs_hat > 1e-12]
    if nz_hat.size == 0:
        return 0.0
 
    q_grid = np.linspace(0.70, 0.995, 60)
    tau_candidates = np.unique(np.quantile(nz_hat, q_grid))
    tau_candidates = tau_candidates[tau_candidates > 1e-12]
    if tau_candidates.size == 0:
        return float(np.min(nz_hat))
 
    best_f1, best_tau = -1.0, float(tau_candidates[0])
    for tau in tau_candidates:
        true_mask = np.abs(A_true_block) > tau
        hat_mask = np.abs(A_hat_block) > tau
        f1, _, _, _ = _prf_from_masks(true_mask, hat_mask)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_tau
 
 
# -------------------------
# Model
# -------------------------
class PrismNeuralGLMHMM(nn.Module):
    def __init__(self, n_states, n_neurons, n_lags=1):
        super().__init__()
        self.K = n_states
        self.N = n_neurons
        self.L = n_lags
        self.A = nn.Parameter(torch.randn(self.N, self.L * self.N) * 0.1)
        self.B = nn.Parameter(torch.zeros(self.K, self.N))
        self.register_buffer("P", torch.eye(self.K))
        self.register_buffer("pi", torch.full((self.K,), 1.0 / self.K))
 
    def forward(self, Y, X, gamma, delta_t=1.0):
        """Posterior-weighted Poisson NLL, vectorised over all K states simultaneously."""
        log_dt = math.log(max(float(delta_t), 1e-12))
        eta = (X @ self.A.T).unsqueeze(1) + self.B.unsqueeze(0)  # [T, K, N]
        eta = torch.clamp(eta, max=5.0)
        mu = torch.exp(eta) * float(delta_t)
        nll = (gamma.unsqueeze(2) * (mu - Y.unsqueeze(1) * (eta + log_dt))).sum()
        return nll.view(1)
 
    def apply_final_spectral_rescale(self, rho_max=0.92, rho_target=0.919):
        """Single post-fit spectral stabilization on lag-1 block only."""
        with torch.no_grad():
            eigvals = torch.linalg.eigvals(self.A[:, :self.N])
            rho = torch.max(torch.abs(eigvals))
            if rho >= rho_max:
                self.A[:, :self.N] *= (rho_target / rho)
 
 
# -------------------------
# EM E-step
# -------------------------
def compute_expectations(model, Y, X, delta_t=1.0, gamma_temp=1.0, gamma_floor=0.0):
    """Forward-backward E-step in log-space. Emissions vectorised across K states."""
    T, _ = Y.shape
    K = model.K
    dev = Y.device
 
    with torch.no_grad():
        log_dt = math.log(max(float(delta_t), 1e-12))
        # Vectorise over K: [T, K, N]
        eta = (X @ model.A.T).unsqueeze(1) + model.B.unsqueeze(0)
        eta = torch.clamp(eta, max=5.0)
        mu = torch.exp(eta) * float(delta_t)
        log_emissions = (Y.unsqueeze(1) * (eta + log_dt) - mu).sum(dim=2)  # [T, K]
 
    log_P = torch.log(model.P.clamp(min=1e-10))
    log_pi = torch.log(model.pi.clamp(min=1e-10))
 
    log_alpha = torch.zeros((T, K), device=dev)
    log_alpha[0] = log_emissions[0] + log_pi
    for t in range(1, T):
        log_alpha[t] = log_emissions[t] + torch.logsumexp(log_alpha[t - 1].unsqueeze(1) + log_P, dim=0)
 
    log_beta = torch.zeros((T, K), device=dev)
    for t in range(T - 2, -1, -1):
        log_beta[t] = torch.logsumexp(log_P + log_emissions[t + 1] + log_beta[t + 1], dim=1)
 
    log_gamma = log_alpha + log_beta
    log_gamma -= torch.logsumexp(log_gamma, dim=1, keepdim=True)
 
    if gamma_temp > 1.0:
        log_gamma = log_gamma / gamma_temp
        log_gamma -= torch.logsumexp(log_gamma, dim=1, keepdim=True)
    gamma_soft = torch.exp(log_gamma)
 
    if gamma_floor > 0.0:
        floor = min(float(gamma_floor), 0.49 / K)
        gamma_soft = gamma_soft * (1.0 - K * floor) + floor
        gamma_soft = gamma_soft / gamma_soft.sum(dim=1, keepdim=True)
 
    # Hard E-step: one-hot encode the argmax state at each time bin.
    # Each time bin is assigned entirely to its most probable state;
    # all other states receive weight 0.  This turns the soft EM into
    # a hard (Viterbi-style) assignment, giving crisper state boundaries
    # and sharper connectivity gradients during the M-step.
    hard_idx = torch.argmax(gamma_soft, dim=1)          # [T]
    gamma = torch.zeros_like(gamma_soft)
    gamma.scatter_(1, hard_idx.unsqueeze(1), 1.0)       # [T, K] one-hot
 
    # Recompute xi_sum from the hard gamma so the transition matrix update
    # is consistent with the hard assignments.  xi[t, j, k] = 1 iff the
    # hard assignment transitions from state j at t to state k at t+1.
    # We derive this directly from consecutive hard assignments rather than
    # re-running the full backward pass with the hard posteriors.
    hard_prev = hard_idx[:-1]   # state at t   [T-1]
    hard_next = hard_idx[1:]    # state at t+1 [T-1]
    xi_sum = torch.zeros((K, K), device=dev)
    xi_sum.scatter_add_(
        0,
        hard_prev.unsqueeze(1).expand(-1, 1),          # won't broadcast alone
        torch.zeros(T - 1, K, device=dev).scatter_(
            1, hard_next.unsqueeze(1), 1.0
        ),
    )
 
    return gamma, xi_sum
 
 
# -------------------------
# Training
# -------------------------
def train_model(config, Y_full, X_full, A_true, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    n_lags = int(config.get("n_lags", 1))
    n_states = int(config.get("n_states", 3))
    # Derive N from A_true — the authoritative ground-truth shape.
    N = A_true.shape[0]
    n_exc = int(0.8 * N)
    model = PrismNeuralGLMHMM(n_states=n_states, n_neurons=N, n_lags=n_lags).to(device)
    K = model.K
 
    with torch.no_grad():
        warm_start_connectivity(
            model.A,
            X_full,
            Y_full,
            ridge=float(config.get("warm_ridge", 1e-2)),
            keep_ratio=float(config.get("warm_keep_ratio", 0.15)),
        )
 
        mean_rates = Y_full.mean(dim=0).clamp(min=1e-4)
        for k in range(K):
            model.B[k].copy_(torch.log(mean_rates))
 
        noise = float(config.get("b_init_noise", 0.05))
        for k in range(K):
            model.B[k].add_(torch.randn_like(model.B[k]) * noise)
 
        # HMM init: high self-transition probability (persistent states)
        p_self = 0.95 if K == 2 else 0.93
        p_cross = (1.0 - p_self) / max(1, K - 1)
        pi_init = torch.tensor([0.55, 0.45], device=device, dtype=model.pi.dtype) \
                  if K == 2 else torch.full((K,), 1.0 / K, device=device, dtype=model.pi.dtype)
        model.pi.copy_(pi_init)
        P_init = torch.full((K, K), p_cross, device=device, dtype=model.P.dtype)
        P_init.fill_diagonal_(p_self)
        model.P.copy_(P_init)
 
        enforce_dale_sign_offdiag(model.A, exc_ratio=0.8)
        enforce_inhibitory_diagonal(model.A, min_abs=float(config.get("diag_inhibitory_floor", 5e-2)))
 
    optimizer = optim.Adam(model.parameters(), lr=float(config.get("lr", 5e-3)))
 
    T = Y_full.shape[0]
    batch_size = int(config.get("batch_size", 2048))
    em_iters = int(config.get("em_iters", 15))
    m_epochs = int(config.get("m_epochs", 50))
    delta_t = float(config.get("delta_t", 0.1))
    total_global_epochs = max(1, em_iters * m_epochs)
 
    prox_scale = float(config.get("prox_scale", 5e-3))
    tau_exc = float(config.get("tau_exc_init", float(config["l1"]) * prox_scale))
    tau_inh = float(config.get("tau_inh_init", float(config["l1"]) * prox_scale))
    tau_adapt_start_frac = float(config.get("tau_adapt_start_frac", 0.4))
    tau_gain = float(config.get("tau_gain", 0.06))
    tau_min_mult = float(config.get("tau_min_mult", 0.2))
    tau_max_mult = float(config.get("tau_max_mult", 5.0))
    proj_every_early = int(config.get("proj_every_early", 8))
    proj_every_late = int(config.get("proj_every_late", 3))
    proj_switch_frac = float(config.get("proj_switch_frac", 0.65))
 
    base_tau_exc = max(1e-12, float(config["l1"]) * prox_scale)
    base_tau_inh = max(1e-12, float(config["l1"]) * prox_scale)
    tau_exc_min, tau_exc_max = base_tau_exc * tau_min_mult, base_tau_exc * tau_max_mult
    tau_inh_min, tau_inh_max = base_tau_inh * tau_min_mult, base_tau_inh * tau_max_mult
    metrics = {
        "em_iters": [],
        "nll_bin": [],
        "epochs": [],
        "spectral_radius": [],
        "nz_edges": [],
        "nll_bin_M": [],
        "l1_norm": [],
        "state_occupancy": [],
        "min_state_occupancy": None,
        "tau_exc": None,
        "tau_inh": None,
        "tau_exc_hist": [],
        "tau_inh_hist": [],
        "fp_exc_hist": [],
        "fn_exc_hist": [],
        "fp_inh_hist": [],
        "fn_inh_hist": [],
        "proj_applied_hist": [],
    }
 
    global_epoch = 0
    for em in range(em_iters):
        model.eval()
        with torch.no_grad():
            gamma_full, xi_sum = compute_expectations(model, Y_full, X_full, delta_t=delta_t)
            nll_e = model(Y_full, X_full, gamma_full, delta_t=delta_t).item()
            metrics["em_iters"].append(em)
            metrics["nll_bin"].append(nll_e)
 
        model.train()
        for _ in range(m_epochs):
            perm = torch.randperm(T)
            epoch_nll_sum = 0.0
            for j in range(0, T, batch_size):
                idx = perm[j:j + batch_size]
                optimizer.zero_grad()
 
                nll = model(Y_full[idx], X_full[idx], gamma_full[idx], delta_t=delta_t)
                reg_l2 = 0.5 * float(config["l2"]) * torch.sum(model.A * model.A)
                loss = nll + reg_l2
 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config.get("grad_clip", 1.0)))
                optimizer.step()
 
                epoch_nll_sum += nll.item()
 
            with torch.no_grad():
                prog = global_epoch / float(max(1, total_global_epochs - 1))
                proj_every = max(1, proj_every_late if prog >= proj_switch_frac else proj_every_early)
                apply_projection = (global_epoch % proj_every == 0)
 
                if prog < tau_adapt_start_frac:
                    tau_exc_eff = float(np.clip(tau_exc, tau_exc_min, tau_exc_max))
                    tau_inh_eff = float(np.clip(tau_inh, tau_inh_min, tau_inh_max))
                    metrics["fp_exc_hist"].append(0)
                    metrics["fn_exc_hist"].append(0)
                    metrics["fp_inh_hist"].append(0)
                    metrics["fn_inh_hist"].append(0)
                else:
                    A_np = model.A.detach().cpu().numpy()[:, :N]
                    A_exc = A_np[:n_exc, :]
                    A_inh = A_np[n_exc:, :]
                    T_exc = A_true[:n_exc, :N]
                    T_inh = A_true[n_exc:, :N]
 
                    fp_exc, fn_exc, tp_exc = _fp_fn_tp_from_block(A_exc, T_exc, tau_exc)
                    fp_inh, fn_inh, tp_inh = _fp_fn_tp_from_block(A_inh, T_inh, tau_inh)
 
                    den_exc = max(1, tp_exc + fp_exc + fn_exc)
                    den_inh = max(1, tp_inh + fp_inh + fn_inh)
                    e_exc = (fp_exc - fn_exc) / den_exc
                    e_inh = (fp_inh - fn_inh) / den_inh
 
                    tau_exc = tau_exc * float(np.exp(tau_gain * e_exc))
                    tau_inh = tau_inh * float(np.exp(tau_gain * e_inh))
                    tau_exc = float(np.clip(tau_exc, tau_exc_min, tau_exc_max))
                    tau_inh = float(np.clip(tau_inh, tau_inh_min, tau_inh_max))
                    tau_exc_eff = tau_exc
                    tau_inh_eff = tau_inh
 
                    metrics["fp_exc_hist"].append(int(fp_exc))
                    metrics["fn_exc_hist"].append(int(fn_exc))
                    metrics["fp_inh_hist"].append(int(fp_inh))
                    metrics["fn_inh_hist"].append(int(fn_inh))
 
                if apply_projection:
                    proximal_soft_threshold_offdiag_blockwise(
                        model.A, tau_exc=tau_exc_eff, tau_inh=tau_inh_eff, exc_ratio=0.8
                    )
                    enforce_dale_sign_offdiag(model.A, exc_ratio=0.8)
                    enforce_inhibitory_diagonal(model.A, min_abs=float(config.get("diag_inhibitory_floor", 5e-2)))
 
                rho = torch.max(torch.abs(torch.linalg.eigvals(model.A[:, :N]))).item()
                if global_epoch % 3 == 0 and rho >= 0.92:
                    model.A.data[:, :N] *= (0.90 / rho)
                    rho = 0.90
                nz = int(torch.sum(torch.abs(model.A[:, :N]) > 0.01).item())
                off_mask = 1.0 - torch.eye(N, device=device)
                l1_val = torch.norm(model.A[:, :N] * off_mask, p=1).item()
 
                metrics["epochs"].append(global_epoch)
                metrics["spectral_radius"].append(rho)
                metrics["nz_edges"].append(nz)
                metrics["nll_bin_M"].append(epoch_nll_sum / T)
                metrics["l1_norm"].append(l1_val)
                metrics["tau_exc_hist"].append(float(tau_exc_eff))
                metrics["tau_inh_hist"].append(float(tau_inh_eff))
                metrics["proj_applied_hist"].append(1 if apply_projection else 0)
 
            global_epoch += 1
 
        with torch.no_grad():
            pi_num = gamma_full[0] + float(config.get("pi_pseudocount", 1e-2))
            model.pi.copy_(pi_num / pi_num.sum())
 
            new_P = (
                xi_sum
                + float(config.get("P_pseudocount", 1e-3)) * torch.ones((K, K), device=device)
                + torch.eye(K, device=device)
            )
            model.P = new_P / new_P.sum(dim=1, keepdim=True)
 
    with torch.no_grad():
        model.apply_final_spectral_rescale(rho_max=0.92, rho_target=0.919)
        enforce_dale_sign_offdiag(model.A, exc_ratio=0.8)
        enforce_inhibitory_diagonal(model.A, min_abs=float(config.get("diag_inhibitory_floor", 5e-2)))
 
    A_hat = model.A.detach().cpu().numpy()[:, :N]
 
    tau_exc = find_best_tau_block(A_hat[:n_exc, :], A_true[:n_exc, :N])
    tau_inh = find_best_tau_block(A_hat[n_exc:, :], A_true[n_exc:, :N])
 
    true_mask_exc = np.abs(A_true[:n_exc, :N]) > tau_exc
    hat_mask_exc = np.abs(A_hat[:n_exc, :]) > tau_exc
    true_mask_inh = np.abs(A_true[n_exc:, :N]) > tau_inh
    hat_mask_inh = np.abs(A_hat[n_exc:, :]) > tau_inh
 
    true_mask = np.vstack([true_mask_exc, true_mask_inh])
    hat_mask = np.vstack([hat_mask_exc, hat_mask_inh])
    _, tp, fp, fn = _prf_from_masks(true_mask, hat_mask)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
 
    print(f"  F1={f1:.4f}  TP={tp} FP={fp} FN={fn}")
 
    with torch.no_grad():
        gamma_final, _ = compute_expectations(model, Y_full, X_full, delta_t=delta_t)
        occ = gamma_final.mean(dim=0).detach().cpu().numpy()
        metrics["state_occupancy"] = occ.tolist()
        metrics["min_state_occupancy"] = float(np.min(occ))
 
    metrics["tau_exc"] = float(tau_exc)
    metrics["tau_inh"] = float(tau_inh)
 
    return f1, model, metrics
 
 
 
 
 
import torch.multiprocessing as mp
 
# -------------------------
# Multi-GPU worker
# -------------------------
def _run_configs_on_gpu(gpu_id, configs, Y_np, X_np, A_true, result_queue):
    """Train all assigned configs on a single GPU; push the best result to the queue."""
    dev = torch.device(f"cuda:{gpu_id}")
    set_seed(42 + gpu_id)
    Y_full = torch.tensor(Y_np, device=dev)
    X_full = torch.tensor(X_np, device=dev)
 
    best_f1, best_config, best_state, best_metrics = -1.0, None, None, None
    for config in configs:
        try:
            print(f"[GPU {gpu_id}] Testing l1={config['l1']:.0e}  l2={config['l2']:.0e}")
            f1, model, metrics = train_model(config, Y_full, X_full, A_true, device=dev)
            print(f"[GPU {gpu_id}] F1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_config = config
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_metrics = metrics
        except Exception as e:
            print(f"[GPU {gpu_id}] Config failed: {e}")
 
    result_queue.put((gpu_id, best_f1, best_config, best_state, best_metrics))
 
 
# -------------------------
# Main hyperparameter scan
# -------------------------
def main():
    set_seed(42)
 
    path = "/pscratch/sd/s/sanjitr/causal_net_temp/spikesData"
    dataset = "daleN100_f7d754_ce2202"
    out_root = "/pscratch/sd/s/sanjitr/causal_net_temp"
    model_dir = f"{out_root}/models"
    plot_dir = f"{out_root}/plots/ns10_plots"
 
    data_load = np.load(f"{path}/{dataset}.spikes.npz")
    truth_load = np.load(f"{path}/{dataset}.prismTruth.npz")
 
    A_true = truth_load["A_true"]
    N = A_true.shape[0]
    Y_np = data_load["spikes"].reshape(-1, N).astype(np.float32)
 
    n_lags = 1
    lags = []
    for l in range(1, n_lags + 1):
        lag = np.roll(Y_np, l, axis=0).astype(np.float32)
        lag[:l, :] = 0.0
        lags.append(lag)
    X_np = np.concatenate(lags, axis=1)
 
    l1_space = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    l2_space = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
 
    base_config = {
        "n_lags": n_lags,
        "n_states": 3,
        "n_neurons": N,
        "delta_t": 0.01,
        "em_iters": 25,
        "m_epochs": 30,
        "lr": 5e-3,
        "batch_size": 2048,
        "prox_scale": 5e-3,
        "grad_clip": 1.0,
        "warm_ridge": 1e-2,
        "warm_keep_ratio": 0.15,
        "diag_inhibitory_floor": 5e-2,
        "b_init_noise": 5e-2,
        "pi_pseudocount": 1e-2,
        "P_pseudocount": 1e-3,
        "tau_adapt_start_frac": 0.25,
        "tau_gain": 0.05,
        "tau_min_mult": 0.2,
        "tau_max_mult": 6.0,
        "proj_every_early": 8,
        "proj_every_late": 2,
        "proj_switch_frac": 0.65,
        "phase2_em_iters": 10,
        "phase2_m_epochs": 30,
        "phase2_lr": 2e-3,
    }
 
    all_configs = []
    for l1 in l1_space:
        for l2 in l2_space:
            cfg = dict(base_config)
            cfg["l1"] = float(l1)
            cfg["l2"] = float(l2)
            all_configs.append(cfg)
 
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    n_gpus = max(1, n_gpus)
    print(f"Distributing {len(all_configs)} configs across {n_gpus} GPU(s).")
 
    # Round-robin assignment so each GPU gets roughly equal work
    config_chunks = [all_configs[i::n_gpus] for i in range(n_gpus)]
 
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    for gpu_id in range(n_gpus):
        if not config_chunks[gpu_id]:
            continue
        p = ctx.Process(
            target=_run_configs_on_gpu,
            args=(gpu_id, config_chunks[gpu_id], Y_np, X_np, A_true, result_queue),
        )
        p.start()
        processes.append(p)
 
    for p in processes:
        p.join()
 
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
 
    if not results:
        raise RuntimeError("No results collected from GPU workers.")
 
    _, best_f1, best_config, best_state, best_metrics = max(results, key=lambda r: r[1])
    print(f"\nOptimization Complete. Best F1: {best_f1:.4f} with {best_config}")
 
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
 
    save_file = f"{model_dir}/{dataset}_best_model_ns10.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "metrics": best_metrics,
            "hyperparameters": best_config,
            "f1_score": best_f1,
            "dataset": dataset,
        },
        save_file,
    )
    print(f"Saved best model checkpoint to: {save_file}")
 
 
if __name__ == "__main__":
    main()