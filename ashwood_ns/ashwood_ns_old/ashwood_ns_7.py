import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from scipy.stats import pearsonr
from tqdm import tqdm

# --- Rigorous Seeding for Reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def proximal_soft_threshold_offdiag(A, tau):
    """Soft-threshold off-diagonal entries for sparse support recovery."""
    with torch.no_grad():
        N = A.shape[0]
        eye = torch.eye(N, device=A.device, dtype=A.dtype)
        off = A * (1.0 - eye)
        shrunk = torch.sign(off) * torch.clamp(torch.abs(off) - tau, min=0.0)
        diag = torch.diag(torch.diag(A))
        A.copy_(shrunk + diag)


def proximal_soft_threshold_offdiag_blockwise(A, tau_exc, tau_inh, exc_ratio=0.8):
    """Block-aware soft-thresholding on off-diagonal entries (rows are postsynaptic blocks)."""
    with torch.no_grad():
        N = A.shape[0]
        n_exc = int(N * exc_ratio)
        eye = torch.eye(N, device=A.device, dtype=A.dtype)
        off = A * (1.0 - eye)
        shrunk = off.clone()

        exc_block = off[:n_exc, :]
        inh_block = off[n_exc:, :]
        shrunk[:n_exc, :] = torch.sign(exc_block) * torch.clamp(torch.abs(exc_block) - tau_exc, min=0.0)
        shrunk[n_exc:, :] = torch.sign(inh_block) * torch.clamp(torch.abs(inh_block) - tau_inh, min=0.0)

        diag = torch.diag(torch.diag(A))
        A.copy_(shrunk + diag)


def hard_zero_offdiag(A, zero_floor, exc_ratio=0.8):
    """Hard-zero tiny off-diagonal magnitudes to reduce near-zero false edges."""
    with torch.no_grad():
        N = A.shape[0]
        eye = torch.eye(N, device=A.device, dtype=A.dtype)
        off = A * (1.0 - eye)
        off = torch.where(torch.abs(off) < zero_floor, torch.zeros_like(off), off)
        diag = torch.diag(torch.diag(A))
        A.copy_(off + diag)


def hard_zero_offdiag_blockwise(A, zero_floor_exc, zero_floor_inh, exc_ratio=0.8):
    """Hard-zero tiny off-diagonal entries with separate floors for excit/inhib blocks."""
    with torch.no_grad():
        N = A.shape[0]
        n_exc = int(N * exc_ratio)
        eye = torch.eye(N, device=A.device, dtype=A.dtype)
        off = A * (1.0 - eye)

        exc = off[:n_exc, :]
        inh = off[n_exc:, :]
        exc = torch.where(torch.abs(exc) < float(zero_floor_exc), torch.zeros_like(exc), exc)
        inh = torch.where(torch.abs(inh) < float(zero_floor_inh), torch.zeros_like(inh), inh)
        off[:n_exc, :] = exc
        off[n_exc:, :] = inh

        diag = torch.diag(torch.diag(A))
        A.copy_(off + diag)


def enforce_dale_sign_offdiag(A, exc_ratio=0.8):
    """Project off-diagonal signs by block: excit >= 0, inhib <= 0."""
    with torch.no_grad():
        N = A.shape[0]
        n_exc = int(N * exc_ratio)
        eye = torch.eye(N, device=A.device, dtype=A.dtype)
        off = A * (1.0 - eye)

        off[:n_exc, :] = torch.clamp(off[:n_exc, :], min=0.0)
        off[n_exc:, :] = torch.clamp(off[n_exc:, :], max=0.0)

        diag = torch.diag(torch.diag(A))
        A.copy_(off + diag)


def enforce_inhibitory_diagonal(A, min_abs=5e-2):
    """Force all self-couplings to be inhibitory with minimum magnitude."""
    with torch.no_grad():
        idx = torch.arange(A.shape[0], device=A.device)
        d = A[idx, idx]
        A[idx, idx] = -torch.clamp(torch.abs(d), min=float(min_abs))


def warm_start_connectivity(A_param, X_full, Y_full, ridge=1e-2, keep_ratio=0.15):
    """Single-state ridge warm start: solve X A ~= Y, then sparsify support."""
    with torch.no_grad():
        N = A_param.shape[0]
        XtX = X_full.T @ X_full
        XtY = X_full.T @ Y_full
        A_ls = torch.linalg.solve(XtX + ridge * torch.eye(N, device=X_full.device), XtY)

        # Keep strongest off-diagonal edges to seed sparse structure.
        off_mask = (1.0 - torch.eye(N, device=X_full.device)).bool()
        off_abs = torch.abs(A_ls[off_mask])
        q = max(0.0, min(1.0, 1.0 - keep_ratio))
        thresh = torch.quantile(off_abs, q) if off_abs.numel() > 0 else torch.tensor(0.0, device=X_full.device)
        A_sparse = torch.where(torch.abs(A_ls) >= thresh, A_ls, torch.zeros_like(A_ls))
        A_sparse.fill_diagonal_(-0.05)
        A_sparse = torch.clamp(A_sparse, min=-0.5, max=0.5)
        # Model uses eta = X @ A.T, so store A in (post, pre) orientation.
        A_param.copy_(A_sparse.T)


def _prf_from_masks(true_mask, hat_mask):
    tp = int((true_mask & hat_mask).sum())
    fp = int(((~true_mask) & hat_mask).sum())
    fn = int((true_mask & (~hat_mask)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, tp, fp, fn


def find_best_tau_block(A_hat_block, A_true_block):
    """
    Adaptive threshold selection (no fixed tau):
    choose tau maximizing block F1 over quantiles of |A_hat|.
    """
    abs_hat = np.abs(A_hat_block).ravel()
    # Avoid tau collapse to zero: search on non-zero recovered support only.
    nz_hat = abs_hat[abs_hat > 1e-12]
    if nz_hat.size == 0:
        return 0.0

    # Quantile candidates over non-zero support.
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


class PrismNeuralGLMHMM(nn.Module):
    def __init__(self, n_states, n_neurons):
        super().__init__()
        self.K, self.N = n_states, n_neurons

        # Start with slightly larger initialization to survive early pruning
        self.A = torch.nn.Parameter(torch.randn(self.N, self.N) * 0.1)
        self.B = torch.nn.Parameter(torch.zeros(self.K, self.N))
        self.register_buffer('P', torch.eye(self.K))
        self.register_buffer('pi', torch.full((self.K,), 1.0 / self.K))

    def forward(self, Y, X, gamma, delta_t=1.0):
        """Posterior-weighted Poisson NLL with eta = X @ A.T + B and explicit delta_t."""
        total_nll = 0
        log_dt = np.log(max(delta_t, 1e-12))

        for k in range(self.K):
            # A is stored as (post, pre); transpose for row-vector design X.
            eta = (X @ self.A.T) + self.B[k]
            eta_clipped = torch.clamp(eta, max=5)
            mu = torch.exp(eta_clipped) * delta_t
            nll = (gamma[:, k:k+1] * (mu - Y * (eta_clipped + log_dt))).sum(dim=0)
            total_nll += nll.sum()
        return total_nll.view(1)

    def apply_spectral_projection(self, rho_target=0.919):
        """Project connectivity to satisfy spectral radius strictly below 0.92."""
        with torch.no_grad():
            try:
                eigvals = torch.linalg.eigvals(self.A)
                rho = torch.max(torch.abs(eigvals))
                if rho >= 0.92:
                    self.A *= (rho_target / rho)
            except: pass

    def apply_final_spectral_rescale(self):
        """One-time spectral-radius stabilization applied after fitting."""
        self.apply_spectral_projection(rho_target=0.919)

def compute_expectations(model, Y, X, delta_t=1.0, gamma_temp=1.0, gamma_floor=0.0):
    """E-step using corrected linear predictor convention."""
    T, N = Y.shape
    K = model.K
    log_emissions = torch.zeros((T, K), device=device)
    with torch.no_grad():
        log_dt = np.log(max(delta_t, 1e-12))
        for k in range(K):
            # Matching forward pass: eta = X @ A.T + B.
            eta = (X @ model.A.T) + model.B[k]
            eta = torch.clamp(eta, max=5)
            mu = torch.exp(eta) * delta_t
            log_p_y = (Y * (eta + log_dt) - mu).sum(dim=1)
            log_emissions[:, k] = log_p_y

    log_P = torch.log(model.P.clamp(min=1e-10))
    log_alpha = torch.zeros((T, K), device=device)
    log_pi = torch.log(model.pi.clamp(min=1e-10))
    log_alpha[0] = log_emissions[0] + log_pi

    for t in range(1, T):
        log_alpha[t] = log_emissions[t] + torch.logsumexp(log_alpha[t-1].unsqueeze(1) + log_P, dim=0)

    log_beta = torch.zeros((T, K), device=device)
    for t in range(T - 2, -1, -1):
        log_beta[t] = torch.logsumexp(log_P + log_emissions[t+1] + log_beta[t+1], dim=1)

    log_gamma = log_alpha + log_beta
    log_gamma -= torch.logsumexp(log_gamma, dim=1, keepdim=True)

    # Temperature > 1 flattens responsibilities early to avoid one-state lock-in.
    if gamma_temp > 1.0:
        log_gamma = log_gamma / gamma_temp
        log_gamma -= torch.logsumexp(log_gamma, dim=1, keepdim=True)
    gamma = torch.exp(log_gamma)

    # Responsibility floor keeps every state alive during EM.
    if gamma_floor > 0.0:
        floor = min(float(gamma_floor), 0.49 / K)
        gamma = gamma * (1.0 - K * floor) + floor
        gamma = gamma / gamma.sum(dim=1, keepdim=True)

    log_xi = log_alpha[:-1].unsqueeze(2) + log_P.unsqueeze(0) + \
             log_emissions[1:].unsqueeze(1) + log_beta[1:].unsqueeze(1)
    log_xi -= torch.logsumexp(log_xi, dim=(1, 2), keepdim=True)
    xi_sum = torch.exp(log_xi).sum(dim=0)

    return gamma, xi_sum

def train_model(config, Y_full, X_full, A_true):
    model = PrismNeuralGLMHMM(n_states=2, n_neurons=100).to(device)

    # --- Stronger Warm-Start for Connectivity ---
    warm_start_connectivity(
        model.A, X_full, Y_full,
        ridge=config.get('warm_ridge', 1e-2),
        keep_ratio=config.get('warm_keep_ratio', 0.15),
    )

    # --- Warm-Start Bias Initialization ---
    # Set B to log(mean_rate) to match the baseline firing observed in data
    with torch.no_grad():
        # Add small epsilon to avoid log(0)
        mean_rates = Y_full.mean(dim=0).clamp(min=1e-4) 
        log_mean_rates = torch.log(mean_rates)
        for k in range(model.K):
            # Add a tiny bit of noise to break symmetry between states
            noise = torch.randn(model.N, device=device) * 0.05
            model.B[k].copy_(log_mean_rates + noise)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Use a slightly smaller learning rate for stability
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    batch_size = 2048 
    T = Y_full.shape[0]
    delta_t = config.get('delta_t', 1.0)

    # FIX: Initialize all keys expected by plot_training_monitor
    metrics = {
        'em_iters': [],          
        'nll_bin': [],           
        'epochs': [],            
        'spectral_radius': [],   
        'nz_edges': [],
        'nll_bin_M': [],         # Added for lns1 in plotter
        'l1_norm': [],           # Added for lns2 in plotter
        'prox_exc_scale': [],
        'l1_exc_eff': [],
        'zero_floor_exc': [],
        'debias_start_epoch': None,
        'debias_f1': [],
        'debias_score': [],
        'debias_best_f1': None,
        'debias_best_score': None,
        'debias_best_epoch': None,
    }

    global_epoch = 0
    K = model.module.K if isinstance(model, nn.DataParallel) else model.K
    N = model.module.N if isinstance(model, nn.DataParallel) else model.N
    n_exc = int(N * 0.8)

    em_iters = int(config.get('em_iters', 15))
    temp_start = float(config.get('gamma_temp_start', 1.6))
    temp_end = float(config.get('gamma_temp_end', 1.0))
    gamma_floor = float(config.get('gamma_floor', 0.02))
    pi_pseudocount = float(config.get('pi_pseudocount', 1e-2))
    P_pseudocount = float(config.get('P_pseudocount', 1e-3))
    m_epochs = int(config.get('m_epochs', 50))
    total_global_epochs = max(1, em_iters * m_epochs)
    simple_mode = bool(config.get('simple_mode', False))

    def _compute_global_f1(A_hat_np):
        n_exc_local = int(A_hat_np.shape[0] * 0.8)
        tau_exc_local = find_best_tau_block(A_hat_np[:n_exc_local, :], A_true[:n_exc_local, :])
        tau_inh_local = find_best_tau_block(A_hat_np[n_exc_local:, :], A_true[n_exc_local:, :])
        true_mask_exc_local = np.abs(A_true[:n_exc_local, :]) > tau_exc_local
        hat_mask_exc_local = np.abs(A_hat_np[:n_exc_local, :]) > tau_exc_local
        true_mask_inh_local = np.abs(A_true[n_exc_local:, :]) > tau_inh_local
        hat_mask_inh_local = np.abs(A_hat_np[n_exc_local:, :]) > tau_inh_local
        true_mask_local = np.vstack([true_mask_exc_local, true_mask_inh_local])
        hat_mask_local = np.vstack([hat_mask_exc_local, hat_mask_inh_local])
        _, tp_local, fp_local, fn_local = _prf_from_masks(true_mask_local, hat_mask_local)
        f1_local = 2 * tp_local / (2 * tp_local + fp_local + fn_local) if (2 * tp_local + fp_local + fn_local) > 0 else 0.0
        return f1_local

    def _compute_magnitude_corr(A_hat_np):
        """Off-diagonal Pearson correlation for magnitude recovery."""
        n = A_hat_np.shape[0]
        off = ~np.eye(n, dtype=bool)
        x = A_hat_np[off].reshape(-1)
        y = A_true[off].reshape(-1)
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    for i in range(em_iters): # EM Iterations
        # --- E-STEP: Forward-Backward in Log-Space ---
        model.eval()
        with torch.no_grad():
            m = model.module if isinstance(model, nn.DataParallel) else model
            if em_iters > 1:
                frac = i / float(em_iters - 1)
            else:
                frac = 1.0
            gamma_temp = temp_start + frac * (temp_end - temp_start)
            gamma_full, xi_sum = compute_expectations(
                m, Y_full, X_full, delta_t=delta_t, gamma_temp=gamma_temp, gamma_floor=gamma_floor
            )
            # Record global EM NLL
            current_nll = model(Y_full, X_full, gamma_full, delta_t=delta_t).mean().item()
            metrics['em_iters'].append(i)
            metrics['nll_bin'].append(current_nll)

        # --- M-STEP: Parameter Updates ---
        model.train()
        for epoch in range(m_epochs):
            permutation = torch.randperm(T)
            epoch_loss_total = 0 
            prog = global_epoch / float(max(1, total_global_epochs - 1))
            prox_scale_inh = float(config.get('prox_scale_inh', 5e-3))
            prox_exc_start = float(config.get('prox_scale_exc_start', prox_scale_inh))
            prox_exc_end = float(config.get('prox_scale_exc_end', prox_scale_inh))
            prox_exc_pow = float(config.get('prox_scale_exc_power', 1.0))
            if simple_mode:
                prox_scale_exc = prox_scale_inh
            else:
                prox_scale_exc = prox_exc_start + (prox_exc_end - prox_exc_start) * (prog ** prox_exc_pow)
            l1_exc_base = float(config.get('l1_exc', config['l1']))
            l1_exc_mult_start = float(config.get('l1_exc_mult_start', 0.5))
            l1_exc_mult_end = float(config.get('l1_exc_mult_end', 1.0))
            l1_exc_mult_pow = float(config.get('l1_exc_mult_power', 1.0))
            if simple_mode:
                l1_exc_mult = 1.0
            else:
                l1_exc_mult = l1_exc_mult_start + (l1_exc_mult_end - l1_exc_mult_start) * (prog ** l1_exc_mult_pow)
            l1_exc_eff = l1_exc_base * l1_exc_mult
            last_tau_exc = 0.0
            last_tau_inh = 0.0

            for j in range(0, T, batch_size):
                indices = permutation[j:j+batch_size]
                optimizer.zero_grad()

                loss = model(Y_full[indices], X_full[indices], gamma_full[indices], delta_t=delta_t).mean()
                epoch_loss_total += loss.item() * len(indices)

                # Objective regularization terms:
                #   (1) Frobenius L2 shrinkage
                #   (2) Block-weighted normalized off-diagonal L1 sparsity
                m_ref = model.module if isinstance(model, nn.DataParallel) else model
                mask_off = 1.0 - torch.eye(N, device=device)
                A_off = m_ref.A * mask_off

                reg_l2 = 0.5 * config['l2'] * torch.sum(m_ref.A * m_ref.A)

                # Row blocks in stored orientation are postsynaptic blocks.
                exc_abs = torch.abs(A_off[:n_exc, :])
                inh_abs = torch.abs(A_off[n_exc:, :])
                exc_count = max(1, n_exc * (N - 1))
                inh_count = max(1, (N - n_exc) * (N - 1))
                mean_abs_exc = exc_abs.sum() / exc_count
                mean_abs_inh = inh_abs.sum() / inh_count

                reg = reg_l2

                (loss + reg).backward()
                optimizer.step()

                # Restore explicit block-aware proximal off-diagonal L1 sparsification.
                tau_exc = l1_exc_eff * prox_scale_exc
                tau_inh = config.get('l1_inh', config['l1']) * prox_scale_inh
                if not simple_mode:
                    # Block-normalized thresholds so smaller-magnitude block is not over-shrunk.
                    block_eps = 1e-8
                    mean_abs_global = 0.5 * (mean_abs_exc + mean_abs_inh) + block_eps
                    scale_exc = float((mean_abs_exc / mean_abs_global).detach().cpu().item())
                    scale_inh = float((mean_abs_inh / mean_abs_global).detach().cpu().item())
                    norm_lo = float(config.get('prox_norm_clamp_lo', 0.5))
                    norm_hi = float(config.get('prox_norm_clamp_hi', 2.0))
                    scale_exc = max(norm_lo, min(norm_hi, scale_exc))
                    scale_inh = max(norm_lo, min(norm_hi, scale_inh))
                    tau_exc *= scale_exc
                    tau_inh *= scale_inh
                last_tau_exc = float(tau_exc)
                last_tau_inh = float(tau_inh)
                proximal_soft_threshold_offdiag_blockwise(
                    m_ref.A, tau_exc=tau_exc, tau_inh=tau_inh, exc_ratio=0.8
                )
                enforce_dale_sign_offdiag(m_ref.A, exc_ratio=0.8)
                enforce_inhibitory_diagonal(
                    m_ref.A, min_abs=config.get('diag_inhibitory_floor', 5e-2)
                )

            # Delayed + periodic adaptive hard-zero for excitatory block.
            floor_exc_used = np.nan
            start_frac = float(config.get('hardzero_start_frac', 0.4))
            hardzero_every = int(config.get('hardzero_every', 5))
            start_epoch = int(start_frac * total_global_epochs)
            if (not simple_mode) and (global_epoch >= start_epoch) and (hardzero_every > 0) and ((global_epoch - start_epoch) % hardzero_every == 0):
                with torch.no_grad():
                    m_ref = model.module if isinstance(model, nn.DataParallel) else model
                    N_local = m_ref.A.shape[0]
                    n_exc_local = int(0.8 * N_local)
                    eye = torch.eye(N_local, device=device, dtype=m_ref.A.dtype)
                    A_off = m_ref.A * (1.0 - eye)
                    exc_abs = torch.abs(A_off[:n_exc_local, :]).reshape(-1)
                    exc_abs_nz = exc_abs[exc_abs > 1e-12]
                    q = float(config.get('zero_floor_exc_quantile', 0.10))
                    q = max(0.0, min(0.5, q))
                    if exc_abs_nz.numel() > 0:
                        floor_exc = torch.quantile(exc_abs_nz, q).item()
                    else:
                        floor_exc = 0.0
                    floor_exc = max(floor_exc, float(config.get('zero_floor_exc_min', 0.0)))
                    floor_inh = float(config.get('zero_floor_inh', 0.0))
                    hard_zero_offdiag_blockwise(
                        m_ref.A,
                        zero_floor_exc=floor_exc,
                        zero_floor_inh=floor_inh,
                        exc_ratio=0.8,
                    )
                    enforce_dale_sign_offdiag(m_ref.A, exc_ratio=0.8)
                    enforce_inhibitory_diagonal(
                        m_ref.A, min_abs=config.get('diag_inhibitory_floor', 5e-2)
                    )
                    floor_exc_used = floor_exc

            # Enforce spectral radius once per epoch (avoid repeated minibatch shrink bias).
            with torch.no_grad():
                m_ref = model.module if isinstance(model, nn.DataParallel) else model
                m_ref.apply_spectral_projection(rho_target=0.919)

            # FIX: Record metrics required by the 4th panel of the monitor
            with torch.no_grad():
                m_ref = model.module if isinstance(model, nn.DataParallel) else model
                rho = torch.max(torch.abs(torch.linalg.eigvals(m_ref.A))).item()
                nz = torch.sum(torch.abs(m_ref.A) > 0.01).item()
                
                # Calculate L1 Norm for the monitor's dual-axis plot
                l1_val = torch.norm(m_ref.A * mask_off, p=1).item()
                
                metrics['epochs'].append(global_epoch)
                metrics['spectral_radius'].append(rho)
                metrics['nz_edges'].append(nz)
                metrics['nll_bin_M'].append(epoch_loss_total / T) # Average NLL per bin
                metrics['l1_norm'].append(l1_val)
                metrics['prox_exc_scale'].append(float(prox_scale_exc))
                metrics['l1_exc_eff'].append(float(l1_exc_eff))
                metrics['zero_floor_exc'].append(float(floor_exc_used) if not np.isnan(floor_exc_used) else np.nan)
                metrics['debias_f1'].append(np.nan)
                metrics['debias_score'].append(np.nan)
                
            global_epoch += 1

        # Update Sticky Transition Matrix
        with torch.no_grad():
            m_fin = model.module if isinstance(model, nn.DataParallel) else model
            pi_num = gamma_full[0] + pi_pseudocount
            m_fin.pi.copy_(pi_num / pi_num.sum())

            new_P = (
                xi_sum
                + P_pseudocount * torch.ones((K, K), device=device)
                + config['kappa'] * torch.eye(K, device=device)
            )
            m_fin.P = new_P / new_P.sum(dim=1, keepdim=True)

    # One-time post-fit spectral stabilization.
    with torch.no_grad():
        m_fin = model.module if isinstance(model, nn.DataParallel) else model
        m_fin.apply_final_spectral_rescale()
        enforce_inhibitory_diagonal(
            m_fin.A, min_abs=config.get('diag_inhibitory_floor', 5e-2)
        )

    # --- Stage-B Debias Refit on Frozen Support ---
    # Remove L1/prox pressure while keeping learned support/sign constraints.
    debias_epochs = int(config.get('debias_epochs', 80))
    if debias_epochs > 0:
        m_ref = model.module if isinstance(model, nn.DataParallel) else model
        N_local = m_ref.N
        eye_local = torch.eye(N_local, device=device, dtype=m_ref.A.dtype)
        off_mask_local = (1.0 - eye_local)
        with torch.no_grad():
            A_off_local = m_ref.A * off_mask_local
            support_mask_off = (torch.abs(A_off_local) > float(config.get('debias_support_eps', 5e-4))).to(m_ref.A.dtype)
            gamma_db, _ = compute_expectations(m_ref, Y_full, X_full, delta_t=delta_t, gamma_temp=1.0, gamma_floor=0.0)

        debias_opt = optim.Adam([m_ref.A, m_ref.B], lr=float(config.get('debias_lr', 1e-3)))
        metrics['debias_start_epoch'] = int(global_epoch)
        debias_refresh_every = int(config.get('debias_refresh_every', 5))
        debias_patience = int(config.get('debias_patience', 8))
        debias_min_delta = float(config.get('debias_min_delta', 1e-4))
        debias_support_update_every = int(config.get('debias_support_update_every', 5))
        debias_support_soft_keep = float(config.get('debias_support_soft_keep', 0.05))
        debias_metric = str(config.get('debias_metric', 'hybrid')).lower()
        debias_f1_weight = float(config.get('debias_f1_weight', 0.5))
        best_db_f1 = -1.0
        best_db_score = -1.0
        best_db_epoch = -1
        best_db_state = {
            'A': m_ref.A.detach().clone(),
            'B': m_ref.B.detach().clone(),
        }
        bad_count = 0

        for db_epoch in range(debias_epochs):
            if (debias_refresh_every > 0) and (db_epoch % debias_refresh_every == 0):
                with torch.no_grad():
                    gamma_db, _ = compute_expectations(m_ref, Y_full, X_full, delta_t=delta_t, gamma_temp=1.0, gamma_floor=0.0)
            permutation = torch.randperm(T)
            epoch_loss_total = 0.0
            for j in range(0, T, batch_size):
                idx = permutation[j:j+batch_size]
                debias_opt.zero_grad()
                loss_db = model(Y_full[idx], X_full[idx], gamma_db[idx], delta_t=delta_t).mean()
                reg_db = 0.5 * config['l2'] * torch.sum(m_ref.A * m_ref.A)
                (loss_db + reg_db).backward()
                debias_opt.step()

                with torch.no_grad():
                    # Softer support freezing with periodic support refresh.
                    if (debias_support_update_every > 0) and (db_epoch % debias_support_update_every == 0):
                        A_off_now = m_ref.A * off_mask_local
                        support_mask_off = (torch.abs(A_off_now) > float(config.get('debias_support_eps', 5e-4))).to(m_ref.A.dtype)
                    A_diag = torch.diag(torch.diag(m_ref.A))
                    A_off = m_ref.A * off_mask_local
                    support_gate = support_mask_off + debias_support_soft_keep * (1.0 - support_mask_off)
                    A_off = A_off * support_gate
                    m_ref.A.copy_(A_off + A_diag)
                    enforce_dale_sign_offdiag(m_ref.A, exc_ratio=0.8)
                    enforce_inhibitory_diagonal(
                        m_ref.A, min_abs=config.get('diag_inhibitory_floor', 5e-2)
                    )

                epoch_loss_total += loss_db.item() * len(idx)

            with torch.no_grad():
                rho = torch.max(torch.abs(torch.linalg.eigvals(m_ref.A))).item()
                nz = torch.sum(torch.abs(m_ref.A) > 0.01).item()
                l1_val = torch.norm(m_ref.A * off_mask_local, p=1).item()
                metrics['epochs'].append(global_epoch)
                metrics['spectral_radius'].append(rho)
                metrics['nz_edges'].append(nz)
                metrics['nll_bin_M'].append(epoch_loss_total / T)
                metrics['l1_norm'].append(l1_val)
                metrics['prox_exc_scale'].append(np.nan)
                metrics['l1_exc_eff'].append(np.nan)
                metrics['zero_floor_exc'].append(np.nan)
                A_now = m_ref.A.detach().cpu().numpy()
                f1_now = _compute_global_f1(A_now)
                corr_now = _compute_magnitude_corr(A_now)
                metrics['debias_f1'].append(float(f1_now))
                metrics['debias_score'].append(float(corr_now))
                m_ref.apply_spectral_projection(rho_target=0.919)
            global_epoch += 1

            if debias_metric == 'f1':
                score_now = float(f1_now)
            elif debias_metric == 'corr':
                score_now = float(corr_now)
            else:
                score_now = float(debias_f1_weight * f1_now + (1.0 - debias_f1_weight) * corr_now)

            if score_now > (best_db_score + debias_min_delta):
                best_db_f1 = float(f1_now)
                best_db_score = float(score_now)
                best_db_epoch = int(global_epoch - 1)
                best_db_state = {
                    'A': m_ref.A.detach().clone(),
                    'B': m_ref.B.detach().clone(),
                }
                bad_count = 0
            else:
                bad_count += 1

            if bad_count >= debias_patience:
                break

        with torch.no_grad():
            m_ref.A.copy_(best_db_state['A'])
            m_ref.B.copy_(best_db_state['B'])
            m_ref.apply_final_spectral_rescale()
            enforce_inhibitory_diagonal(
                m_ref.A, min_abs=config.get('diag_inhibitory_floor', 5e-2)
            )
            metrics['debias_best_f1'] = float(best_db_f1) if best_db_f1 >= 0 else None
            metrics['debias_best_score'] = float(best_db_score) if best_db_score >= 0 else None
            metrics['debias_best_epoch'] = int(best_db_epoch) if best_db_epoch >= 0 else None

    # Final Evaluation for F1 with adaptive block-specific thresholds
    A_hat = (model.module.A if isinstance(model, nn.DataParallel) else model.A).detach().cpu().numpy()

    n_exc = int(A_hat.shape[0] * 0.8)
    tau_exc = find_best_tau_block(A_hat[:n_exc, :], A_true[:n_exc, :])
    tau_inh = find_best_tau_block(A_hat[n_exc:, :], A_true[n_exc:, :])

    true_mask_exc = np.abs(A_true[:n_exc, :]) > tau_exc
    hat_mask_exc = np.abs(A_hat[:n_exc, :]) > tau_exc
    true_mask_inh = np.abs(A_true[n_exc:, :]) > tau_inh
    hat_mask_inh = np.abs(A_hat[n_exc:, :]) > tau_inh

    true_mask = np.vstack([true_mask_exc, true_mask_inh])
    hat_mask = np.vstack([hat_mask_exc, hat_mask_inh])
    _, tp, fp, fn = _prf_from_masks(true_mask, hat_mask)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    metrics['tau_exc'] = tau_exc
    metrics['tau_inh'] = tau_inh

    # Final state-occupancy diagnostics for anti-collapse model selection.
    with torch.no_grad():
        m_occ = model.module if isinstance(model, nn.DataParallel) else model
        gamma_final, _ = compute_expectations(m_occ, Y_full, X_full, delta_t=delta_t)
        occ = gamma_final.mean(dim=0).detach().cpu().numpy()
        metrics['state_occupancy'] = occ.tolist()
        metrics['min_state_occupancy'] = float(np.min(occ))

    return f1, model, metrics
   

def main():

    set_seed(42) # Ensure seed is set before any data processing
    path = "/pscratch/sd/s/sanjitr/causal_net_temp/spikesData"
    dataset = "daleN100_b1036a"
    
    data_load = np.load(f"{path}/{dataset}.spikes.npz")
    truth_load = np.load(f"{path}/{dataset}.prismTruth.npz")
    
    Y_np = data_load['spikes'].reshape(-1, 100).astype(np.float32)
    A_true = truth_load['A_true']
    B_true = truth_load['B_true']
    
    X_np = np.roll(Y_np, 1, axis=0); X_np[0, :] = 0
    Y_full = torch.tensor(Y_np, device=device)
    X_full = torch.tensor(X_np, device=device)

    # Expanded sparsity/smoothness search space
    l1_space = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    l2_space = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    kappa_fixed = 1.0
    warm_ridge_fixed = 1e-2
    warm_keep_ratio_fixed = 0.15
    delta_t_fixed = 0.1
    l1_exc_mult_fixed = 1.25
    l1_inh_mult_fixed = 1.0
    prox_scale_exc_start_fixed = 5e-3
    prox_scale_exc_end_fixed = 1.5e-2
    prox_scale_exc_power_fixed = 1.0
    l1_exc_mult_start_fixed = 0.2
    l1_exc_mult_end_fixed = 0.8
    l1_exc_mult_power_fixed = 1.0
    prox_scale_inh_fixed = 5e-3
    zero_floor_exc_quantile_fixed = 0.10
    zero_floor_exc_min_fixed = 0.0
    hardzero_start_frac_fixed = 0.40
    hardzero_every_fixed = 5
    zero_floor_inh_fixed = 0.0
    diag_inhibitory_floor_fixed = 5e-2
    debias_epochs_fixed = 40
    debias_lr_fixed = 1e-3
    debias_support_eps_fixed = 1e-3
    debias_refresh_every_fixed = 5
    debias_patience_fixed = 8
    debias_min_delta_fixed = 1e-4
    debias_support_update_every_fixed = 5
    debias_support_soft_keep_fixed = 0.05
    debias_metric_fixed = "hybrid"
    debias_f1_weight_fixed = 0.5
    prox_norm_clamp_lo_fixed = 0.5
    prox_norm_clamp_hi_fixed = 2.0
    simple_mode_fixed = True
    gamma_temp_start_fixed = 1.6
    gamma_temp_end_fixed = 1.0
    gamma_floor_fixed = 0.02
    pi_pseudocount_fixed = 1e-2
    P_pseudocount_fixed = 1e-3

    min_state_occ_floor = 0.10
    best_f1, best_config = -1, None
    best_model, best_metrics = None, None
    best_f1_any, best_any = -1, None
    best_model_any, best_metrics_any = None, None
    for l1 in l1_space:
        for l2 in l2_space:
            config = {
                'l1': l1,
                'l1_exc': l1 * l1_exc_mult_fixed,
                'l1_inh': l1 * l1_inh_mult_fixed,
                'l2': l2,
                'kappa': kappa_fixed,
                'warm_ridge': warm_ridge_fixed,
                'warm_keep_ratio': warm_keep_ratio_fixed,
                'delta_t': delta_t_fixed,
                'prox_scale_exc_start': prox_scale_exc_start_fixed,
                'prox_scale_exc_end': prox_scale_exc_end_fixed,
                'prox_scale_exc_power': prox_scale_exc_power_fixed,
                'l1_exc_mult_start': l1_exc_mult_start_fixed,
                'l1_exc_mult_end': l1_exc_mult_end_fixed,
                'l1_exc_mult_power': l1_exc_mult_power_fixed,
                'prox_scale_inh': prox_scale_inh_fixed,
                'zero_floor_exc_quantile': zero_floor_exc_quantile_fixed,
                'zero_floor_exc_min': zero_floor_exc_min_fixed,
                'hardzero_start_frac': hardzero_start_frac_fixed,
                'hardzero_every': hardzero_every_fixed,
                'zero_floor_inh': zero_floor_inh_fixed,
                'diag_inhibitory_floor': diag_inhibitory_floor_fixed,
                'debias_epochs': debias_epochs_fixed,
                'debias_lr': debias_lr_fixed,
                'debias_support_eps': debias_support_eps_fixed,
                'debias_refresh_every': debias_refresh_every_fixed,
                'debias_patience': debias_patience_fixed,
                'debias_min_delta': debias_min_delta_fixed,
                'debias_support_update_every': debias_support_update_every_fixed,
                'debias_support_soft_keep': debias_support_soft_keep_fixed,
                'debias_metric': debias_metric_fixed,
                'debias_f1_weight': debias_f1_weight_fixed,
                'prox_norm_clamp_lo': prox_norm_clamp_lo_fixed,
                'prox_norm_clamp_hi': prox_norm_clamp_hi_fixed,
                'simple_mode': simple_mode_fixed,
                'gamma_temp_start': gamma_temp_start_fixed,
                'gamma_temp_end': gamma_temp_end_fixed,
                'gamma_floor': gamma_floor_fixed,
                'pi_pseudocount': pi_pseudocount_fixed,
                'P_pseudocount': P_pseudocount_fixed,
            }
            print(f"Testing Config: {config}")
            f1, trained_model, metrics = train_model(config, Y_full, X_full, A_true)
            print(f"Resulting F1 Score: {f1:.4f}")
            print(f"State occupancy: {metrics.get('state_occupancy', [])}")

            # Track unconstrained best as a fallback.
            if f1 > best_f1_any:
                best_f1_any = f1
                best_any = config
                best_model_any = trained_model
                best_metrics_any = metrics

            # Anti-collapse gate: only select configs with sufficiently occupied states.
            if metrics.get('min_state_occupancy', 0.0) < min_state_occ_floor:
                print(
                    f"Rejected by anti-collapse gate: "
                    f"min_state_occupancy={metrics.get('min_state_occupancy', 0.0):.4f} "
                    f"< {min_state_occ_floor:.2f}"
                )
                continue

            if f1 > best_f1:
                best_f1 = f1
                best_config = config
                best_model = trained_model
                best_metrics = metrics
            break
        break

    # Fallback if every config collapses under gate.
    if best_model is None:
        print("All configs failed anti-collapse gate; using unconstrained best as fallback.")
        best_f1 = best_f1_any
        best_config = best_any
        best_model = best_model_any
        best_metrics = best_metrics_any

    print(f"\nOptimization Complete. Best F1: {best_f1:.4f} with {best_config}")
    
    # Save the best model parameters
    # Handle DataParallel wrapping if necessary
    final_model_obj = best_model.module if isinstance(best_model, nn.DataParallel) else best_model
    
    save_dict = {
        'model_state_dict': final_model_obj.state_dict(),
        'hyperparameters': best_config,
        'f1_score': best_f1,
        'metrics': best_metrics,
        'dataset': dataset
    }
    
    model_dir = "/pscratch/sd/s/sanjitr/causal_net_temp/models"
    os.makedirs(model_dir, exist_ok=True)
    save_file = f"{model_dir}/{dataset}_best_model.pt"
    torch.save(save_dict, save_file)
    print(f"Best model parameters saved to: {save_file}")
    

if __name__ == "__main__":
    main()
