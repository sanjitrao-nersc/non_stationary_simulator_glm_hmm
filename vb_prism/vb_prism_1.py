import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

# -------------------------
# Reproducibility and device
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# PRISM Biological Constraints
# -------------------------
def proximal_soft_threshold_offdiag(A, tau_exc, tau_inh, exc_ratio=0.8):
    """Adaptive L1 sparsification using PRISM tau-controller logic."""
    with torch.no_grad():
        n = A.shape[0]
        n_exc = int(n * exc_ratio)
        eye = torch.eye(n, device=A.device)
        
        # Isolate off-diagonal
        off = A * (1.0 - eye)
        
        # Apply row-blockwise thresholds (Excitatory vs Inhibitory rows)
        shrunk = off.clone()
        shrunk[:n_exc, :] = torch.sign(off[:n_exc, :]) * torch.clamp(torch.abs(off[:n_exc, :]) - float(tau_exc), min=0.0)
        shrunk[n_exc:, :] = torch.sign(off[n_exc:, :]) * torch.clamp(torch.abs(off[n_exc:, :]) - float(tau_inh), min=0.0)
        
        diag = torch.diag(torch.diag(A))
        A.copy_(shrunk + diag)

def enforce_dale_sign_prism(A, exc_ratio=0.8):
    """Enforce Dale's Law: Excitatory rows >= 0, Inhibitory rows <= 0[cite: 4309]."""
    with torch.no_grad():
        n = A.shape[0]
        n_exc = int(n * exc_ratio)
        eye = torch.eye(n, device=A.device)
        off = A * (1.0 - eye)
        
        # PRISM logic: Dale applies to rows (postsynaptic influence)
        off[:n_exc, :] = torch.clamp(off[:n_exc, :], min=0.0)
        off[n_exc:, :] = torch.clamp(off[n_exc:, :], max=0.0)
        
        diag = torch.diag(torch.diag(A))
        A.copy_(off + diag)

def enforce_spectral_stability(A, rho_max=0.92, rho_target=0.90):
    """Project A onto a stable spectral radius[cite: 4310, 4326]."""
    with torch.no_grad():
        eigvals = torch.linalg.eigvals(A)
        rho = torch.max(torch.abs(eigvals))
        if rho >= rho_max:
            A.mul_(rho_target / rho)

def enforce_self_inhibition(A, min_abs=0.05):
    """Force A_ii <= -0.05[cite: 4311]."""
    with torch.no_grad():
        idx = torch.arange(A.shape[0], device=A.device)
        d = A[idx, idx]
        A[idx, idx] = -torch.clamp(torch.abs(d), min=float(min_abs))

# -------------------------
# Evaluation helpers
# -------------------------
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


# -------------------------
# Fix 2: Ridge Warm Start for A
# -------------------------
def warm_start_connectivity(A_param, X_full, Y_full, ridge=1e-2, keep_ratio=0.15, exc_ratio=0.8):
    """Ridge regression warm start; sparsifies and enforces constraints before EM begins."""
    with torch.no_grad():
        n = A_param.shape[0]
        xtx = X_full.T @ X_full
        xty = X_full.T @ Y_full
        A_ls = torch.linalg.solve(xtx + float(ridge) * torch.eye(n, device=X_full.device), xty)
        # A_ls is [N, N] (pre -> post); transpose to (post, pre) to match A_param layout

        off_mask = (1.0 - torch.eye(n, device=X_full.device)).bool()
        off_abs = torch.abs(A_ls[off_mask])
        q = max(0.0, min(1.0, 1.0 - float(keep_ratio)))
        thr = torch.quantile(off_abs, q) if off_abs.numel() > 0 else torch.tensor(0.0, device=X_full.device)
        A_sparse = torch.where(torch.abs(A_ls) >= thr, A_ls, torch.zeros_like(A_ls))

        idx = torch.arange(n, device=X_full.device)
        A_sparse[idx, idx] = -0.05
        A_sparse = torch.clamp(A_sparse, min=-0.5, max=0.5)

        A_param.copy_(A_sparse.T)
        enforce_dale_sign_prism(A_param, exc_ratio=exc_ratio)
        enforce_self_inhibition(A_param, min_abs=5e-2)


# -------------------------
# Model Core: Variational PRISM
# -------------------------
class VariationalPrismNeuralGLM(nn.Module):
    def __init__(self, n_states, n_neurons):
        super().__init__()
        self.K = n_states
        self.N = n_neurons
        
        # Shared Global Connectivity (The A Matrix)
        self.A = nn.Parameter(torch.randn(self.N, self.N) * 0.05)
        
        # State-specific Baselines (B_k)
        self.B = nn.Parameter(torch.zeros(self.K, self.N))
        
        # Variational parameters for transition matrix and initial state
        self.register_buffer("P", torch.eye(self.K))
        self.register_buffer("pi", torch.full((self.K,), 1.0 / self.K))

    def compute_log_emissions(self, Y, X, delta_t=1.0):
        """
        Local Variational approximation of Poisson NLL[cite: 3683, 4322].
        Returns [T, K] tensor.
        """
        T = Y.shape[0]
        # X @ A.T provides the recurrent input [T, N]
        recurrent_term = X @ self.A.T # [T, N]
        
        # Add baseline B per state [T, K, N]
        eta = recurrent_term.unsqueeze(1) + self.B.unsqueeze(0)
        eta = torch.clamp(eta, max=10.0) # Stability clip
        
        # Rate mu = exp(eta) * dt
        mu = torch.exp(eta) * float(delta_t)
        
        # Poisson Log-Likelihood: Y*log(mu) - mu
        # [T, K, N] summed over N -> [T, K]
        log_dt = math.log(delta_t)
        log_emissions = (Y.unsqueeze(1) * (eta + log_dt) - mu).sum(dim=2)
        return log_emissions

# -------------------------
# VB-SSSM E-Step: Forward-Backward
# -------------------------
def run_variational_e_step(model, Y, X, delta_t=1.0):
    """Forward-Backward in log-space[cite: 3664, 4337]."""
    T, K = Y.shape[0], model.K
    dev = Y.device
    
    log_emissions = model.compute_log_emissions(Y, X, delta_t)
    log_P = torch.log(model.P.clamp(min=1e-10))
    log_pi = torch.log(model.pi.clamp(min=1e-10))

    # Forward pass
    log_alpha = torch.zeros((T, K), device=dev)
    log_alpha[0] = log_emissions[0] + log_pi
    for t in range(1, T):
        log_alpha[t] = log_emissions[t] + torch.logsumexp(log_alpha[t-1].unsqueeze(1) + log_P, dim=0)

    # Backward pass
    log_beta = torch.zeros((T, K), device=dev)
    for t in range(T - 2, -1, -1):
        log_beta[t] = torch.logsumexp(log_P + log_emissions[t+1] + log_beta[t+1], dim=1)

    # Compute Responsibilities (Gamma)
    log_gamma = log_alpha + log_beta
    log_gamma -= torch.logsumexp(log_gamma, dim=1, keepdim=True)
    gamma = torch.exp(log_gamma)

    # Compute expected transitions (Xi) [cite: 4331]
    # log_xi[t, j, k] = alpha[t,j] + P[j,k] + emission[t+1,k] + beta[t+1,k]
    log_xi = log_alpha[:-1].unsqueeze(2) + log_P.unsqueeze(0) + \
             log_emissions[1:].unsqueeze(1) + log_beta[1:].unsqueeze(1)
    # Normalize xi per time step
    log_xi -= torch.logsumexp(log_xi.view(T-1, -1), dim=1).view(T-1, 1, 1)
    xi_sum = torch.exp(log_xi).sum(dim=0)
    
    return gamma, xi_sum, log_gamma

# -------------------------
# VB-SSSM M-Step / Training
# -------------------------
def train_vb_prism(config, Y_full, X_full, A_true, device):
    N = A_true.shape[0]
    K = config.get("n_states", 3)
    model = VariationalPrismNeuralGLM(K, N).to(device)

    with torch.no_grad():
        # Fix 2: Ridge warm start for A
        warm_start_connectivity(
            model.A,
            X_full,
            Y_full,
            ridge=float(config.get("warm_ridge", 1e-2)),
            keep_ratio=float(config.get("warm_keep_ratio", 0.15)),
        )

        # Fix 1: Initialize B_k from log mean rates + symmetry-breaking noise
        mean_rates = Y_full.mean(dim=0).clamp(min=1e-4)
        log_mean = torch.log(mean_rates)
        for k in range(K):
            model.B[k].copy_(log_mean + torch.randn(N, device=device) * 0.2)

        # Fix 3: Start with weak sticky prior so early EM iters are data-driven.
        # p_self=0.93 is correct for a converged model but locks state collapse at init.
        p_self = 0.80 if K == 2 else 0.70
        p_cross = (1.0 - p_self) / max(1, K - 1)
        P_init = torch.full((K, K), p_cross, device=device, dtype=model.P.dtype)
        P_init.fill_diagonal_(p_self)
        model.P.copy_(P_init)

    lr = config.get("lr", 5e-3)
    # Fix 1: B gets 5× lower lr so it can't absorb all signal before A has a chance to learn.
    optimizer = optim.Adam([
        {'params': [model.A], 'lr': lr},
        {'params': [model.B], 'lr': lr / 5.0},
    ])

    T = Y_full.shape[0]
    em_iters = config.get("em_iters", 25)
    m_epochs = config.get("m_epochs", 30)
    delta_t = config.get("delta_t", 0.01)

    # Fix 2: prox_scale default raised to 0.2 so tau ~ l1*0.2 ~ 2e-3, meaningful for
    # warm-start weight magnitudes (0.01-0.3). Old default 5e-3 gave tau=5e-5, a no-op.
    tau_exc = config.get("l1", 1e-2) * config.get("prox_scale", 0.2)
    tau_inh = tau_exc

    for em in range(em_iters):
        # 1. Variational E-Step
        model.eval()
        with torch.no_grad():
            gamma, xi_sum, _ = run_variational_e_step(model, Y_full, X_full, delta_t)
            
        # 2. Variational M-Step (Optimization of A and B)
        model.train()
        for epoch in range(m_epochs):
            optimizer.zero_grad()
            
            # ELBO Likelihood Term [cite: 4322]
            log_emissions = model.compute_log_emissions(Y_full, X_full, delta_t)
            loss = -(gamma * log_emissions).sum() # Negative Weighted Log-Likelihood
            
            # L2 Prior on A
            reg_l2 = 0.5 * config.get("l2", 1e-3) * torch.sum(model.A**2)
            (loss + reg_l2).backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Apply PRISM Projections [cite: 4324, 4325, 4326]
            with torch.no_grad():
                proximal_soft_threshold_offdiag(model.A, tau_exc, tau_inh)
                enforce_dale_sign_prism(model.A)
                enforce_self_inhibition(model.A)
                if epoch % 3 == 0:
                    enforce_spectral_stability(model.A)

        # 3. Update Transition Matrix P and Pi [cite: 4320, 4321]
        with torch.no_grad():
            # Transitions
            P_new = xi_sum + config.get("P_pseudocount", 1e-3)
            P_new += torch.eye(K, device=device) * 0.1 # Sticky boost
            model.P.copy_(P_new / P_new.sum(dim=1, keepdim=True))
            
            # Init dist
            pi_new = gamma[0] + config.get("pi_pseudocount", 1e-2)
            model.pi.copy_(pi_new / pi_new.sum())

    # Final Evaluation: adaptive threshold F1 (block-aware, matches ashwood_ns_10)
    n_exc = int(0.8 * N)
    A_hat = model.A.detach().cpu().numpy()

    tau_exc = _find_best_tau(A_hat[:n_exc, :], A_true[:n_exc, :])
    tau_inh = _find_best_tau(A_hat[n_exc:, :], A_true[n_exc:, :])

    mask_true = np.vstack([
        np.abs(A_true[:n_exc, :]) > tau_exc,
        np.abs(A_true[n_exc:, :]) > tau_inh,
    ])
    mask_hat = np.vstack([
        np.abs(A_hat[:n_exc, :]) > tau_exc,
        np.abs(A_hat[n_exc:, :]) > tau_inh,
    ])
    tp = int(np.sum(mask_true & mask_hat))
    fp = int(np.sum((~mask_true) & mask_hat))
    fn = int(np.sum(mask_true & (~mask_hat)))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    print(f"EM Iter {em} Complete. F1={f1:.4f}  TP={tp} FP={fp} FN={fn}  tau_exc={tau_exc:.4f} tau_inh={tau_inh:.4f}")
    return f1, model

# -------------------------
# Main Entry Point
# -------------------------
def main():
    set_seed(42)
    # Path/Dataset variables matching ashwood_ns_10.py
    path = "/pscratch/sd/s/sanjitr/causal_net_temp/spikesData"
    dataset = "daleN100_f7d754_ce2202"
    
    try:
        data_load = np.load(f"{path}/{dataset}.spikes.npz")
        truth_load = np.load(f"{path}/{dataset}.prismTruth.npz")
        A_true = truth_load["A_true"]
        Y_np = data_load["spikes"].astype(np.float32)
    except FileNotFoundError:
        print("Data files not found. Using synthetic placeholder for demo.")
        N, T = 100, 5000
        A_true = np.random.randn(N, N) * 0.05
        Y_np = np.random.poisson(0.1, (T, N)).astype(np.float32)

    N = A_true.shape[0]
    Y_torch = torch.from_numpy(Y_np).to(device)
    # Recurrent input is the lagged activity (Lag-1)
    X_torch = torch.zeros_like(Y_torch)
    X_torch[1:] = Y_torch[:-1]

    config = {
        "n_states": 3,
        "lr": 5e-3,
        "l1": 1e-2,
        "l2": 1e-4,
        "em_iters": 20,
        "m_epochs": 30,
        "delta_t": 0.01,
        "prox_scale": 0.2,
        "P_pseudocount": 1e-3,
        "pi_pseudocount": 1e-2
    }

    print(f"Starting Variational PRISM Training (N={N}, K={config['n_states']})...")
    f1, model = train_vb_prism(config, Y_torch, X_torch, A_true, device)
    
    models_dir = "/pscratch/sd/s/sanjitr/causal_net_temp/models"
    os.makedirs(models_dir, exist_ok=True)
    save_path = f"{models_dir}/{dataset}_vb_prism.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Training finished. Best F1: {f1:.4f}. Model saved to {save_path}")

if __name__ == "__main__":
    main()