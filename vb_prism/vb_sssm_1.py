import os
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.special import digamma, gammaln

# -------------------------
# Device and Seed
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------
# Helper Functions
# -------------------------
def tridiagonal_precision(M, device):
    """
    Constructs the tridiagonal Lambda for the random-walk firing-rate prior
    (paper §2.2): p(x^n) ∝ ∏_m exp(-β^n/2 * ((x_m - μ_m) - (x_{m-1} - μ_{m-1}))^2).
    """
    diag = 2.0 * torch.ones(M, device=device)
    diag[0], diag[-1] = 1.0, 1.0
    off_diag = -1.0 * torch.ones(M - 1, device=device)
    Lambda = torch.diag(diag) + torch.diag(off_diag, -1) + torch.diag(off_diag, 1)
    return Lambda

def log2cosh(x):
    """
    Numerically stable log(2 cosh(x)) = |x| + log(1 + exp(-2|x|)).
    Avoids float32 overflow: cosh(x) overflows for |x| > ~88.
    """
    ax = torch.abs(x)
    return ax + torch.log1p(torch.exp(-2.0 * ax))

# -------------------------
# VB-SSSM Model Core
# -------------------------
class VB_SSSM(nn.Module):
    def __init__(self, n_states, n_bins, device):
        super().__init__()
        self.N = n_states
        self.M = n_bins
        self.device = device

        # Fixed prior hyperparameters — preserved across M-steps (paper §2.2, Eq. 8-9)
        self.gamma_pi_prior = torch.ones(self.N, device=device)
        self.gamma_a_prior = (10.0 * torch.eye(self.N, device=device)
                              + 2.5 * (1 - torch.eye(self.N, device=device)))

        # Variational Dirichlet parameters (updated each M-step)
        self.gamma_pi = self.gamma_pi_prior.clone()
        self.gamma_a  = self.gamma_a_prior.clone()

        # Prior mean μ^n for the random-walk prior (Eq. 7).  Fixed at 0 after init.
        # NOT updated to track the posterior mean — doing so collapses diff→0 in
        # the β update, forcing β_new = M·β/(M-1) > β every step → divergence.
        # Small Gaussian noise breaks the perfect state symmetry at init only.
        self.mu_prior = torch.randn((self.N, self.M), device=device) * 0.1
        self.beta_n   = torch.ones(self.N, device=device)

        # Local variational parameters ξ^n_m (Eq. 12)
        self.xi_nm = torch.ones((self.N, self.M), device=device)

        # Posterior quantities stored between E-step and M-step
        self.mu_hat    = torch.zeros((self.N, self.M), device=device)  # <x^n>
        self.inv_W_diag = torch.ones((self.N, self.M), device=device)  # diag((W^n)^{-1})
        self.trace_Lambda_Winv = torch.ones(self.N, device=device)     # Tr[Λ (W^n)^{-1}]

        # <z_m^n> — soft state assignments
        self.gamma_nm = torch.full((self.N, self.M), 1.0 / self.N, device=device)

    def get_log_pi(self):
        return torch.digamma(self.gamma_pi) - torch.digamma(self.gamma_pi.sum())

    def get_log_A(self):
        return torch.digamma(self.gamma_a) - torch.digamma(self.gamma_a.sum(dim=1, keepdim=True))

# -------------------------
# E-Step: Variational Inference
# -------------------------
def variational_e_step(model, eta_hat, Lambda, C, temperature=1.0):
    """
    VB-E step: updates q(x^n) and q(z) (paper §3.2, Eqs. 14-15).

    eta_hat     : (M,) tensor — coarse-bin spike indicator sums η̂_m = Σ_u η_{(m-1)C+u},
                  each η_k = +1 (spike) or -1 (no spike), so η̂_m ∈ [-C*N_neurons, +C*N_neurons].
    temperature : scalar in (0, 1].  Multiplies log_b before forward-backward (deterministic
                  annealing, paper §3.2).  Ramp from 0.1 → 1.0 over the first ~20 iterations.
    """
    N, M = model.N, model.M
    inv_W_diag = torch.zeros((N, M), device=model.device)

    eye_M = torch.eye(M, device=model.device)

    for n in range(N):
        # Bug 1 fix: L^n_{mm} = <z_m^n> * tanh(ξ_m^n) / ξ_m^n  (§3.2, not /2ξ)
        L_diag = model.gamma_nm[n] * (torch.tanh(model.xi_nm[n]) / (model.xi_nm[n] + 1e-10))
        # ε·I regularization: Lambda's null space (constant vectors) makes W_n singular
        # whenever a state is being pruned (L_diag → 0).  The small ridge makes W_n
        # positive-definite without materially affecting active states.
        W_n = C * torch.diag(L_diag) + model.beta_n[n] * Lambda + 1e-4 * eye_M

        # Bug 2 fix: w^n has mth component <z_m^n>·η̂_m  (§3.2, below Eq. 14)
        rhs = model.gamma_nm[n] * eta_hat + model.beta_n[n] * torch.mv(Lambda, model.mu_prior[n])
        model.mu_hat[n] = torch.linalg.solve(W_n, rhs)

        W_inv = torch.inverse(W_n)
        inv_W_diag[n] = torch.diag(W_inv)
        # Store Tr[Λ (W^n)^{-1}] for β^n update in M-step (Eq. 19)
        model.trace_Lambda_Winv[n] = torch.trace(Lambda @ W_inv)

    # ξ update: (ξ_m^n)^2 = <(x_m^n)^2> = μ̂_m^2 + (W^n)^{-1}_{mm}  (Eq. 19)
    model.xi_nm = torch.sqrt(model.mu_hat**2 + inv_W_diag)
    model.inv_W_diag = inv_W_diag

    # Bug 3 fix: b̂_m^n = η̂_m <x_m^n> - C log(2 cosh ξ_m^n)
    # The middle quadratic correction in Eq. 15 vanishes after substituting ξ = sqrt(<x^2>).
    # Use log2cosh() — torch.cosh overflows float32 for |ξ| > ~88.
    log_b = eta_hat.unsqueeze(0) * model.mu_hat - C * log2cosh(model.xi_nm)
    # Deterministic annealing: flatten the likelihood landscape early in training
    # so the chain can escape uniform-assignment local optima (paper §3.2).
    log_b = log_b * temperature

    log_A  = model.get_log_A()
    log_pi = model.get_log_pi()

    # Scaled forward-backward (paper §3.2).
    # Per-step log-normalisation prevents float32 underflow over long sequences
    # (M ≈ 20,000 steps × log_b ≈ -650/bin → log_alpha[M] ≈ -1.3e7 without scaling).
    # The per-step normalisation constants sum to log p(y) (the ELBO likelihood term).
    log_norm  = torch.tensor(0.0, device=model.device)
    log_alpha = torch.zeros((M, N), device=model.device)
    log_alpha[0] = log_pi + log_b[:, 0]
    c0 = torch.logsumexp(log_alpha[0], dim=0)
    log_alpha[0] -= c0
    log_norm += c0
    for m in range(1, M):
        log_alpha[m] = log_b[:, m] + torch.logsumexp(log_alpha[m-1].unsqueeze(1) + log_A, dim=0)
        cm = torch.logsumexp(log_alpha[m], dim=0)
        log_alpha[m] -= cm
        log_norm += cm

    log_beta = torch.zeros((M, N), device=model.device)
    for m in range(M-2, -1, -1):
        log_beta[m] = torch.logsumexp(log_A + log_b[:, m+1] + log_beta[m+1], dim=1)
        log_beta[m] -= torch.logsumexp(log_beta[m], dim=0)

    log_gamma = log_alpha + log_beta
    log_gamma -= torch.logsumexp(log_gamma, dim=1, keepdim=True)
    model.gamma_nm = torch.exp(log_gamma).T  # shape (N, M)

    # Expected pairwise transitions Σ_m <z_m^n z_{m+1}^k>  (Eq. 17)
    log_xi_trans = (log_alpha[:-1].unsqueeze(2)
                    + log_A.unsqueeze(0)
                    + log_b[:, 1:].T.unsqueeze(1)
                    + log_beta[1:].unsqueeze(1))
    log_xi_trans -= torch.logsumexp(log_xi_trans.view(M-1, -1), dim=1).view(M-1, 1, 1)
    expected_transitions = torch.exp(log_xi_trans).sum(dim=0)

    return expected_transitions, log_norm.item()

# -------------------------
# M-Step: Parameter Update
# -------------------------
def variational_m_step(model, expected_transitions, Lambda):
    """
    EM + VB-M step: updates γ̂, β^n, and μ^n (paper §3.2-3.3, Eqs. 16-17, 19).
    """
    # Bug 5 fix: add the fixed prior hyperparameters (not hard-coded 1.0)
    model.gamma_pi = model.gamma_pi_prior + model.gamma_nm[:, 0]
    model.gamma_a  = model.gamma_a_prior  + expected_transitions

    # β_max = 100: smoothness precision beyond which the trajectory is effectively frozen.
    # mu_prior is intentionally NOT updated here.
    #
    # C_REG fixes the null-mode divergence that survives even with fixed mu_prior:
    # Λ has eigenvalue 0 for any constant vector, so diff^T Λ diff = 0 when
    # diff = (mu_hat - mu_prior) is approximately flat — which it is once mu_hat
    # converges to a stationary trajectory.  Adding C_REG·‖diff‖² to the denominator
    # captures constant-offset deviations that Λ misses.  Fixed point:
    #   β* ≈ 1 / (C_REG · (‖diff‖/√M)²)
    # With C_REG=0.01 and ‖diff‖/√M ≈ 1 (typical logit-domain offset), β* ≈ 100.
    BETA_MAX = 100.0
    C_REG    = 0.01
    for n in range(model.N):
        diff      = model.mu_hat[n] - model.mu_prior[n]
        quad_form = torch.dot(diff, torch.mv(Lambda, diff))
        l2_reg    = C_REG * torch.dot(diff, diff)
        model.beta_n[n] = torch.clamp(
            model.M / (model.trace_Lambda_Winv[n] + quad_form + l2_reg + 1e-6),
            max=BETA_MAX
        )

# -------------------------
# Checkpoint helpers
# -------------------------

def save_checkpoint(model, path, metrics=None):
    torch.save({
        'N': model.N, 'M': model.M,
        'gamma_pi':       model.gamma_pi,
        'gamma_a':        model.gamma_a,
        'gamma_pi_prior': model.gamma_pi_prior,
        'gamma_a_prior':  model.gamma_a_prior,
        'mu_prior':       model.mu_prior,
        'mu_hat':         model.mu_hat,
        'beta_n':         model.beta_n,
        'xi_nm':          model.xi_nm,
        'gamma_nm':       model.gamma_nm,
        'inv_W_diag':     model.inv_W_diag,
        'trace_Lambda_Winv': model.trace_Lambda_Winv,
        'metrics':        metrics or {},
    }, path)

def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    model = VB_SSSM(ck['N'], ck['M'], device)
    for key in ('gamma_pi', 'gamma_a', 'gamma_pi_prior', 'gamma_a_prior',
                'mu_prior', 'mu_hat', 'beta_n', 'xi_nm', 'gamma_nm',
                'inv_W_diag', 'trace_Lambda_Winv'):
        setattr(model, key, ck[key].to(device))
    metrics = ck.get('metrics', {})
    return model, metrics

# -------------------------
# Main Implementation
# -------------------------
def main():
    set_seed(42)

    path    = "/pscratch/sd/s/sanjitr/causal_net_temp/spikesData"
    dataset = "daleN100_f7d754_ce2202"

    try:
        data_load = np.load(f"{path}/{dataset}.spikes.npz")
        Y_np = data_load["spikes"].astype(np.float32)
    except:
        print("Data not found, generating synthetic spike train...")
        Y_np = np.random.poisson(0.1, (1000, 100)).astype(np.float32)

    T, N_neurons = Y_np.shape
    C        = 10          # fine bins per coarse bin (paper: r = C·Δ)
    M        = T // C
    K_states = 5
    # For multi-neuron data with shared change points (paper §5), the per-coarse-bin
    # log-likelihood sums over J neurons: Σ_j [η̂_m^j · x_m^n - C·log2cosh(x_m^n)]
    # = η̂_m_total · x_m^n - C_eff · log2cosh(x_m^n), where C_eff = C · N_neurons.
    # Using C_eff keeps the likelihood/prior balance the same as the single-neuron case.
    C_eff = C * N_neurons

    # Bug 6 fix: η̂_m = Σ_u η_{(m-1)C+u} where η_k = +1 (spike) / -1 (no spike)
    # For multi-neuron data with shared change points, sum across neurons (paper §5).
    # Values range in [-C*N_neurons, +C*N_neurons] — do NOT collapse to a single ±1.
    Y_binary = np.where(Y_np[:M*C] > 0, 1.0, -1.0).astype(np.float32)  # (T, N_neurons)
    eta_hat  = torch.from_numpy(
        Y_binary.reshape(M, C, N_neurons).sum(axis=1).sum(axis=1)
    ).to(device)  # shape (M,)

    model  = VB_SSSM(K_states, M, device)
    Lambda = tridiagonal_precision(M, device)

    print(f"Training VB-SSSM with {K_states} initial states...")

    N_ITER       = 200
    ANNEAL_ITERS = 20   # ramp temperature 0.1 → 1.0 over first 20 iterations
    metrics = {'iters': [], 'log_norm_per_bin': [], 'active_states': [],
               'beta_mean': [], 'beta_max': [], 'temperature': []}

    for i in range(N_ITER):
        temperature = min(1.0, 0.1 + 0.9 * i / max(ANNEAL_ITERS - 1, 1))
        trans_counts, log_norm = variational_e_step(model, eta_hat, Lambda, C_eff,
                                                    temperature=temperature)
        variational_m_step(model, trans_counts, Lambda)

        # ARD: label n is pruned if <z_m^n> < 1e-5 for all m (paper §4)
        active_mask   = model.gamma_nm.max(dim=1).values > 1e-5
        active_states = active_mask.sum().item()
        beta_active   = model.beta_n[active_mask]

        metrics['iters'].append(i)
        metrics['log_norm_per_bin'].append(log_norm / M)
        metrics['active_states'].append(active_states)
        metrics['beta_mean'].append(beta_active.mean().item() if active_states > 0 else 0.0)
        metrics['beta_max'].append(beta_active.max().item()  if active_states > 0 else 0.0)
        metrics['temperature'].append(temperature)

        if i % 20 == 0:
            print(f"Iter {i:3d} (τ={temperature:.2f}): {active_states} active states, "
                  f"log_norm/bin={log_norm/M:.3f}, "
                  f"β_mean={metrics['beta_mean'][-1]:.2f}")

    best_states = torch.argmax(model.gamma_nm, dim=0)
    print("Optimization Complete.")
    print(f"Final State Sequence (First 20 bins): {best_states[:20].tolist()}")

    models_dir = "/pscratch/sd/s/sanjitr/causal_net_temp/models"
    os.makedirs(models_dir, exist_ok=True)
    ckpt_path = f"{models_dir}/{dataset}_vb_sssm.pt"
    save_checkpoint(model, ckpt_path, metrics=metrics)
    print(f"Checkpoint saved to: {ckpt_path}")

if __name__ == "__main__":
    main()
