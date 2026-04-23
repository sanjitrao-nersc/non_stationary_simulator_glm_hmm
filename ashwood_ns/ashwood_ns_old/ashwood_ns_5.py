import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import pearsonr
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm # Requirement: real-time progress visibility

# --- 1. GLOBAL DATA BUFFERS ---
GLOBAL_X = None
GLOBAL_Y = None
GLOBAL_GAMMA = None

def init_worker(X_shared, Y_shared, gamma_shared):
    global GLOBAL_X, GLOBAL_Y, GLOBAL_GAMMA
    GLOBAL_X = X_shared
    GLOBAL_Y = Y_shared
    GLOBAL_GAMMA = gamma_shared

# --- 2. OPTIMIZER WITH FREQUENCY CORRECTION ---
def optimize_single_neuron_parallel(args):
    n, K, init_v, init_b_stack, lambda_lasso, threshold = args
    X = GLOBAL_X
    Y_n = GLOBAL_Y[:, n]
    gamma = GLOBAL_GAMMA
    
    X_bias = X[:, 0:1] 
    X_interact = X[:, 1:]
    
    # Jan's Tip: Frequency correction factor
    n_spikes = np.sum(Y_n)
    freq_scale = 1.0 / n_spikes if n_spikes > 0 else 1.0

    def objective(params):
        v = params[:X_interact.shape[1]]
        b_states = params[X_interact.shape[1]:]
        
        total_nll = 0
        for k in range(K):
            # Jan's Tip: Consistent clipping at max=5
            log_lam = (X_interact @ v) + (X_bias @ [b_states[k]])
            lam = np.exp(np.clip(log_lam, None, 5)) 
            total_nll -= np.sum(gamma[:, k] * (Y_n * log_lam - lam))
        
        # Apply frequency-corrected NLL and LASSO
        penalty = lambda_lasso * np.sum(np.abs(v)) 
        return (total_nll * freq_scale) + penalty

    res = minimize(objective, np.concatenate([init_v, init_b_stack]), method='L-BFGS-B')
    optimized_params = res.x
    new_v = optimized_params[:X_interact.shape[1]]
    new_b_stack = optimized_params[X_interact.shape[1]:]
    
    new_v[np.abs(new_v) < threshold] = 0
    return n, new_v, new_b_stack

class NeuralGLMHMM:
    def __init__(self, n_states, n_neurons, input_dim, lambda_lasso=0.2, alpha_trans=2.0):
        self.K = n_states
        self.N = n_neurons
        self.M = input_dim
        self.lambda_lasso = lambda_lasso  
        self.alpha = alpha_trans          
        
        # Jan's Tip: Shared Connectivity V, State-Specific Bias B
        self.V = np.random.randn(self.N, self.N) * 0.01  
        self.B = np.random.randn(self.K, self.N) * 0.1  
        self.pi = np.ones(self.K) / self.K
        self.A = 0.95 * np.eye(self.K) + 0.05 * (np.ones((self.K, self.K)) / self.K)
        self.A /= self.A.sum(axis=1, keepdims=True)

    def _compute_log_emissions(self, Y, X):
        T = Y.shape[0]
        log_E = np.zeros((T, self.K))
        X_interact = X[:, 1:]
        for k in range(self.K):
            log_lambdas = (X_interact @ self.V.T) + self.B[k]
            lambdas = np.exp(np.clip(log_lambdas, None, 5))
            log_E[:, k] = np.sum(Y * log_lambdas - lambdas, axis=1)
        return log_E

    def _forward_backward(self, log_E):
        T = log_E.shape[0]
        log_A = np.log(self.A + 1e-12)
        log_alpha = np.zeros((T, self.K))
        log_alpha[0] = np.log(self.pi + 1e-12) + log_E[0]
        for t in range(1, T):
            for k in range(self.K):
                log_alpha[t, k] = log_E[t, k] + logsumexp(log_alpha[t-1] + log_A[:, k])
        log_beta = np.zeros((T, self.K))
        for t in range(T-2, -1, -1):
            for j in range(self.K):
                log_beta[t, j] = logsumexp(log_A[j, :] + log_E[t+1] + log_beta[t+1])
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        log_xi = np.zeros((T-1, self.K, self.K))
        for t in range(T-1):
            for j in range(self.K):
                for k in range(self.K):
                    log_xi[t, j, k] = log_alpha[t, j] + log_A[j, k] + log_E[t+1, k] + log_beta[t+1, k]
            log_xi[t] -= logsumexp(log_xi[t])
        return np.exp(log_gamma), np.exp(log_xi)
    
    def _m_step(self, Y, X, gamma, xi, threshold=1e-3):
        self.pi = gamma[0] / (np.sum(gamma[0]) + 1e-12)
        self.A = (self.alpha - 1 + np.sum(xi, axis=0))
        self.A /= np.sum(self.A, axis=1, keepdims=True)
        
        tasks = [(n, self.K, self.V[n, :], self.B[:, n], self.lambda_lasso, threshold) for n in range(self.N)]
        
        ctx = mp.get_context("spawn")
        results = []
        
        # Integration of real-time progress bar for neuron optimization
        with tqdm(total=self.N, desc="Optimizing Neurons (M-Step)", unit="neuron") as pbar:
            with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=ctx,
                                     initializer=init_worker, initargs=(X, Y, gamma)) as executor:
                for result in executor.map(optimize_single_neuron_parallel, tasks):
                    results.append(result)
                    pbar.update(1)
        
        for n, new_v, new_b_stack in results:
            self.V[n, :] = new_v
            self.B[:, n] = new_b_stack

    def fit(self, Y, X, n_iters=30):
        for i in range(n_iters):
            start = time.time()
            log_E = self._compute_log_emissions(Y, X)
            gamma, xi = self._forward_backward(log_E)
            self._m_step(Y, X, gamma, xi)
            print(f"Iteration {i+1}/{n_iters} complete in {time.time() - start:.1f}s")
        return gamma

# --- 3. DIAGNOSTIC PLOTS ---
def plot_weight_distribution_histogram(model, A_true, dataset_name):
    v_rec = model.V.copy()
    np.fill_diagonal(v_rec, 0)
    v_rec_flat = v_rec[v_rec != 0]

    plt.figure(figsize=(10, 6))
    plt.hist(v_rec_flat, bins=100, color='gray', alpha=0.7, label='Recovered Weights (V)')
    a_true_off = A_true.copy()
    np.fill_diagonal(a_true_off, 0)
    plt.hist(a_true_off[a_true_off > 0], bins=50, color='red', alpha=0.3, label='True Exc')
    plt.hist(a_true_off[a_true_off < 0], bins=50, color='blue', alpha=0.3, label='True Inh')
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f"Weight Distribution Diagnostic | {dataset_name}")
    plt.legend()
    plt.savefig(f"ashwood_plots/truthDale/{dataset_name}/weight_histogram.png")

def plot_correlation_scatter(model, A_true, dataset_name):
    v_flat = model.V.flatten()
    a_flat = A_true.flatten()
    r_val, _ = pearsonr(a_flat, v_flat)
    plt.figure(figsize=(6, 6))
    plt.scatter(a_flat, v_flat, alpha=0.3, s=10)
    plt.plot([-0.5, 0.5], [-0.5, 0.5], color='red', linestyle='--')
    plt.title(f"A_true vs V | Pearson R: {r_val:.3f}")
    plt.xlabel("True Weights")
    plt.ylabel("Recovered Weights")
    plt.savefig(f"ashwood_plots/truthDale/{dataset_name}/correlation_scatter.png")


# --- 3. REINSTATED AND MODIFIED PLOTTING FUNCTIONS ---
def generate_dynamics_zoom(model, Y, X, z_true, dataset_name, zoom_start=199850, zoom_len=500):
    """Modified to handle state transitions in high-volume 600k data [cite: 40, 1682]"""
    log_E = model._compute_log_emissions(Y, X)
    gamma, _ = model._forward_backward(log_E)
    winner_path = np.argmax(gamma, axis=1)
    acc = np.mean(winner_path == z_true) * 100 if z_true is not None else 0

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    t_range = np.arange(zoom_start, zoom_start + zoom_len)
    for k in range(model.K):
        axes[0].plot(t_range, gamma[zoom_start:zoom_start+zoom_len, k], label=f"p(State {k})")
    axes[0].set_title(f"State Dynamics Zoom | Global Acc: {acc:.2f}%")
    axes[0].legend(loc='upper right')
    
    axes[1].step(t_range, winner_path[zoom_start:zoom_start+zoom_len], where='post', color='black', label='Recovered')
    if z_true is not None:
        axes[1].step(t_range, z_true[zoom_start:zoom_start+zoom_len], where='post', color='red', alpha=0.3, label='Truth')
    axes[1].set_xlabel("Time Bin"); axes[1].set_ylabel("State ID"); axes[1].legend()
    plt.savefig(f"ashwood_plots/truthDale/{dataset_name}/dynamics_zoom.png")

def plot_connectivity_grid(model, A_true, dataset_name):
    """Compares shared V matrix to ground truth"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(A_true, cmap='RdBu_r', vmin=-0.4, vmax=0.4)
    axes[0].set_title("True Connectivity (A_true)")
    im1 = axes[1].imshow(model.V, cmap='RdBu_r', vmin=-0.4, vmax=0.4)
    axes[1].set_title("Recovered Shared Connectivity (V)")
    plt.colorbar(im0, ax=axes[0]); plt.colorbar(im1, ax=axes[1])
    plt.savefig(f"ashwood_plots/truthDale/{dataset_name}/connectivity_comparison.png")

def main():
    dataset_name = "daleN100_fa5eb3"
    try:
        spks = np.load(f'ashwood_sim_dale_data/truthDale/{dataset_name}.spikes.npz', allow_pickle=True)
        tru = np.load(f'ashwood_sim_dale_data/truthDale/{dataset_name}.simTruth.npz', allow_pickle=True)
        S, T_block, N = spks['spikes'].shape
        Y = spks['spikes'].reshape(-1, N).astype(int)
        z_true = np.repeat(np.arange(S), T_block) # Reconstruct labels 
        A_true = tru['A_true']
    except: return

    X = np.hstack((np.ones((Y.shape[0], 1)), np.roll(Y, 1, axis=0))); X[0, 1:] = 0
    model = NeuralGLMHMM(n_states=3, n_neurons=N, input_dim=X.shape[1], lambda_lasso=0.2)
    model.fit(Y, X, n_iters=25)

    os.makedirs(f"ashwood_plots/truthDale/{dataset_name}", exist_ok=True)
    generate_dynamics_zoom(model, Y, X, z_true, dataset_name)
    plot_connectivity_grid(model, A_true, dataset_name)
    np.savez(f"{dataset_name}_recovered.npz", V=model.V, B=model.B, A=model.A)
    print(f"Run Finished. Dynamics and Connectivity saved to ashwood_plots/truthDale/{dataset_name}/")

if __name__ == "__main__":
    main()