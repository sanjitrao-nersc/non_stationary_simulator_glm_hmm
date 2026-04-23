import numpy as np
import os
import time
from scipy.optimize import minimize
from scipy.special import logsumexp
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
#import seaborn as sns

# --- 1. GLOBAL BUFFERS ---
GLOBAL_X_PROJ = None # Projected latent history
GLOBAL_Y = None
GLOBAL_GAMMA = None

def init_worker(X_proj_shared, Y_shared, gamma_shared):
    global GLOBAL_X_PROJ, GLOBAL_Y, GLOBAL_GAMMA
    GLOBAL_X_PROJ = X_proj_shared
    GLOBAL_Y = Y_shared
    GLOBAL_GAMMA = gamma_shared

# --- 2. OPTIMIZER: Optimization in Latent Space ---
def optimize_neuron_latent(args):
    n, K, rank, init_u, init_b_stack, lambda_lasso = args
    # Now using the projected latent history (T, Rank)
    X_proj = GLOBAL_X_PROJ 
    Y_n = GLOBAL_Y[:, n]
    gamma = GLOBAL_GAMMA
    
    n_spikes = np.sum(Y_n)
    freq_scale = 1.0 / n_spikes if n_spikes > 0 else 1.0

    def objective(params):
        u = params[:rank] # Interaction weights with the R latent factors
        b_states = params[rank:]
        
        total_nll = 0
        for k in range(K):
            # Log-lambda = (History projected to factors) @ neuron_factors + bias
            log_lam = (X_proj @ u) + b_states[k]
            lam = np.exp(np.clip(log_lam, None, 5)) 
            total_nll -= np.sum(gamma[:, k] * (Y_n * log_lam - lam))
        
        penalty = lambda_lasso * np.sum(np.abs(u)) 
        return (total_nll * freq_scale) + penalty

    res = minimize(objective, np.concatenate([init_u, init_b_stack]), method='L-BFGS-B')
    return n, res.x[:rank], res.x[rank:]

class LowRankNeuralGLMHMM:
    def __init__(self, n_states, n_neurons, rank=5, lambda_lasso=0.1):
        self.K = n_states
        self.N = n_neurons
        self.R = rank
        self.lambda_lasso = lambda_lasso
        
        # Shared factors: U (N x R) and projection weights W (N x R)
        # Reconstructing V = U @ W.T
        self.U = np.random.randn(self.N, self.R) * 0.01
        self.W = np.random.randn(self.N, self.R) * 0.1 # Projection of history into factors
        self.B = np.random.randn(self.K, self.N) * 0.1
        
        self.A = 0.95 * np.eye(self.K) + 0.05 * (np.ones((self.K, self.K)) / self.K)
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.pi = np.ones(self.K) / self.K

    def _compute_log_emissions(self, Y, X):
        T = Y.shape[0]
        log_E = np.zeros((T, self.K))
        # Step 1: Project spiking history into latent space (T, R)
        X_latent = X @ self.W
        
        for k in range(self.K):
            # Step 2: Use low-rank weights U to get neural intensities
            log_lambdas = (X_latent @ self.U.T) + self.B[k]
            lambdas = np.exp(np.clip(log_lambdas, None, 5))
            log_E[:, k] = np.sum(Y * log_lambdas - lambdas, axis=1)
        return log_E

    def _forward_backward(self, log_E):
        T, K = log_E.shape
        log_A = np.log(self.A + 1e-12)
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self.pi + 1e-12) + log_E[0]
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = log_E[t, k] + logsumexp(log_alpha[t-1] + log_A[:, k])
        log_beta = np.zeros((T, K))
        for t in range(T-2, -1, -1):
            for j in range(K):
                log_beta[t, j] = logsumexp(log_A[j, :] + log_E[t+1] + log_beta[t+1])
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        log_xi = np.zeros((T-1, K, K))
        for t in range(T-1):
            for j in range(K):
                for k in range(K):
                    log_xi[t, j, k] = log_alpha[t, j] + log_A[j, k] + log_E[t+1, k] + log_beta[t+1, k]
            log_xi[t] -= logsumexp(log_xi[t])
        return np.exp(log_gamma), np.exp(log_xi)

    def _m_step(self, Y, X, gamma, xi):
        self.pi = gamma[0] / (np.sum(gamma[0]) + 1e-12)
        self.A = (1.1 + np.sum(xi, axis=0)) # Sticky prior [cite: 147-148, 1539]
        self.A /= np.sum(self.A, axis=1, keepdims=True)
        
        # We need to optimize W (projection) or U (weights). 
        # For simplicity/speed, we pre-project X to the latent space for the U update
        X_proj = X @ self.W 
        
        tasks = [(n, self.K, self.R, self.U[n, :], self.B[:, n], self.lambda_lasso) for n in range(self.N)]
        ctx = mp.get_context("spawn")
        results = []
        with tqdm(total=self.N, desc="M-Step (Low-Rank Factors)", unit="neuron") as pbar:
            with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=ctx,
                                     initializer=init_worker, initargs=(X_proj, Y, gamma)) as executor:
                for result in executor.map(optimize_neuron_latent, tasks):
                    results.append(result)
                    pbar.update(1)
        
        for n, new_u, new_b_stack in results:
            self.U[n, :] = new_u
            self.B[:, n] = new_b_stack

    def fit(self, Y, X, n_iters=15):
        for i in range(n_iters):
            start = time.time()
            log_E = self._compute_log_emissions(Y, X)
            gamma, xi = self._forward_backward(log_E)
            self._m_step(Y, X, gamma, xi)
            print(f"Iteration {i+1} complete in {time.time()-start:.1f}s")
        return gamma

# --- 3. DIAGNOSTICS ---
def plot_state_diagnostics(model, gamma, z_true=None):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    t_range = np.arange(len(gamma))
    
    for k in range(model.K):
        axes[0].plot(t_range, gamma[:, k], label=f"p(State {k})")
    axes[0].set_title("Inferred Latent State Strategy [cite: 269-273]")
    axes[0].legend()

    winner_path = np.argmax(gamma, axis=1)
    axes[1].step(t_range, winner_path, where='post', color='black', label='Inferred')
    if z_true is not None:
        axes[1].step(t_range, z_true, where='post', color='red', alpha=0.3, label='Truth')
        acc = np.mean(winner_path == z_true) * 100
        axes[1].set_title(f"Global Accuracy: {acc:.2f}% [cite: 298-302]")
    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
# import seaborn as sns  <-- Comment this out

def plot_reconstructed_connectivity(model, A_true=None):
    reconstructed_V = model.U @ model.U.T
    plt.figure(figsize=(8, 6))
    # Standard matplotlib imshow instead of sns.heatmap
    im = plt.imshow(reconstructed_V, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im)
    plt.title(f"Recovered Shared Connectivity (Rank {model.R})")
    plt.show()

def plot_state_biases(model):
    """
    Displays the state-specific bias B for each neuron.
    Equivalent to 'strategy biases' in the behavioral paper [cite: 254-256].
    """
    plt.figure(figsize=(10, 4))
    for k in range(model.K):
        plt.plot(model.B[k], label=f"State {k} (Bias)", alpha=0.7)
    plt.title("State-Specific Biases (Baseline Excitability)")
    plt.xlabel("Neuron ID")
    plt.ylabel("Log-firing Rate Bias")
    plt.legend()
    plt.show()

def plot_state_trajectories(gamma, z_true=None, zoom_len=500):
    """
    Compares predicted state probabilities (gamma) against actual states.
    Directly replicates the dynamics zoom from the paper[cite: 1682].
    """
    T_zoom = min(zoom_len, len(gamma))
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Probabilities
    for k in range(gamma.shape[1]):
        axes[0].plot(gamma[:T_zoom, k], label=f"p(State {k})")
    axes[0].set_title("Inferred Latent State Probabilities")
    axes[0].set_ylabel("Probability")
    axes[0].legend()

    # Discrete Path
    winner_path = np.argmax(gamma, axis=1)
    axes[1].step(np.arange(T_zoom), winner_path[:T_zoom], where='post', color='black', label='Inferred')
    if z_true is not None:
        axes[1].step(np.arange(T_zoom), z_true[:T_zoom], where='post', color='red', alpha=0.4, label='Truth')
        acc = np.mean(winner_path == z_true) * 100
        axes[1].set_title(f"State Path (Accuracy: {acc:.2f}%)")
    
    axes[1].set_xlabel("Time Bin")
    axes[1].set_ylabel("State ID")
    axes[1].legend()
    plt.tight_layout()
    plt.show()

def compute_and_plot_deviance(model, Y, X, gamma):
    """
    Calculates smoothed Poisson Deviance over time bins.
    High deviance indicates poor fit in specific time windows.
    """
    T, N = Y.shape
    winner_path = np.argmax(gamma, axis=1)
    deviance = []
    
    for t in range(T):
        k = winner_path[t]
        log_lam = (X[t] @ model.U.T) + model.B[k]
        lam = np.exp(np.clip(log_lam, None, 5))
        # Poisson Deviance formula
        d_val = np.sum(2 * (Y[t] * np.log((Y[t] + 1e-12) / (lam + 1e-12)) - (Y[t] - lam)))
        deviance.append(d_val)
    
    smoothed = np.convolve(deviance, np.ones(50)/50, mode='valid')
    plt.figure(figsize=(15, 4))
    plt.plot(smoothed, color='darkgreen', lw=2)
    plt.title("Smoothed Poisson Deviance (Lower is better)")
    plt.ylabel("Deviance")
    plt.xlabel("Time Bin")
    plt.show()

def main():
    dataset_name = "daleN50_fa5eb3"
    try:
        spks = np.load(f'{dataset_name}.spikes.npz', allow_pickle=True)
        tru = np.load(f'{dataset_name}.simTruth.npz', allow_pickle=True)
        data = spks['spikes']
        if data.ndim == 3:
            S, T_block, N = data.shape
            Y = data.reshape(-1, N).astype(int)
            z_true = np.repeat(np.arange(S), T_block)
        else:
            Y = data.astype(int)
            z_true = tru['z_true'] if 'z_true' in tru.files else None
        
        A_true = tru['A_true']
    except Exception as e:
        print(f"File Error: {e}")
        return

    # Use history (interaction) as the covariate matrix
    X = np.roll(Y, 1, axis=0); X[0, :] = 0
    
    # 50 Neurons, 3 States, Rank 5 Interaction Factor
    model = LowRankNeuralGLMHMM(n_states=3, n_neurons=Y.shape[1], rank=5, lambda_lasso=0.2)
    gamma = model.fit(Y, X, n_iters=15)

    plot_state_diagnostics(model, gamma, z_true)

if __name__ == "__main__":
    main()