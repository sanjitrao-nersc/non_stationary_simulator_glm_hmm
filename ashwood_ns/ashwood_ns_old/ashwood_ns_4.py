import numpy as np
import os
import time
import hashlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment

class NeuralGLMHMM:
    def __init__(self, n_states, n_neurons, input_dim, lambda_lasso=0.1, alpha_trans=2.0):
        """
        Initializes the GLM-HMM with a LASSO (L1) penalty for sparsity[cite: 34, 53].
        """
        self.K = n_states
        self.N = n_neurons
        self.M = input_dim
        
        self.lambda_lasso = lambda_lasso  
        self.alpha = alpha_trans          
        
        # Random initialization for W with sufficient variance to encourage state separation
        self.W = np.random.randn(self.K, self.N, self.M) * 0.1
        self.pi = np.ones(self.K) / self.K
        
        # Sticky transition initialization favoring self-transitions [cite: 27, 30, 113]
        self.A = 0.95 * np.eye(self.K) + 0.05 * (np.ones((self.K, self.K)) / self.K)
        self.A /= self.A.sum(axis=1, keepdims=True)

    def _compute_log_emissions(self, Y, X):
        """Poisson emission probabilities[cite: 18, 21]."""
        T = Y.shape[0]
        log_E = np.zeros((T, self.K))
        for k in range(self.K):
            # lambda = exp(X * W_k) [cite: 21]
            log_lambdas = X @ self.W[k].T 
            lambdas = np.exp(np.clip(log_lambdas, -20, 20))
            log_E[:, k] = np.sum(Y * log_lambdas - lambdas, axis=1)
        return log_E

    def _forward_backward(self, log_E):
        """E-Step: Forward-Backward algorithm in log-space[cite: 38, 46, 47]."""
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
        """
        M-Step: Maximize ECLL with Lasso Proximal Operator for hard-sparsity[cite: 33, 34, 42].
        """
        self.pi = gamma[0] / np.sum(gamma[0] + 1e-12)
        
        # Transition update with Sticky Dirichlet Prior [cite: 28, 30]
        self.A = (self.alpha - 1 + np.sum(xi, axis=0))
        self.A /= np.sum(self.A, axis=1, keepdims=True)
        
        for k in range(self.K):
            for n in range(self.N):
                def objective(w):
                    log_lam = X @ w
                    lam = np.exp(np.clip(log_lam, -20, 20)) 
                    nll = -np.sum(gamma[:, k] * (Y[:, n] * log_lam - lam))
                    penalty = self.lambda_lasso * np.sum(np.abs(w[1:])) # L1 on filters only [cite: 34]
                    return nll + penalty

                res = minimize(objective, self.W[k, n], method='L-BFGS-B')
                new_w = res.x
                
                # Lasso Proximal Operator: Hard-threshold weak edges
                interaction_weights = new_w[1:]
                interaction_weights[np.abs(interaction_weights) < threshold] = 0
                new_w[1:] = interaction_weights
                
                self.W[k, n] = new_w

    def fit(self, Y, X, n_iters=15):
        """Full EM Loop[cite: 11, 44]."""
        for i in range(n_iters):
            log_E = self._compute_log_emissions(Y, X)
            gamma, xi = self._forward_backward(log_E)
            self._m_step(Y, X, gamma, xi)
            print(f"Iteration {i+1}/{n_iters} complete.")
        return gamma

def remap_model_states(model, B_true):
    """Aligns model states to ground truth using Hungarian Algorithm on B-terms."""
    B_rec = model.W[:, :, 0] 
    cost_matrix = np.zeros((model.K, model.K))
    for i in range(model.K):
        for j in range(model.K):
            cost_matrix[i, j] = np.linalg.norm(B_true[i] - B_rec[j])
    
    _, col_ind = linear_sum_assignment(cost_matrix)
    model.W = model.W[col_ind]
    model.A = model.A[np.ix_(col_ind, col_ind)]
    model.pi = model.pi[col_ind]
    return col_ind

def compute_edge_error_analysis(A_true, A_rec, neur_revFreqIdx, num_excite, threshold=1e-4):
    """
    Performs TP/FP/FN analysis on connectivity edges.
    Separates results for Excitatory vs. Inhibitory neurons (based on source neuron j).
    """
    K, N, _ = A_true.shape
    
    # Identify which sorted indices correspond to Excitatory neurons
    # j is the sorted index, neur_revFreqIdx[j] is the original index
    is_excitatory = (neur_revFreqIdx < num_excite)
    
    metrics = {
        'exc': {'tp': 0, 'fp': 0, 'fn': 0},
        'inh': {'tp': 0, 'fp': 0, 'fn': 0}
    }

    for k in range(K):
        # A[k, i, j] is connection from neuron j to neuron i
        # We look at all columns j to group by source neuron type
        for j in range(N):
            neuron_type = 'exc' if is_excitatory[j] else 'inh'
            
            true_col = A_true[k, :, j]
            rec_col = A_rec[k, :, j]
            
            true_exists = np.abs(true_col) > threshold
            rec_exists = np.abs(rec_col) > threshold
            
            metrics[neuron_type]['tp'] += np.sum(true_exists & rec_exists)
            metrics[neuron_type]['fp'] += np.sum(~true_exists & rec_exists)
            metrics[neuron_type]['fn'] += np.sum(true_exists & ~rec_exists)
            
    return metrics

def compute_edge_error_analysis(model, A_true, A_rec, neur_revFreqIdx, num_excite, dataset_name, threshold=1e-4):
    """
    Performs TP/FP/FN analysis on connectivity edges and generates a bar plot.
    Separates results for Excitatory vs. Inhibitory neurons.
    """
    K, N, _ = A_true.shape
    is_excitatory = (neur_revFreqIdx < num_excite)
    
    metrics = {
        'exc': {'tp': 0, 'fp': 0, 'fn': 0},
        'inh': {'tp': 0, 'fp': 0, 'fn': 0}
    }

    # Calculate metrics across all states
    for k in range(K):
        for j in range(N):
            neuron_type = 'exc' if is_excitatory[j] else 'inh'
            true_exists = np.abs(A_true[k, :, j]) > threshold
            rec_exists = np.abs(A_rec[k, :, j]) > threshold
            
            metrics[neuron_type]['tp'] += np.sum(true_exists & rec_exists)
            metrics[neuron_type]['fp'] += np.sum(~true_exists & rec_exists)
            metrics[neuron_type]['fn'] += np.sum(true_exists & ~rec_exists)
            
    # --- Plotting Logic ---
    labels = ['True Positives (TP)', 'False Positives (FP)', 'False Negatives (FN)']
    exc_vals = [metrics['exc']['tp'], metrics['exc']['fp'], metrics['exc']['fn']]
    inh_vals = [metrics['inh']['tp'], metrics['inh']['fp'], metrics['inh']['fn']]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, exc_vals, width, label='Excitatory', color='skyblue', edgecolor='black', alpha=0.8)
    plt.bar(x + width/2, inh_vals, width, label='Inhibitory', color='salmon', edgecolor='black', alpha=0.8)

    plt.ylabel('Count of Synaptic Edges')
    plt.title(f'Circuit Recovery Error Analysis: {dataset_name}')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Save to the unified plot directory
    plot_dir = f"ashwood_plots/{dataset_name}"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = f"{plot_dir}/edge_error_analysis.png"
    plt.savefig(plot_path)
    print(f"Edge error analysis bar plot saved to {plot_path}")
    
    return metrics

def generate_unified_plots(model, Y, X, A_true, z_true, dataset_name, zoom_len=300):
    """Comprehensive plotting suite for 3-state non-stationary dynamics[cite: 71, 144, 163, 164]."""
    plot_dir = f"ashwood_plots/{dataset_name}"
    os.makedirs(plot_dir, exist_ok=True)
    T, N = Y.shape
    
    log_E = model._compute_log_emissions(Y, X)
    gamma, _ = model._forward_backward(log_E)
    winner_path = np.argmax(gamma, axis=1)
    global_acc = np.mean(winner_path == z_true) * 100 if z_true is not None else 0

    # 1. Zoomed Dynamics & Accuracy [cite: 71, 72]
    plot_len = min(zoom_len, T)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    for k in range(model.K):
        mask = (z_true == k)
        s_acc = np.mean(winner_path[mask] == z_true[mask]) * 100 if np.any(mask) else 0
        axes[0].plot(gamma[:plot_len, k], label=f"State {k} (Acc: {s_acc:.1f}%)")
    axes[0].set_title(f"State Inference | Global Accuracy: {global_acc:.2f}%")
    axes[0].legend(loc='upper right')
    
    axes[1].step(np.arange(plot_len), winner_path[:plot_len], where='post', color='black', label='Recovered')
    if z_true is not None:
        axes[1].step(np.arange(plot_len), z_true[:plot_len], where='post', color='red', alpha=0.3, lw=2, label='Truth')
    axes[1].set_ylabel("State ID")
    plt.savefig(f"{plot_dir}/dynamics_zoom.png")

    # 2. Kx2 Connectivity Matrix Comparison [cite: 144, 145]
    
    fig, axes = plt.subplots(model.K, 2, figsize=(12, 4 * model.K))
    for k in range(model.K):
        im1 = axes[k, 0].imshow(A_true[k], cmap='RdBu_r', vmin=-0.4, vmax=0.4)
        axes[k, 0].set_title(f"True State {k} Connectivity")
        im2 = axes[k, 1].imshow(model.W[k, :, 1:], cmap='RdBu_r', vmin=-0.4, vmax=0.4)
        axes[k, 1].set_title(f"Recovered State {k} Connectivity")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/connectivity_grid.png")

    # 3. Smoothed Poisson Deviance with Transition Markers [cite: 39, 166]
    deviance = []
    for t in range(min(1000, T)):
        k = winner_path[t]
        log_lam = X[t] @ model.W[k].T
        lam = np.exp(np.clip(log_lam, -20, 20))
        d_val = np.sum(2 * (Y[t] * np.log(Y[t] / (lam + 1e-12) + 1e-12) - (Y[t] - lam)))
        deviance.append(d_val)
    
    smoothed = np.convolve(deviance, np.ones(10)/10, mode='valid')
    plt.figure(figsize=(15, 4))
    plt.plot(deviance, alpha=0.2, color='blue')
    plt.plot(smoothed, color='darkblue', lw=2, label='10-bin Smooth')
    if z_true is not None:
        transitions = np.where(np.diff(z_true[:1000]) != 0)[0]
        for tr in transitions:
            plt.axvline(tr, color='red', linestyle='--', alpha=0.5)
    plt.title("Poisson Deviance (Red lines indicate true transitions)")
    plt.savefig(f"{plot_dir}/deviance_smooth.png")


def main():
    dataset_name = "datasetB_ns"
    K_states = 3
    lambda_val = 0.1 
                                                                           
    print(f"Loading Non-Stationary Data: {dataset_name}...")
    spikes_data = np.load(f'ashwood_sim_dale_data/{dataset_name}/test_dale.spikes.npz', allow_pickle=True)
    truth_data = np.load(f'ashwood_sim_dale_data/{dataset_name}/test_dale.simTruth.npz', allow_pickle=True)
    
    Y = spikes_data['spikes'].astype(int)
    T, N = Y.shape
    A_true = truth_data['A_true'] # Now (K, N, N)
    B_true = truth_data['B_true'] # Now (K, N)
    z_true = truth_data['z_true'] if 'z_true' in truth_data.files else None
    
    history = np.roll(Y, 1, axis=0); history[0, :] = 0
    X = np.hstack((np.ones((T, 1)), history)) 
    
    model = NeuralGLMHMM(n_states=K_states, n_neurons=N, input_dim=X.shape[1], lambda_lasso=lambda_val)
    
    print("Starting Sparse EM Fit...")
    model.fit(Y, X, n_iters=15)
    
    print("Aligning states to ground truth...")
    remap_model_states(model, B_true)
    

    ## PLOTTING:
    A_rec = model.W[:, :, 1:] 
    num_excite = truth_data['dale_conf'].item()['num_excite'] 
    neur_revFreqIdx = truth_data['neur_revFreqIdx']
    compute_edge_error_analysis(
        model, A_true, A_rec, neur_revFreqIdx, num_excite, dataset_name
    )
    generate_unified_plots(model, Y, X, A_true, z_true, dataset_name)
    print(f"\nPipeline Complete. Plots saved to ashwood_plots/{dataset_name}/")

if __name__ == "__main__":
    main()