import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment

class NeuralGLMHMM:
    def __init__(self, n_states, n_neurons, input_dim, lambda_lasso=0.1, alpha_trans=2.0):
        """
        Initializes the GLM-HMM with a LASSO (L1) penalty hyperparameter.
        """
        self.K = n_states
        self.N = n_neurons
        self.M = input_dim
        
        # Updated Hyperparameters 
        self.lambda_lasso = lambda_lasso  # L1 regularization strength (replaces sigma_w)
        self.alpha = alpha_trans          # Dirichlet prior shape for transitions
        
        # Parameters to learn
        self.W = np.random.randn(self.K, self.N, self.M) * 0.1
        self.pi = np.ones(self.K) / self.K
        
        # Sticky transition initialization
        self.A = 0.95 * np.eye(self.K) + 0.05 * (np.ones((self.K, self.K)) / self.K)
        self.A /= self.A.sum(axis=1, keepdims=True)

    def generate_stable_states(self, T, avg_dwell_time):
        """
        Generates a state sequence with a guaranteed minimum dwelling time.
        Logic: 1/3 guaranteed dwell, 2/3 probabilistic Markov window.
        """
        states = np.zeros(T, dtype=int)
        # 1. Split the dwell time: 1/3 is fixed, 2/3 is probabilistic
        min_dwell = max(1, int(avg_dwell_time // 3))
        prob_dwell = max(1, avg_dwell_time - min_dwell)
        
        # 2. Probability of switching during the Markovian (2/3) window
        p_switch = 1.0 / prob_dwell 
        
        current_state = np.random.choice(self.K)
        t = 0
        while t < T:
            # Step A: Apply the "Extra 1/3" guaranteed dwelling
            dwell_end = min(t + min_dwell, T)
            states[t:dwell_end] = current_state
            t = dwell_end
            
            # Step B: Apply the "Current MC 2/3" probabilistic window
            while t < T:
                states[t] = current_state
                # Roll for a state switch
                if np.random.rand() < p_switch:
                    # Switch to a different random state
                    others = [s for s in range(self.K) if s != current_state]
                    current_state = np.random.choice(others)
                    t += 1
                    break # Trigger the next guaranteed refractory period
                t += 1
        return states

    def _compute_log_emissions(self, Y, X):
        """Poisson emission probabilities for neural spikes."""
        T = Y.shape[0]
        log_E = np.zeros((T, self.K))
        for k in range(self.K):
            # Poisson intensity: lambda = exp(X * W_k)
            # log(P(y|lambda)) = y*log(lambda) - lambda - log(y!)
            # We ignore log(y!) as it is constant for optimization
            log_lambdas = X @ self.W[k].T 
            lambdas = np.exp(np.clip(log_lambdas, -20, 20))
            
            # Sum log-likelihood across all N neurons
            log_E[:, k] = np.sum(Y * log_lambdas - lambdas, axis=1)
        return log_E

    def _forward_backward(self, log_E):
        """E-Step: Forward-Backward algorithm in log-space[cite: 971, 972]."""
        T = log_E.shape[0]
        log_A = np.log(self.A + 1e-12)
        
        # Forward pass 
        log_alpha = np.zeros((T, self.K))
        log_alpha[0] = np.log(self.pi + 1e-12) + log_E[0]
        for t in range(1, T):
            for k in range(self.K):
                log_alpha[t, k] = log_E[t, k] + logsumexp(log_alpha[t-1] + log_A[:, k])
        
        # Backward pass
        log_beta = np.zeros((T, self.K))
        for t in range(T-2, -1, -1):
            for j in range(self.K):
                log_beta[t, j] = logsumexp(log_A[j, :] + log_E[t+1] + log_beta[t+1])
                
        # Posterior state probabilities (gamma)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        
        # Joint posterior for transitions (xi)
        log_xi = np.zeros((T-1, self.K, self.K))
        for t in range(T-1):
            for j in range(self.K):
                for k in range(self.K):
                    log_xi[t, j, k] = log_alpha[t, j] + log_A[j, k] + log_E[t+1, k] + log_beta[t+1, k]
            log_xi[t] -= logsumexp(log_xi[t])
            
        return np.exp(log_gamma), np.exp(log_xi)
    
    def _m_step(self, Y, X, gamma, xi):
        """
        M-Step: Maximize the Expected Complete Log-Likelihood (ECLL) 
        using a LASSO (L1) penalty to promote sparsity[cite: 53].
        """
        T = Y.shape[0]
        
        # 1. Update Initial Distribution [cite: 44]
        self.pi = gamma[0] / np.sum(gamma[0])
        
        # 2. Update Transition Matrix with Dirichlet Prior
        self.A = (self.alpha - 1 + np.sum(xi, axis=0))
        self.A /= np.sum(self.A, axis=1, keepdims=True)
        
        # 3. Update GLM Weights via Numerical Optimization (L-BFGS)
        for k in range(self.K):
            for n in range(self.N):
                def objective(w):
                    log_lam = X @ w
                    lam = np.exp(np.clip(log_lam, -20, 20)) 
                    
                    # Weighted Poisson Negative Log-Likelihood [cite: 18]
                    nll = -np.sum(gamma[:, k] * (Y[:, n] * log_lam - lam))
                    
                    # LASSO Penalty (L1): Sum of absolute values of weights
                    # Note: Typically the bias (w[0]) is excluded from the penalty
                    penalty = self.lambda_lasso * np.sum(np.abs(w[1:])) 
                    
                    return nll + penalty

                # L-BFGS-B is used to minimize the non-differentiable (L1) objective
                res = minimize(objective, self.W[k, n], method='L-BFGS-B')
                self.W[k, n] = res.x

    def fit(self, Y, X, n_iters=20):
        """Full EM Loop[cite: 898, 1070]."""
        for i in range(n_iters):
            log_E = self._compute_log_emissions(Y, X)
            gamma, xi = self._forward_backward(log_E)
            self._m_step(Y, X, gamma, xi)
            print(f"Iteration {i+1}/{n_iters} complete.")
        return gamma

    def viterbi(self, Y, X):
        """Decodes the most likely state path."""
        T = Y.shape[0]
        log_E = self._compute_log_emissions(Y, X)
        log_A = np.log(self.A + 1e-12)
        
        v = np.zeros((T, self.K))
        ptr = np.zeros((T, self.K), dtype=int)
        
        v[0] = np.log(self.pi + 1e-12) + log_E[0]
        for t in range(1, T):
            for k in range(self.K):
                prob = v[t-1] + log_A[:, k]
                ptr[t, k] = np.argmax(prob)
                v[t, k] = log_E[t, k] + prob[ptr[t, k]]
        
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(v[-1])
        for t in range(T-2, -1, -1):
            path[t] = ptr[t+1, path[t+1]]
        return path
    

def compute_segmented_correlations(A_true, A_rec, minW=0.03):
    """
    Refined Pearson-r calculation for 3 distinct weight regions.
    """
    true_flat = A_true.flatten()
    rec_flat = A_rec.flatten()
    
    # Masks based on Jan's insight
    mask_inh = true_flat < -minW
    mask_sparse = (true_flat >= -minW) & (true_flat <= minW)
    mask_exc = true_flat > minW
    
    # Independent r calculations
    r_inh, _ = pearsonr(true_flat[mask_inh], rec_flat[mask_inh]) if np.sum(mask_inh) > 2 else (0, 0)
    r_sparse, _ = pearsonr(true_flat[mask_sparse], rec_flat[mask_sparse]) if np.sum(mask_sparse) > 2 else (0, 0)
    r_exc, _ = pearsonr(true_flat[mask_exc], rec_flat[mask_exc]) if np.sum(mask_exc) > 2 else (0, 0)
    
    return r_inh, r_sparse, r_exc

def remap_model_states(model, B_true):
    """
    Finds the optimal mapping and physically permutes model parameters.
    B_true should be shape (K, N) representing the true bias for each state.
    """
    # 1. Calculate cost matrix (L2 distance between biases)
    B_rec = model.W[:, :, 0] 
    num_states = model.K
    cost_matrix = np.zeros((num_states, num_states))
    for i in range(num_states):
        for j in range(num_states):
            cost_matrix[i, j] = np.linalg.norm(B_true[i] - B_rec[j])
    
    # 2. Hungarian Algorithm for optimal matching
    _, col_ind = linear_sum_assignment(cost_matrix)
    # col_ind[i] tells us which recovered index matches true state i
    
    # 3. Permute model parameters to align with ground truth order
    model.W = model.W[col_ind]
    model.A = model.A[np.ix_(col_ind, col_ind)]
    model.pi = model.pi[col_ind]
    
    print(f"States remapped: Learned indices {col_ind} -> True indices [0, 1, 2]")
    return col_ind

def generate_unified_plots(model, Y, X, A_true, z_true, dataset_name, zoom_len=300):
    """
    Final unified suite using aligned parameters and smoothed deviance.
    """
    plot_dir = f"ashwood_plots/unified_{dataset_name}"
    os.makedirs(plot_dir, exist_ok=True)
    T, N = Y.shape
    
    # 1. Inference with ALIGNED parameters
    log_E = model._compute_log_emissions(Y, X)
    gamma, _ = model._forward_backward(log_E)
    winner_path = np.argmax(gamma, axis=1)
    
    # Accuracy reporting as requested by Jan
    accuracy = np.mean(winner_path == z_true) * 100 if z_true is not None else 0
    
    # 2. Zoomed Latent Dynamics (300 steps)
    plot_len = min(zoom_len, T)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    for k in range(model.K):
        # Calculate per-state accuracy for Jan's "middle-frequency" check
        mask = (z_true == k)
        s_acc = np.mean(winner_path[mask] == z_true[mask]) * 100 if np.any(mask) else 0
        axes[0].plot(gamma[:plot_len, k], label=f"State {k} (Acc: {s_acc:.1f}%)")
    
    axes[0].set_title(f"Posterior Probabilities | GLOBAL ACCURACY: {accuracy:.2f}%")
    axes[0].set_ylabel("Confidence")
    axes[0].legend(loc='upper right')
    
    axes[1].step(np.arange(plot_len), winner_path[:plot_len], where='post', color='black', label='Recovered')
    if z_true is not None:
        print("z_true is found")
        axes[1].step(np.arange(plot_len), z_true[:plot_len], where='post', color='red', alpha=0.3, lw=2, label='Truth')
    axes[1].set_ylabel("State ID")
    axes[1].legend(loc='upper right')
    plt.savefig(f"{plot_dir}/dynamics_zoom.png")


    # 3. Transition Matrix Heatmap (Sticky Prior Visualization)
    plt.figure(figsize=(7, 6))
    plt.imshow(model.A, cmap='Blues', vmin=0, vmax=1)
    for i in range(model.K):
        for j in range(model.K):
            color = "white" if model.A[i,j] > 0.5 else "black"
            plt.text(j, i, f"{model.A[i,j]:.3f}", ha="center", va="center", color=color, fontweight='bold')
    plt.title("Recovered Transition Matrix (Sticky Prior)")
    plt.colorbar(label="Probability")
    plt.savefig(f"{plot_dir}/transitions.png")

    # 4. Connectivity Heatmaps (Dale's Law & Sparsity)
    A_rec = model.W[0][:, 1:] # Interaction weights for State 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(A_true, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[0].set_title("Ground Truth ($A_{true}$)")
    im2 = axes[1].imshow(A_rec, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[1].set_title("Recovered ($A_{rec}$)")
    plt.colorbar(im1, ax=axes[0]); plt.colorbar(im2, ax=axes[1])
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/connectivity_heatmaps.png")

    # 5. Segmented Correlation Plot (Excitatory vs Inhibitory vs Sparse)
    r_inh, r_sparse, r_exc = compute_segmented_correlations(A_true, A_rec, minW=0.03)
    r_glob, _ = pearsonr(A_true.flatten(), A_rec.flatten())
    
    plt.figure(figsize=(7, 7))
    plt.scatter(A_true.flatten(), A_rec.flatten(), alpha=0.3, s=10, color='darkcyan')
    plt.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', alpha=0.5)
    plt.title(f"Weight Recovery (Global r={r_glob:.3f})\nExc: r={r_exc:.3f} | Inh: r={r_inh:.3f} | Sparse: r={r_sparse:.3f}")
    plt.xlabel("True Weight"); plt.ylabel("Recovered Weight")
    plt.grid(True, alpha=0.2)
    plt.savefig(f"{plot_dir}/correlation_segmented.png")

    # 6. Poisson Deviance (Error Tracking)
    deviance = []
    for t in range(min(1000, T)):
        k = winner_path[t]
        log_lam = X[t] @ model.W[k].T
        lam = np.exp(np.clip(log_lam, -20, 20))
        term1 = Y[t] * np.log(Y[t] / lam + 1e-12)
        term1[Y[t] == 0] = 0
        deviance.append(np.sum(2 * (term1 - (Y[t] - lam))))
        
    plt.figure(figsize=(15, 4))
    plt.plot(deviance, color='royalblue', alpha=0.7)
    plt.title("Poisson Deviance per Time Step")
    plt.ylabel("Deviance"); plt.xlabel("Time (Micro-steps)")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/deviance.png")
    
    print(f"Unified analysis plots saved to {plot_dir}")

def main():
    # --- 1. Configuration ---
    dataset_name = "datasetB"
    K_states = 3
    avg_dwell = 30  # Requested average persistence
    lambda_val = 0.1 # LASSO strength for sparsity
                                                                           
    # --- 2. Synthetic Data Generation (Testing the Stable Logic) ---
    print("Generating synthetic 'stable' dataset...")
    # Initialize a model with N=50 neurons and 51 inputs (Bias + 50 Lags)
    #gen_model = NeuralGLMHMM(n_states=K_states, n_neurons=50, input_dim=51)
    
    # Step A: Use the '1/3 + 2/3' logic to create a stable state sequence
   # z_true = gen_model.generate_stable_states(T=10000, avg_dwell_time=avg_dwell)
    
    # --- 3. Loading Real Data for Fitting ---
    # (Assuming we continue to fit the uploaded 'test_dale' files)
    print(f"Loading data for fitting: {dataset_name}...")
    spikes_data = np.load(f'ashwood_sim_dale_data/{dataset_name}/test_dale.spikes.npz', allow_pickle=True)
    truth_data = np.load(f'ashwood_sim_dale_data/{dataset_name}/test_dale.simTruth.npz', allow_pickle=True)
    
    Y = spikes_data['spikes'].astype(int)
    T, N = Y.shape
    A_true = truth_data['A_true']
    
    # Preprocessing: Design Matrix [cite: 23]
    history = np.roll(Y, 1, axis=0); history[0, :] = 0
    X = np.hstack((np.ones((T, 1)), history)) 
    
    # --- 4. Fitting with LASSO and Evaluaton ---
    # The L1 penalty ensures recovered connectivity heatmaps are sparse
    model = NeuralGLMHMM(n_states=K_states, n_neurons=N, input_dim=X.shape[1], lambda_lasso=lambda_val)
    
    print("Starting Sparse EM Fit...")
    model.fit(Y, X, n_iters=15)
    
    # Unified Plotting: Dynamics accuracy, segmented r-values, and 300-step zoom
    # We use z_true from the truth_data if available to calculate accuracy
    z_true_loaded = truth_data['z_true'] if 'z_true' in truth_data.files else None

    ## REMAP THE MODEL
    B_true = truth_data['B_true']
    print(B_true.shape) 
    if B_true.ndim == 1: # If only one bias provided, tile it or find the state variations
        B_true = np.tile(B_true, (K_states, 1)) 
    remap_model_states(model, B_true)
    
    generate_unified_plots(
        model=model, 
        Y=Y, 
        X=X, 
        A_true=A_true, 
        z_true=z_true_loaded, 
        dataset_name=dataset_name,
        zoom_len=300
    )
    
    print(f"\nPipeline Complete. Stable dynamics and sparse connectivity saved to ashwood_plots/unified_{dataset_name}/")

if __name__ == "__main__":
    main()
