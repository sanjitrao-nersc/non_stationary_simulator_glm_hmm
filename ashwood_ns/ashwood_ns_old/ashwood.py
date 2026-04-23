import numpy as np
import ssm
import matplotlib.pyplot as plt

def build_neural_design_matrix(stimulus, spikes):
    """
    Constructs the design matrix X for neural data[cite: 3476, 3532].
    Includes: 1) Stimulus, 2) Bias, 3) Self-History, 4) Network Coupling
    """
    num_trials, num_neurons = spikes.shape
    
    # 1. External sensory stimulus (z-scored) [cite: 3482]
    stim = (stimulus - np.mean(stimulus)) / np.std(stimulus)
    if stim.ndim == 1:
        stim = stim.reshape(-1, 1)
    
    # 2. Constant base rate (bias) [cite: 3467, 3535]
    bias = np.ones((num_trials, 1))
    
    # 3. History/Coupling: Lag-1 activity of all neurons [cite: 3470, 3472]
    # We use y_{t-1} as the covariate for time t
    history = np.zeros_like(spikes)
    history[1:] = spikes[:-1]
    
    # Final Design Matrix X_t
    design_matrix = np.hstack((stim, bias, history))
    return design_matrix

def initialize_neural_glm_hmm(num_states, num_neurons, input_dim):
    """
    Initializes the HMM with state-dependent Poisson observations[cite: 3479, 3577].
    - num_states: K (discrete regimes)
    - num_neurons: D (observation dimension)
    - input_dim: M (number of covariates)
    """
    # Using 'poisson' observations with M inputs implements the Neural GLM-HMM
    model = ssm.HMM(num_states, 
                    num_neurons, # D (Number of neurons)
                    M=input_dim, # M (Covariates)
                    observations="poisson", 
                    transitions="standard")
    return model

def apply_dales_law(weights, num_neurons):
    """
    Constrains the coupling weights to follow Dale's Law (20/80 E-I Split).
    Matches Section VII-A and VIII-A.
    """
    # 20% Excitatory, 80% Inhibitory
    num_exc = int(0.2 * num_neurons)
    
    # weights shape is (K, N, M). 
    # M = [Stimulus, Base Rate, Neuron 0 Lag, Neuron 1 Lag, ...]
    # Coupling/History weights start at index 2.
    for j in range(num_neurons):
        coupling_col_idx = 2 + j
        if j < num_exc:
            # Excitatory: neuron j's effect must be >= 0
            weights[:, :, coupling_col_idx] = np.maximum(0, weights[:, :, coupling_col_idx])
        else:
            # Inhibitory: neuron j's effect must be <= 0
            weights[:, :, coupling_col_idx] = np.minimum(0, weights[:, :, coupling_col_idx])
            
    return weights

def fit_with_constraints(model, spikes, inputs, num_neurons, total_iters=50):
    """
    Fits the model using an iterative projection method to enforce Dale's Law.
    """
    print(f"Fitting with Dale's Law (20/80 E-I split)...")
    
    # Run fit in small chunks to apply constraints iteratively
    chunk_size = 5
    for i in range(0, total_iters, chunk_size):
        model.fit(spikes, inputs=inputs, method="em", num_iters=chunk_size)
        
        # Manually constrain the observations
        current_weights = model.observations.params
        constrained_weights = apply_dales_law(current_weights, num_neurons)
        model.observations.params = constrained_weights
        
    return model

def plot_neural_strategies(transitions, state_weights, num_neurons):
    """
    Plots the learned firing rate filters for a specific neuron across regimes.
    Consistent with Section III-A's state-dependent instantaneous firing rate.
    """
    # 1. Convert transitions to probabilities
    # Diagonal values > 0.9 indicate temporal stability (stickiness) [cite: 3493]
    prob_matrix = np.exp(transitions)
    print("\nState Transition Probabilities:\n", prob_matrix)

    # 2. Extract and format weights
    # In ssm.HMM(poisson), weights are often (K, N, M)
    # Ensure weights are 3D
    if state_weights.ndim == 2:
        # If (K, N*M), reshape it
        weights = state_weights.reshape(transitions.shape[0], num_neurons, -1)
    else:
        weights = state_weights

    # covariates per neuron (M) = 1 (Stim) + 1 (Base Rate) + N (Coupling)
    num_covariates = weights.shape[2]
    labels = ['Stimulus', 'Base Rate'] + [f'Coup: N{i}' for i in range(num_neurons)]
    
    # Ensure labels length matches covariate dimension
    labels = labels[:num_covariates]

    plt.figure(figsize=(10, 6))
    
    # Plotting strategy for Neuron 0 across all regimes (states)
    for k in range(weights.shape[0]):
        # weights[k, 0, :] is the filter for Neuron 0 in State k [cite: 3465]
        plt.plot(labels, weights[k, 0, :], marker='o', label=f'Regime {k+1}')
    
    plt.axhline(0, color='k', linestyle='--')
    plt.ylabel("Filter Weight (Log-Rate Scale)") ## [cite: 3431, 3572]
    plt.title("Neural State Strategies: Neuron 0 Filters")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("neural_dale_strategies.png")
    print("\nNeural strategy plot saved as 'neural_strategies.png'.")


def main():
    # 1. Model Parameters [cite: 3458, 3673]
    K = 3  # Latent states
    N = 5  # Number of neurons (simulated small population)
    num_trials = 2000
    
    # 2. Simulate Mock Neural Data
    # Stimulus, Spikes (Counts >= 0), and resulting Design Matrix
    mock_stim = np.random.randn(num_trials)
    mock_spikes = np.random.poisson(lam=1.0, size=(num_trials, N)) # [cite: 3432]
    
    X = build_neural_design_matrix(mock_stim, mock_spikes)
    D_in = X.shape[1]
  
    # 3. Initialize and Train
    print(f"Fitting Neural GLM-HMM with K={K} states for {N} neurons...")
    model = initialize_neural_glm_hmm(K, N, D_in)
    
    # Fit using EM [cite: 3675]
    lls = model.fit(mock_spikes, inputs=X, method="em", num_iters=50)
    
    #model = fit_with_constraints(model, mock_spikes, mock_stim, N)

    # 4. Extract Results
    # Transitions (K, K) and Poisson weights (K, D, M)
    transitions = model.transitions.params[0]
    state_weights = model.observations.params 
    
    plot_neural_strategies(transitions, state_weights, N)
    print(f"\nFinal Log-Likelihood: {lls[-1]:.2f}")

if __name__ == "__main__":
    main()