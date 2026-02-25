import numpy as np
import ssm
#from ssm.utils import one_hot
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import numpy as np

def load_dale_spikes(file_path='ashwood_sim_dale_data/test_dale.spikes.npz'):
    """
    Loads spike data from the simulation NPZ file and prepares the stimulus array.
    
    Returns:
        y_spikes (np.ndarray): A (10000, 50) array of uint8 spike counts.
        x_stim (np.ndarray): A (10000, 0) empty array for external stimulus features.
    """
    # Load the npz file containing the simulated spikes 
    data = np.load(file_path, allow_pickle=True)
    
    # Extract the spike matrix (T=10000, N=50) 
    y_spikes = data['spikes'].astype(int)
    
    # Get the total number of time steps
    T_total = y_spikes.shape[0]
    
    # Since the provided simulation data does not contain external stimulus regressors,
    # we initialize x_stim with 0 features. This allows the HMM to rely on population 
    # history and bias terms as defined in your design matrix logic.
    x_stim = np.zeros((T_total, 0))
    
    return y_spikes, x_stim

# Usage in your main pipeline:
# y_spikes, x_stim = load_dale_spikes()
# X_design = prepare_neural_design_matrix(y_spikes, x_stim)

def prepare_neural_design_matrix(spikes, stimulus, lag=1):
    """
    Constructs the design matrix X including stimulus and neural history filters.
    Balances fast spiking (micro-time) with lagged network coupling[cite: 3460, 3475].
    """
    T, N = spikes.shape
    # Create self-history and network coupling by shifting spikes [cite: 3427, 3428]
    history = np.roll(spikes, lag, axis=0)
    history[0:lag, :] = 0  # Zero out first entries to handle boundary
    
    # Add a constant column for the base log-firing rate (bias term) [cite: 3430, 3466]
    bias = np.ones((T, 1))
    
    # Concatenate: [Stimulus, Bias, Lagged Spikes] [cite: 3475]
    design_matrix = np.hstack((stimulus, bias, history))
    return design_matrix

def fit_global_initialization(y_all, x_all, n_neurons, input_dim):
    """
    Stage 1: Fits a standard stationary Poisson GLM to obtain baseline weights[cite: 1041, 3437].
    This corresponds to the 1-state model which ensures stable weight initialization.
    """
    print("Fitting Global 1-State GLM...")
    glm_base = ssm.HMM(1, n_neurons, input_dim, 
                       observations="poisson", 
                       #observation_kwargs=dict(link="log")
                       )
    
    # Fit using Maximum Likelihood Estimation [cite: 1041]
    glm_base.fit(y_all, inputs=x_all, method="em", num_iters=50)
    
    # Retrieve the baseline weight vector [cite: 175]
    return glm_base.observations.params[0]

def fit_multistate_glm_hmm(y_all, x_all, n_neurons, input_dim, k_states, base_weights):
    """
    Stage 2: Fits the global GLM-HMM with sticky transitions and noisy weights[cite: 1042, 1047].
    Implements stickiness to favor sustained neural regimes[cite: 3492].
    """
    print(f"Fitting Global {k_states}-State GLM-HMM...")
    
    # transitions="sticky" enforces higher probability on the diagonal [cite: 1539, 3492]
    global_model = ssm.HMM(k_states, n_neurons, input_dim, 
                           observations="poisson", 
                           transitions="sticky")
    
    # Initialize K states with baseline weights + Gaussian noise (sigma=0.2) [cite: 1043]
    for k in range(k_states):
        noise = 0.2 * np.random.randn(*base_weights.shape)
        global_model.observations.params[k] = base_weights + noise
        
    global_model.fit(y_all, inputs=x_all, method="em", num_iters=100)
    return global_model

def infer_session_states(model, y_session, x_session):
    """
    Inference: Recovers latent state trajectories from spiking data using the Forward Algorithm[cite: 3494, 3500].
    """
    # Get posterior state probabilities (gamma) for each macro-time window [cite: 947, 3504]
    posterior_probs = model.expected_states(y_session, input=x_session)[0]
    
    # Determine the most probable state path (Viterbi) [cite: 3508]
    most_likely_states = model.most_likely_states(y_session, input=x_session)
    
    print(np.unique(most_likely_states))

    return posterior_probs, most_likely_states


import matplotlib.pyplot as plt

def plot_inferred_dynamics(model, spikes, inputs, title="Inferred States & Poisson Intensity"):
    """
    Overlays the model's predicted Poisson firing rate (lambda) on the inferred states.
    Matches the Balewski Step A/B logic for state-dependent firing[cite: 3498, 3503].
    """
    # 1. Inference Logic [cite: 3494, 3508]
    posterior_probs = model.expected_states(spikes, input=inputs)[0]
    predicted_states = model.most_likely_states(spikes, input=inputs)
    
    T, N = spikes.shape
    time = np.arange(T)
    K = model.K

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, 
                             gridspec_kw={'height_ratios': [2, 2]})

    # --- TOP PLOT: State Probabilities (Stickiness/Confidence) [cite: 3492, 3552] ---
    for k in range(K):
        axes[0].plot(time, posterior_probs[:, k], label=f"State {k+1} Prob")
    axes[0].set_ylabel("p(state)")
    axes[0].set_title(title)
    axes[0].legend(loc='upper right')

    # --- BOTTOM PLOT: Inferred States + Poisson Intensity Overlay ---
    # Discrete State Path 
    axes[1].step(time, predicted_states + 1, where='post', color='black', 
                 linewidth=3, label="Inferred State (Discrete)", zorder=5)
    

    axes[1].set_yticks(range(1, K + 1))
    axes[1].set_ylabel("Active State ID")
    axes[1].set_xlabel("Time (Micro-steps)")
    
    plt.tight_layout()

    plt.savefig("ashwood_plots/dale/inferred_ns_state_dynamics_dale.png")
    print("\nInferred state dynamics plot DALE saved as inferred_ns_state_dynamics_dale.png. ")

def plot_transition_matrix_heatmap(model, title="Transition Matrix Heatmap"):
    """
    Produces a heatmap of the transition matrix from the fitted HMM.
    Reflects the 'stickiness' and state transition probabilities[cite: 145, 3538].
    """
    # Extract the K x K matrix from the ssm model
    tm = model.transitions.transition_matrix
    K = model.K
    
    fig, ax = plt.subplots(figsize=(7, 6))
    # Use a sequential colormap to highlight high probabilities (persistence)
    im = ax.imshow(tm, cmap='Blues', vmin=0, vmax=1)
    
    # Add a colorbar for scale
    plt.colorbar(im, ax=ax, label='Transition Probability')
    
    # Annotate each cell with its probability value [cite: 1625, 1626]
    for i in range(K):
        for j in range(K):
            # Dynamic text color for readability against dark/light cells
            text_color = "white" if tm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{tm[i, j]:.3f}", 
                    ha="center", va="center", color=text_color, fontweight='bold')
    
    # Labeling axes with State IDs
    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels([f"To State {k+1}" for k in range(K)])
    ax.set_yticklabels([f"From State {k+1}" for k in range(K)])
    
    ax.set_title(title)
    plt.tight_layout()
    
    plt.savefig("ashwood_plots/dale/dale_transition_matrix_heatmap.png")
    print("\nTransition matrix heatmap saved as dale_transition_matrix_heatmap.png.")


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def compare_connectivity(model, state_id, true_connectivity_path='ashwood_plots/dale/test_dale.SimTruth.npz'):
    """
    Compares recovered weights from the GLM-HMM to ground truth simulation weights.
    """
    # 1. Load Ground Truth
    truth_data = np.load(true_connectivity_path, allow_pickle=True)
    A_true = truth_data['A_true']  # Shape (50, 50)
    
    # 2. Extract Recovered Weights for the specific state
    # model.observations.Ws is shape (K, N, M) -> (3, 50, 51)
    state_params = model.observations.params[state_id]
    print(state_params)
    print(state_params.shape)
    # Extract the weight matrix (N_neurons, input_dim)
    # Usually state_params is a tuple; if it's already an array, use it directly
    Ws = state_params[0] if isinstance(state_params, tuple) else state_params
    print(Ws.shape)
    print(Ws)
    # 3. Extract Connectivity Block
    # Column 0: Bias (from your design matrix logic)
    # Columns 1-50: Causal History (the 50x50 connectivity matrix)
    A_rec = Ws[:, 1:] 
    
    # 4. Quantitative Metrics
    from scipy.stats import pearsonr
    r_val, _ = pearsonr(A_true.flatten(), A_rec.flatten())
    mse = np.mean((A_true - A_rec)**2)
    
    print(f"\n--- Connectivity Analysis (State {state_id + 1}) ---")
    print(f"Pearson Correlation: {r_val:.4f}")
    print(f"Mean Squared Error:  {mse:.4f}")
    
    # 4. Visual Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # True Connectivity
    im1 = axes[0].imshow(A_true, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title("Ground Truth (A_true)")
    plt.colorbar(im1, ax=axes[0])
    
    # Recovered Connectivity
    im2 = axes[1].imshow(A_rec, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title(f"Recovered Weights (State {state_id + 1})")
    plt.colorbar(im2, ax=axes[1])
    
    plt.savefig(f"ashwood_plots/dale/connectivity_comparison_state_{state_id+1}.png")
    print(f"Comparison plot saved as ashwood_plots/connectivity_comparison_state_{state_id+1}.png")
    
    return A_rec


def main():
    # --- Simulated Data Parameters ---
    T_total = 10000        # Total time steps (micro-time) [cite: 3460]
    N_neurons = 50         # Neurons in population [cite: 3417]
    #M_stim = 2             # External stimulus features [cite: 3481]
    K = 3                  # Discrete latent regimes [cite: 41, 3457]
    
    # Generate dummy spiking data (y) and stimuli (x)
    #y_spikes = np.random.poisson(0.1, (T_total, N_neurons))
    #x_stim = np.random.randn(T_total, M_stim)
    y_spikes, x_stim = load_dale_spikes()

    # --- Pipeline Execution ---
    
    # 1. Preprocessing
    X_design = prepare_neural_design_matrix(y_spikes, x_stim)
    input_dim = X_design.shape[1]
    
    # 2. Global Initialization (Stage 1)
    base_w = fit_global_initialization(y_spikes, X_design, N_neurons, input_dim)
    
    # 3. Fit Global HMM (Stage 2)
    trained_model = fit_multistate_glm_hmm(y_spikes, X_design, N_neurons, input_dim, K, base_w)
    # 4. Inference on a specific session (Stage 3)
    # For this example, we treat a slice as a session
    y_sess, x_sess = y_spikes[:2000], X_design[:200]
    probs, states = infer_session_states(trained_model, y_sess, x_sess)

    ## PLOTTING
    plot_inferred_dynamics(trained_model, y_sess, x_sess)    
    plot_transition_matrix_heatmap(trained_model)

    '''for state_id in range(4):
        compare_connectivity(trained_model, state_id, true_connectivity_path='ashwood_sim_dale_data/test_dale.SimTruth.npz')
    '''
    print("\nPipeline Complete.")

    print(f"Median state  occupancy: {np.mean(states == 0):.2%}")
    print(f"Median state 2 occupancy: {np.mean(states == 1):.2%}")
    print(f"Median state 3 occupancy: {np.mean(states == 2):.2%}")


if __name__ == "__main__":
    main()