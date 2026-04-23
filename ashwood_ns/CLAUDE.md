The goal of this project is to model non-stationary neural dynamics. The idea is to do the following:
- 1) recover the K latent states of the data
- 2) recover the general connectivity matrix A (exic, inhib, no edge)

Current challenges:
- The inhibitory block of the data is nearly perfectly recovered, now excitatory block contains nearly all of the remaining False Positives and False Negatives. We need to almost completelty eliminate these FPs and FNs.