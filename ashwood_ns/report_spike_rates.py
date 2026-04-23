"""
Report per-state mean spike frequencies (Hz) for K=2 and K=3 datasets.

Usage:
    python3 report_spike_rates.py
"""

import numpy as np

DELTA_T = 0.1  # seconds per bin

datasets = [
    {
        "label": "K=2  daleN100_b1036a",
        "spikes": "/pscratch/sd/s/sanjitr/causal_net_temp/spikesData/daleN100_b1036a.spikes.npz",
        "truth":  "/pscratch/sd/s/sanjitr/causal_net_temp/spikesData/daleN100_b1036a.simTruth.npz",
        "spikes_key": "spikes",
        "z_key": "S_true",
    },
    {
        "label": "K=3  daleN100_15b40d_0e5b13",
        "spikes": "/pscratch/sd/s/sanjitr/causal_net_temp/spikesData/daleN100_15b40d_0e5b13.spikes.npz",
        "truth":  "/pscratch/sd/s/sanjitr/causal_net_temp/spikesData/daleN100_15b40d_0e5b13.simTruth.npz",
        "spikes_key": "spikes",
        "z_key": "S_true",
    },
]

for ds in datasets:
    print(f"\n{'='*60}")
    print(f"Dataset: {ds['label']}")
    print(f"{'='*60}")

    spikes = np.load(ds["spikes"])[ds["spikes_key"]].reshape(-1, 100).astype(np.float32)
    truth  = np.load(ds["truth"])
    z_true = truth[ds["z_key"]]

    T, N = spikes.shape
    states = sorted(set(z_true.tolist()))

    print(f"  T={T} bins  N={N} neurons  dt={DELTA_T}s")
    print(f"  Overall mean rate: {spikes.mean() / DELTA_T:.4f} Hz/neuron")
    print()

    for k in states:
        mask = z_true == k
        frac = mask.mean()
        rate_all   = spikes[mask].mean() / DELTA_T
        rate_exc   = spikes[mask, :80].mean() / DELTA_T
        rate_inh   = spikes[mask, 80:].mean() / DELTA_T
        print(f"  State {k}:  occupancy={frac:.3f}  "
              f"mean={rate_all:.4f} Hz  "
              f"exc={rate_exc:.4f} Hz  "
              f"inh={rate_inh:.4f} Hz")
