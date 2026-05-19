"""Probe how polygram.cluster_experts behaves on the K=211 jbloom basis.

The MoE prototype showed that at coherence_threshold=0.30 with
max_features_per_expert=N/target_E, polygram produces mostly-singleton
expert sets (E≈256 from a 523-feature basis when target was 4 or 8).
This script sweeps coherence_threshold across [0.0, 0.5] to find a
value that yields a meaningful E with non-singleton-dominated clusters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from polygram import Dictionary, Feature, HEA_Rung2, cluster_experts
from saeforge.basis import FeatureBasis

ROOT = Path(__file__).resolve().parents[1]
SAE_PATH = ROOT / "smoke_fix_scale_boost" / "baseline" / "_materialised" / "hea" / "pareto" / "k_211.safetensors"


def main():
    basis = FeatureBasis.from_polygram_checkpoint(SAE_PATH)
    print(f"basis: n_features={basis.n_features}, d_model={basis.d_model}")

    features = [Feature(name=f"f_{i}", cluster="c0", beta=0.0) for i in range(basis.n_features)]
    dictionary = Dictionary(
        name="probe",
        features=features,
        hierarchy={"c0": [f"f_{i}" for i in range(basis.n_features)]},
        encoding=HEA_Rung2(depth=1, n_qubits=10),
    )

    # Sweep threshold; vary max_features_per_expert to see effect.
    print(f"\n{'thresh':>7} {'max/exp':>8} {'n_experts':>10} {'singletons':>11} {'median_size':>12} {'max_size':>9}")
    print("-" * 64)
    for thresh in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        for max_per_expert in [None, basis.n_features // 4, basis.n_features // 8]:
            try:
                ed = cluster_experts(
                    dictionary,
                    decoder_vectors=basis.W_dec,
                    method="cosine",
                    coherence_threshold=thresh,
                    max_features_per_expert=max_per_expert,
                )
            except Exception as e:
                print(f"  {thresh:>5} {str(max_per_expert):>8}  raised: {type(e).__name__}: {e}")
                continue
            ft = np.asarray(ed._feature_to_expert, dtype=np.int64)
            sizes = [int((ft == e).sum()) for e in range(ed.n_experts)]
            singletons = sum(1 for s in sizes if s == 1)
            print(
                f"  {thresh:>5} {str(max_per_expert):>8} {ed.n_experts:>10} {singletons:>11} "
                f"{sorted(sizes)[len(sizes)//2]:>12} {max(sizes):>9}"
            )

    # Also report the pairwise cosine distribution on this basis: how
    # many feature pairs are above various cosine thresholds?
    print(f"\nPairwise cosine survey:")
    W = basis.W_dec.astype(np.float64)
    W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-10)
    cosines = W_norm @ W_norm.T
    np.fill_diagonal(cosines, 0.0)
    flat = cosines.flatten()
    for thresh in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        n_pairs = int(np.sum(flat > thresh)) // 2  # symmetric
        print(f"  pairs above cos > {thresh}: {n_pairs}")


if __name__ == "__main__":
    main()
