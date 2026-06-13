#!/usr/bin/env python3
"""De-risk gate for the capability-trained encoder (change add-capability-trained-encoder, task 4.4).

Runs the CORE claim — trained-E held-out retained-mAUC >= pinv baseline (and beats it where a real gap
exists) — on a controlled synthetic fixture, BEFORE the sweep/CLI surface or the (unimplemented)
DownstreamCapabilityTarget exist. Exercises tasks 1 (SubspaceProjector.encoder_override) + 2 (train_encoder)
end-to-end on real torch.

Why synthetic (and why it's a fair de-risk): we control the ground truth. The host encoder is **nonlinear**
(ReLU SAE) through a genuine basis bottleneck — exactly the regime where the Frobenius `pinv` projection is
NOT optimal for preserving post-nonlinearity latents (the LayerNorm/TopK tax mechanism, Reckoning #5), so a
distill-trained E *should* recover capability the pinv leaves on the table. The SATURATED control (no
bottleneck) checks the opposite: both near host ⇒ trained ties pinv, no spurious gain.

  python scripts/forge_trained_encoder_gate.py
The real bio-sae fixtures (uniref50 n=512 / n=16) are the formal gate (task 6.1) and run once
add-downstream-capability-target lands its CapabilityDataset + the bundles are present — flagged, not run here.
"""
import argparse
import numpy as np

from saeforge.basis import FeatureBasis
from saeforge.training import train_encoder


def _relu_sae(W_enc, b_enc):
    """A nonlinear host task-encoder: z = ReLU(x @ W_enc + b). torch-differentiable callable."""
    import torch

    Wt = torch.tensor(W_enc, dtype=torch.float32)
    bt = torch.tensor(b_enc, dtype=torch.float32)

    def enc(x):  # x: (M, d_model) torch tensor
        return torch.relu(x @ Wt + bt)

    return enc


def make_fixture(*, d_model, n_features, latent_width, n_items, n_labels, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_items, d_model)).astype(np.float64)
    # anisotropy: a few high-variance, label-IRRELEVANT directions the L2-optimal pinv will chase
    scales = np.ones(d_model)
    scales[: d_model // 3] = 3.0
    X *= scales[None, :]
    W_enc = rng.standard_normal((d_model, latent_width)).astype(np.float64) / np.sqrt(d_model)
    b_enc = (-0.2 * np.abs(rng.standard_normal(latent_width))).astype(np.float64)  # sparsify the ReLU
    host_encoder = _relu_sae(W_enc, b_enc)
    # host latents → labels: top-tertile of a random latent dim per label (host genuinely discriminates)
    import torch

    with torch.no_grad():
        Zh = host_encoder(torch.tensor(X, dtype=torch.float32)).numpy()
    label_dims = rng.choice(latent_width, size=n_labels, replace=False)
    Y = np.zeros((n_items, n_labels), dtype=np.float64)
    for v, ld in enumerate(label_dims):
        thr = np.quantile(Zh[:, ld], 0.67)
        Y[:, v] = (Zh[:, ld] > thr).astype(np.float64)
    # basis: random decoder, n_features rows (the bottleneck when n_features < d_model)
    W_dec = (rng.standard_normal((n_features, d_model)) / np.sqrt(d_model)).astype(np.float64)
    basis = FeatureBasis(
        kept_ids=np.arange(n_features, dtype=np.int64),
        W_dec=W_dec,
        merged_norms=np.linalg.norm(W_dec, axis=1).astype(np.float32),
        original_norms=np.linalg.norm(W_dec, axis=1).astype(np.float32),
    )
    return basis, X, host_encoder, Y


def run(label, *, d_model, n_features, latent_width, n_items, n_labels, steps, lr, seed):
    basis, X, host_encoder, Y = make_fixture(
        d_model=d_model, n_features=n_features, latent_width=latent_width,
        n_items=n_items, n_labels=n_labels, seed=seed,
    )
    E, rep = train_encoder(
        basis=basis, host_acts=X, host_encoder=host_encoder, labels=Y,
        objective="distill", loss="cosine", steps=steps, lr=lr, holdout_frac=0.3, seed=seed,
    )
    gate = "PASS (>= baseline)" if rep.delta_heldout >= -1e-6 else "FAIL (< baseline)"
    print(f"\n== {label} (d_model={d_model}, n_features={n_features}, "
          f"{'BOTTLENECK' if n_features < d_model else 'saturated'}; n={n_items}) ==")
    print(f"   held-out retained-mAUC:  pinv {rep.retained_mauc_pinv_baseline:.3f}  "
          f"→ trained {rep.retained_mauc_trained:.3f}   (Δ {rep.delta_heldout:+.3f})  [{gate}]")
    print(f"   fit-split trained retained-mAUC {rep.retained_mauc_trained_fit:.3f}   "
          f"overfit_flag={rep.overfit_flag}   steps_run={rep.steps_run}/{rep.steps}   "
          f"n_fit={rep.n_fit}/n_held={rep.n_heldout}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--n-items", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print("== capability-trained-encoder de-risk gate (synthetic, ReLU host-encoder) ==")
    # the regime the change targets: a genuine basis bottleneck + nonlinear encoder
    r_bottleneck = run("BOTTLENECK (n_features < d_model)", d_model=32, n_features=10, latent_width=24,
                       n_items=args.n_items, n_labels=6, steps=args.steps, lr=args.lr, seed=args.seed)
    # control: no bottleneck ⇒ both near host ⇒ trained should tie pinv (no spurious gain)
    r_saturated = run("SATURATED control (n_features >= d_model)", d_model=32, n_features=48, latent_width=24,
                      n_items=args.n_items, n_labels=6, steps=args.steps, lr=args.lr, seed=args.seed)
    print("\n== verdict ==")
    print(f"   bottleneck Δ {r_bottleneck.delta_heldout:+.3f} (expect > 0: trained recovers tax the pinv leaves), "
          f"saturated Δ {r_saturated.delta_heldout:+.3f} (expect ≈ 0: no spurious gain).")
    ok = r_bottleneck.delta_heldout >= -1e-6 and not r_bottleneck.overfit_flag
    print(f"   CORE CLAIM exercised end-to-end (tasks 1+2): {'OK' if ok else 'CHECK'}. "
          f"Formal bio-sae gate (task 6.1) pending CapabilityDataset + bundles.")


if __name__ == "__main__":
    main()
