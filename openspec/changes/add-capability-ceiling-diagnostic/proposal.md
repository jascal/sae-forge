# Capability-ceiling diagnostic — decompose the forge tax, keep the interpretable forge

Add a **diagnostic** to `sweep_pareto_capability` that, at each width `N`, reports the **capability ceiling** a
trained rank-`N` subspace reaches (R2's now-validated, model-general lever) **alongside** the interpretable
SAE-basis forge — and the **tax decomposition** between them. The trained subspace is an **internal oracle
only; it is never shipped as a forge basis.** The product stays the interpretable `pinv`(SAE-atoms) forge; the
diagnostic just tells you *how much you pay for interpretability and how much of the tax is genuinely
irreducible.*

## Why — turn "train the subspace" from a temptation into a measurement

The capability-trained-**encoder** line (X2) asked the wrong question — *"can I train the encoder `E` to beat
`pinv` into a **fixed** SAE dictionary?"* — and got ~no, because for a fixed decoder `pinv` is already optimal
(it wins only on ill-conditioned ReLU dictionaries; the `add-gpt-neox-adapter` Pythia ladder showed that).
The **right** lever is fieldrun's **R2**: train the **subspace itself** (which `r` directions to keep). R2 is
**real, large, and model-general** — a powered rerun (`fieldrun/lo3a/tau_star_powered.py`, wikitext 20k
tokens, 3 seeds) gives trained-subspace − frozen-SVD open-class R@32 of **GPT-2 +52pp** and **Pythia-70m +31pp
(both ranks)**.

But training the subspace produces directions that are **not SAE features** — shipping that as a forge would
abandon sae-forge's interpretability premise. So instead of a forge mode, use the trained subspace as a
**ceiling/oracle**: it bounds what *any* rank-`N` basis could achieve, so the gap to the interpretable basis is
a *measurement of the interpretability cost*, and the gap to the host is the *genuinely irreducible* floor.

## What — three numbers per (host, width), and a decomposition

At each width `N`, the sweep already computes the interpretable forge. The diagnostic adds two more
**activation-level** retained-mAUC quantities (the basis-quality level — no full forge needed for the oracle):

| quantity | definition | role |
|---|---|---|
| `retained_mauc_svd` | project host activations onto the top-`N` readout-aligned **SVD** subspace | reference (the τ\* / frozen-linear floor) |
| `retained_mauc_pinv` | `pinv`(top-`N` **SAE atoms**) — the interpretable basis | **← the shipped forge's basis** |
| `retained_mauc_ceiling` | a **trained** rank-`N` subspace (init readout-aligned SVD, fit on the capability target, held-out, `overfit_flag`-guarded) | **oracle only — never shipped** |

Derived, surfaced on `ParetoFrontierRow`:

- **`interpretability_tax` = ceiling − pinv** — capability the *interpretable* basis leaves on the table (only
  recoverable by giving up feature-level interpretability).
- **`irreducible_floor_gap` = host − ceiling** — what *even* the capability-optimal subspace can't recover at
  rank `N` (the genuine entropy-rank + cross-layer-composition tax; **OPEN** whether truly irreducible, per
  `no-necessity-claims`).

Note this is the **basis-quality** decomposition (activation level). The existing full-forge tax (LayerNorm /
TopK non-commutation) sits *on top* of it; this diagnostic isolates "is the SAE basis a good rank-`N`
subspace?" from "does the multi-layer forge degrade it further?"

## What it drives (the interpretable pipeline — nothing opaque ships)

- **Large `interpretability_tax`** → the SAE dictionary / **which-atoms selection** is leaving capability on
  the table. Act on the *basis* (better SAE, readout-aligned **selection** of atoms — X1) — **not** the encoder
  (X2's near-no-op). Interpretability preserved; you just pick better *features*.
- **Small `interpretability_tax`** → the SAE basis is already near the rank-`N` optimum; stop tuning the basis,
  the residual is `irreducible_floor_gap`.
- Reported **cross-architecture** (GPT-2 + Pythia, via the merged `gpt_neox` adapter) so the decomposition is
  not single-host.

## Falsifiable acceptance gate (descriptive, both outcomes first-class)

On GPT-2 + Pythia-70m at compressed widths, multi-seed: report the three retained-mAUC quantities + the two
gaps. Expectations (pre-committed, not gating a "win"):

- Per R2 (model-general), **`retained_mauc_ceiling` > `retained_mauc_pinv` and > `retained_mauc_svd`** on both
  hosts → there *is* a measurable interpretability tax (the SAE basis is not the rank-`N` optimum), and it's
  not host-specific.
- If `ceiling ≈ pinv` → the SAE basis already *is* near the rank-`N` optimum (no interpretability tax to pay) —
  an equally-useful, reportable outcome.

The verdict is the **decomposition**, not a pass/fail. No "irreducible" / "closes the tax" language for
`irreducible_floor_gap` — it is a *measured gap at rank `N`*, achievability open.

## Scope / what this is NOT

- **Not a forge mode.** The trained subspace is computed and reported; it is **never** returned as a basis the
  forge uses or recommends. No non-interpretable artifact ships.
- **Activation-level oracle in v1** (R2-style fit on host activations). Training the subspace *through the full
  differentiable forge* is a heavier follow-up (`forge_diff` is esm2-only).
- **Off by default** (`compute_capability_ceiling=False`) — opt-in diagnostic; the default sweep is
  byte-identical.

## Related

- `add-capability-trained-encoder` / `add-full-forge-encoder-training` (X2) — the wrong-knob predecessors;
  this reframes their question as a measurement.
- `add-gpt-neox-adapter` — the merged Pythia adapter + the powered-R2 replication that motivate this.
- fieldrun R2 (`tau_star_trained.py` / powered `tau_star_powered.py`) — the source lever used as the oracle;
  FABLE F2 / X1 (readout-aligned basis is the init + the recommended action when the tax is large).
