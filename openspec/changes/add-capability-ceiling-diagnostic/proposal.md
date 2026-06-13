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

> **CORRECTION (2026-06-13) — use ACTIVATION geometry, not readout geometry, for the baselines.** An earlier
> draft of this proposal made the `svd` reference and the `best_atoms` selection **readout-aligned** (R2's
> basis). That is **wrong for this metric.** R2's +52pp validated readout-alignment for a **decode-side**
> target (next-token argmax); `retained_mauc` is **encoder-side** (does the SAE encoder still recover the
> features through the forge). Polygram's `add-readout-aligned-geometry-profile` (closed/archived) confirmed
> the distinction empirically: readout-alignment *hurts* an encoder-side metric (co-firing Spearman
> 0.64 → 0.27). So the **principle** R2 gives transfers — *a trained subspace beats the closed-form one* — but
> the **basis does not**: this diagnostic's frozen baselines and the ceiling's init use the
> **activation/encoder geometry** (activation-PCA + a capability-supervised atom selection), *not* the readout
> subspace. See [[readout-alignment-is-decode-specific]] / polygram archive.

## What — three numbers per (host, width), and a decomposition

At each width `N`, the sweep already computes the interpretable forge. The diagnostic adds two more
**activation-level** retained-mAUC quantities (the basis-quality level — no full forge needed for the oracle):

**Four reference points** (`retained_mauc`, activation-level), from "what ships" to "the oracle". The middle
two are both **interpretable** (SAE atoms); the gap between them is where action lives.

| quantity | definition | interpretable? |
|---|---|:--:|
| `retained_mauc_svd` | top-`N` **activation-PCA** subspace (the encoder-side frozen-linear reference) | n/a (reference) |
| `retained_mauc_pinv` | `pinv`(top-`N`-**by-norm** SAE atoms) — today's default basis | ✅ **← ships** |
| `retained_mauc_best_atoms` | `pinv`(best-`N` SAE atoms by **capability-supervised selection** — the atoms that most preserve the downstream features) | ✅ best *interpretable* basis |
| `retained_mauc_ceiling` | a **trained** rank-`N` subspace (any directions) — the oracle | ❌ never shipped |

**Why the `best_atoms` row matters (the load-bearing refinement, from review).** `ceiling − pinv` lumps two
very different gaps: "*I picked the wrong atoms*" (fixable) and "*even the best atoms can't span the task as
well as free directions*" (intrinsic to insisting on interpretable features). Splitting at `best_atoms`
separates them, so the diagnostic is **actionable**, not just descriptive. Derived gaps on
`ParetoFrontierRow`:

- **`selection_gap` = best_atoms − pinv** — *fixable by better atom selection* (X1, **interpretability
  preserved**). This is the gap you should chase.
- **`interpretability_tax` = ceiling − best_atoms** — *the intrinsic cost of insisting on SAE atoms*; even the
  best interpretable selection cannot match free directions. Not a bug to fix — a tradeoff to **accept or
  reject** at a given rank.
- **`ceiling_gap` = host − ceiling** — what *even* free directions can't recover at rank `N` (entropy-rank +,
  in the full forge, composition). **A measured gap at rank `N`, achievability OPEN** (`no-necessity-claims`)
  — *renamed from "irreducible_floor_gap" per review; it is not a proven floor.*

**Scope (stated, per review):** this is the **activation-level basis-quality** decomposition. The existing
full-forge tax (LayerNorm / TopK non-commutation) sits *on top*; this isolates "is the SAE basis a good rank-`N`
subspace?" from "does the multi-layer forge degrade it further?"

### How the ceiling oracle is trained (design note, per review — the tax is only as good as this)

`retained_mauc_ceiling` is a **single linear rank-`N` projection** `B` (`d_model × N`), **not** an iterative or
non-linear model: `B` initialised at the **activation-PCA** subspace (encoder-side; *not* the readout SVD —
see the correction above), readout **tied** to the task encoder (matched capacity to a rank-`N` lens, no free
readout to overfit), trained by Adam on a
cross-entropy/distill objective against the held-out capability target, reusing `train_encoder`'s split /
early-stop / **scoring-only-AUC** / `overfit_flag` discipline. It is therefore an **empirical ceiling** (the
best subspace *this recipe* finds), not a proven global optimum — so `interpretability_tax` is a **lower
bound** on the intrinsic cost. To bound the recipe's quality the gate SHALL also report a **random-subspace
floor** (mean over a few random rank-`N` projections) and **multi-init** spread, so a reader can see the
ceiling sits well above random and is init-stable. **Circularity guard (per review):** labels are derived from
the SAE's own features, and the oracle trains on them — to avoid the oracle trivially chasing a self-referential
target, the label-defining features SHALL be **held out** of the oracle's training target (train on the
*complement* features / an independent signal where available, e.g. next-token buckets for LM hosts).

## What it drives (the interpretable pipeline — nothing opaque ships)

- **Large `selection_gap`** → you kept the wrong atoms; **fix it with capability-supervised atom selection** —
  score each SAE atom by how much it preserves the downstream features (an **encoder-side / activation**
  criterion, *not* readout-aligned), optionally via Polygram's hierarchical merge machinery. *This is the
  actionable lever; interpretability is preserved.*
- **Large `interpretability_tax` (after selection)** → the *intrinsic* price of an SAE-feature basis at this
  rank — surface it as a **conscious tradeoff** (accept the tax for interpretability, or raise the rank), not
  a tuning target.
- **Large `ceiling_gap`** → even free directions are rank-`N`-limited here; raise `N` or accept the floor.
- Reported **cross-architecture** (GPT-2 + Pythia, via the merged `gpt_neox` adapter) so the decomposition is
  not single-host.

### The metric, defined (per review — not deferred to prior PRs)

`retained_mauc = (basis feature-label AUC) / (host feature-label AUC)`, where for a candidate rank-`N` basis
`P` (projection of host activations): score the SAE encoder applied to `P(host_acts)` against the **labels**
(the SAE's prevalence-band features, binarised; the oracle's training target holds those features out — see
the circularity guard). AUC is per-feature Mann–Whitney, `mAUC` = mean over labels of max-over-latents. All
four quantities use the **same** labels + the **same** held-out items, so the gaps are apples-to-apples.

## Falsifiable acceptance gate (descriptive, both outcomes first-class)

On GPT-2 + Pythia-70m at compressed widths, multi-seed: report the **four** retained-mAUC quantities + the
**three** gaps + the **random-subspace floor** + multi-init spread. Expectations (pre-committed, not gating a
"win"):

- Per R2 (model-general), **`ceiling > best_atoms ≥ pinv > svd ≫ random`** on both hosts → the recipe produces
  a real ceiling and a measurable, host-general decomposition.
- The scientifically interesting splits: is the gap mostly **`selection_gap`** (fixable — chase X1) or mostly
  **`interpretability_tax`** (intrinsic — a real tradeoff)? Either is a useful, reportable answer.

The verdict is the **decomposition**, not a pass/fail. No "irreducible" / "closes the tax" language for
`ceiling_gap` — it is a *measured gap at rank `N`*, achievability open, and the ceiling itself is empirical
(a lower bound on the intrinsic cost).

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
- fieldrun R2 (`tau_star_trained.py` / powered `tau_star_powered.py`) — the source of the *principle* (a
  trained subspace beats the closed-form one). NOTE its **basis** (readout-aligned) is decode-side and does
  **not** transfer to this encoder-side metric — see the correction above and polygram's
  `add-readout-aligned-geometry-profile` (archived: readout-alignment hurts an encoder-side metric).
