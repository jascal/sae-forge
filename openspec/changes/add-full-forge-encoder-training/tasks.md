# Implementation tasks

## Implementation status — COMPLETE (2026-06-13)

All tasks landed on branch `feat/full-forge-encoder-diff-spike` (PR #118). ruff clean; new tests green;
default path (`train_objective="proxy"`) byte-identical so the parent change's tests are unaffected.

- **0.1 spike — PASS** (`scripts/spike_forge_diff_autograd.py`): autograd reaches `E` end-to-end through the
  ESM-2 forge (`sum|grad|=40.79`) and the differentiable path reproduces the inference forge at `E=pinv·scale`
  (`max|Δ|=1.07e-06`). The make-or-break risk is resolved.
- **1** `saeforge/forge_diff.py` (`DifferentiableEsm2Forge`, `differentiable_forge_h`) — reparametrize
  E-dependent forged params as `host_source @ E`, run via `functional_call`. esm2-only. `tests/test_forge_diff.py` (5).
- **2** `train_encoder(objective="forge_distill")` — loss + held-out scoring through the full forge.
  `tests/test_train_encoder_forge.py` (2).
- **3** `sweep_pareto_capability(train_objective={proxy,full_forge})` + `_run_capability_cell`. e2e test on
  the tiny host. **4** CLI `--train-objective`. **5** multi-seed bio gate (`forge_trained_encoder_bio_gate.py`).
- **GATE RESULT — also-plateaus (negative) on ESM-2:** full-forge training does NOT reliably beat `pinv` on
  the real spread fixture (n=128, 3 seeds: Δ **−0.0144 ± 0.0052**, clean; n=256 Δ +0.0395 but **overfit**).
  The basis *projection* (`pinv`) is near-optimal **for this substrate**; the tax is structural beyond `E`.
- **CAUSAL-LM CONTROL — host-class caveat CONFIRMED (`scripts/causal_lm_forge_gate.py`):** the same
  activation-level trained-vs-`pinv` gate run on a **causal** host (GPT-2 + jbloom SAE) vs the **non-causal**
  ESM-2 control, matched N. Trained-`E` beats `pinv` cleanly on GPT-2 (+0.031 @n=128, +0.070 @n=512, layer-
  robust) but ≈ ties on ESM-2 (+0.0016 / 0.000) → **the null is non-causal-specific**, as caveat (iv)
  predicted. Compression-regime & layer confounds ruled out; SAE-type (ReLU vs TopK) is the standing confound.
  Full writeup + table in `proposal.md` "Causal-LM control RESULT"; data in `causal_lm_forge_gate_results.json`.
  NOTE: the *full multi-layer* GPT-2 forge stays blocked (mid-layer SAE vs final-logits-only forge) — future
  plumbing, tracked in proposal "What this does NOT solve".

## 0. Design pre-locks (blocking)

- [x] 0.1 **Differentiability spike (the gating risk):** on a tiny synthetic ESM-2, confirm autograd reaches
  a grad-enabled `E` end-to-end through the projected-weight forward — `loss.backward()` populates `E.grad`
  with a finite, nonzero tensor. If a layer breaks the graph (detach / numpy / in-place), name it and decide
  the torch-native re-expression before any other work. This de-risks the whole change.
- [x] 0.2 Confirm `differentiable_forge_h` reproduces the numpy `project_module → NativeModel → forward`
  output to tolerance when `E = pinv(W_dec)*scale_boost` (the differentiable path must match the inference
  forge at the baseline `E`, else the gate compares apples to oranges).
- [x] 0.3 Lock `E`-only matched capacity (host weights / `W_dec` / encoder are fixed buffers, no grad) and
  the held-out, scoring-only-AUC conventions inherited from `add-capability-trained-encoder`.
- [x] 0.4 Lock the v1 family scope: `esm2` differentiable; all other families raise `NotImplementedError`
  (no silent fallback to the proxy objective).

## 1. `saeforge/forge_diff.py` — the differentiable forge forward

- [x] 1.1 New module. `differentiable_forge_h(host, basis, E, input_ids, *, aggregator, feed, device)`:
  build the `esm2` forged forward in torch with `E`-projected weights (`D@W`, `W@E`, `D@W@E`) as functions of
  the grad-enabled `E`; run the forward; return the aggregated forged hidden state `(N, d_model)` with grad.
- [x] 1.2 Non-`esm2` host families raise `NotImplementedError` naming the family + that it's a follow-up.
- [x] 1.3 Tests `tests/test_forge_diff.py`: (a) baseline match — `E = pinv·scale` reproduces the numpy forge
  `forged_h` to tolerance; (b) autograd — `E.grad` is finite/nonzero after a dummy loss; (c) non-esm2 raises.

## 2. `saeforge/training/encoder.py` — the `forge_distill` objective

- [x] 2.1 Add `objective="forge_distill"` to `train_encoder` (alongside `distill` / `supervised`). Loss =
  `dist(host_encoder(differentiable_forge_h(E, seqs)), host_encoder(host_X))`, default cosine; needs the host
  + sequences + feed/aggregator threaded through (a `forge_ctx` arg bundling them).
- [x] 2.2 Minibatch the sequences per step (Decision 6); precompute the host-latent target once; keep the
  held-out split, `overfit_flag`, early-stop, and the scoring-only AUC from the parent change.
- [x] 2.3 Tests `tests/test_train_encoder_forge.py`: a synthetic ESM-2-shape fixture where the full-forge
  objective runs, `E` updates, the report fields populate, and held-out scoring uses the full forge.

## 3. `saeforge/sweep_capability.py` — `train_objective`

- [x] 3.1 `sweep_pareto_capability(..., train_objective="proxy")` ∈ {`"proxy"`, `"full_forge"`}; when
  `full_forge`, `_run_capability_cell` fits `E` via the `forge_distill` objective (passing host + sequences
  + feed) instead of the activation proxy. `"proxy"` default = byte-identical to the parent change.
- [x] 3.2 Test: a tiny-host e2e cell with `train_objective="full_forge"` populates the trained row fields and
  the pinv baseline column matches a `train_encoder=False` run (apples-to-apples).

## 4. `saeforge/cli.py` — flag

- [x] 4.1 `sae-forge sweep-capability --train-objective {proxy,full_forge}` (default proxy); help text notes
  full_forge is esm2-only in v1 and far heavier.

## 5. Acceptance gate (blocking merge) — multi-seed, both outcomes first-class

- [x] 5.1 `scripts/forge_trained_encoder_bio_gate.py --train-objective full_forge`, ≥3 seeds, on bio-sae
  spread (n∈{64,128,256}) + concentrated, compression-controlled, held-out. Report mean ± std of
  `delta_heldout` per width.
- [x] 5.2 **Descriptive verdict, pre-committed both ways:** SUCCESS (mean Δ > 0 clearing noise on the spread
  mid-widths → the X2 null becomes a win) OR ALSO-PLATEAUS (Δ ≈ 0 through the full forge → the tax is
  structural beyond `E`, Reckoning #5). Route into `add-full-forge-encoder-training/proposal.md` ("Gate
  RESULT") + the parent change's Decision 9 follow-up note. No "irreducible"/"closes the tax" language.
