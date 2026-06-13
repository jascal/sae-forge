# pareto-sweep Specification (delta)

## ADDED Requirements

### Requirement: Causal-host, layer-targeted activation extraction (`host_layer`)

`sweep_pareto_capability` SHALL accept `host_layer: int | None = None`. When `host_layer` is set, the host and
forged activations SHALL be read from the residual stream **at that layer** (the SAE's hook point), so the
full-forge capability gate runs on **causal LM hosts** (whose SAEs live mid-model) and not only on encoder-only
hosts whose SAE reads the final hidden state.

Constraints:

- `host_layer=None` SHALL preserve the existing encoder-only (ESM-2) extraction **byte-identically**:
  `host.esm` / `out.last_hidden_state` with the `[0, 1:-1, :]` CLS/EOS strip.
- When `host_layer` is set, host extraction SHALL run the HF host with `output_hidden_states=True` and take
  `hidden_states[host_layer]` (= `blocks.{host_layer}.hook_resid_pre`) with **no** `[1:-1]` strip (causal LMs
  have no CLS/EOS), shaped per `feed` (`"residue"` per-token or `"pooled"` mean-pooled) as today.
- When `host_layer` is set, forged extraction SHALL capture the forged module's **basis-space** residual at
  the layer's input via a forward-pre-hook on the family's block list (GPT-2: `transformer.h[host_layer]`),
  WITHOUT changing the forged forward signature; the captured `(seq, N)` tensor SHALL be remapped to `d_model`
  by the existing `forged_h @ W_dec` step before the downstream encoder, exactly as for ESM-2.
- v1 SHALL support the `gpt2` causal family; other causal families (Llama/Gemma-2/Qwen) SHALL raise
  `NotImplementedError` naming the family and its block-path. There SHALL be no silent fallback to final-layer
  or to the ESM path.

#### Scenario: causal host scored at its SAE layer

- **GIVEN** a GPT-2 host and an SAE trained at `blocks.8.hook_resid_pre`
- **WHEN** `sweep_pareto_capability(host_model_id="gpt2", host_layer=8, ...)` runs a cell
- **THEN** host activations SHALL be `hidden_states[8]` and forged activations SHALL be the forged module's
  basis-space `resid_pre` at block 8, both with aligned row counts, fed to the SAE encoder for retained-mAUC

#### Scenario: encoder-only host unchanged

- **GIVEN** an ESM-2 host and `host_layer=None`
- **WHEN** the sweep extracts host and forged activations
- **THEN** it SHALL use the `last_hidden_state` + `[1:-1]` path byte-identically to the parent behaviour

#### Scenario: unsupported causal family is rejected

- **GIVEN** a non-`gpt2` causal host and `host_layer` set
- **WHEN** forged extraction runs
- **THEN** it SHALL raise `NotImplementedError` naming the family and the block-path to wire

### Requirement: SAELens-format SAE loading

`_load_encoding_state` SHALL load SAE checkpoints in the **SAELens** key convention (`W_dec` shaped
`(n_features, d_model)`, `W_enc` shaped `(d_model, n_features)`) used by LM SAEs, in addition to the existing
`decoder.weight` `(d_model, n_features)` / `encoder.weight` convention. The format SHALL be detected by key
presence and the resulting `W_dec_full` SHALL be `(n_features, d_model)` for both conventions.

#### Scenario: SAELens and reference state dicts agree on shape

- **GIVEN** a SAELens state dict (`W_dec` `(F, d)`) and a reference state dict (`decoder.weight` `(d, F)`)
  for the same dictionary
- **WHEN** `_load_encoding_state` loads each
- **THEN** both SHALL yield a `(F, d)` `W_dec_full` and identical row norms
