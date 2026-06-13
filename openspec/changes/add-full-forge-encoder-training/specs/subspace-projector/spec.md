# subspace-projector Specification (delta)

## ADDED Requirements

### Requirement: Differentiable forge forward `differentiable_forge_h`

`saeforge.forge_diff.differentiable_forge_h` SHALL produce a forged hidden state that is
**differentiable with respect to the encoder `E`**, so `E` can be trained against the
full-forge metric (not the activation proxy). The numpy `SubspaceProjector.project_module`
path SHALL remain unchanged and detached — this is an additive, training-only surface.

Signature:

```python
differentiable_forge_h(
    host, basis, E: "torch.Tensor",      # E grad-enabled, shape (d_model, n_features)
    input_ids: "torch.Tensor",
    *, aggregator, feed, device="cpu",
) -> "torch.Tensor"                       # (n_items, d_model), requires_grad w.r.t. E
```

Constraints:

- `E` SHALL be the **only** grad-carrying input: the host weights, `basis.W_dec`, and the
  downstream encoder SHALL be fixed (no grad). The forged weights SHALL be built as the
  documented projection algebra (`D @ W`, `W @ E`, `D @ W @ E`) expressed in torch ops on
  `E`, so autograd reaches `E` end-to-end.
- At `E == pinv(W_dec) * scale_boost`, the returned `forged_h` SHALL match the numpy
  `project_module → NativeModel → forward` output to numerical tolerance (the differentiable
  path is the *same* forge at the baseline `E`).
- v1 SHALL implement the `esm2` host family. Other families (`gpt2`/`llama`/`gemma2`/
  `whisper`) SHALL raise `NotImplementedError` naming the family and that the differentiable
  forward is a follow-up — SHALL NOT silently fall back to the activation proxy.

#### Scenario: autograd reaches E end-to-end

- **GIVEN** a tiny `esm2` host, a basis, and `E` with `requires_grad=True`
- **WHEN** a scalar loss on `differentiable_forge_h(...)` is backpropagated
- **THEN** `E.grad` SHALL be a finite, nonzero tensor of shape `(d_model, n_features)`

#### Scenario: baseline E reproduces the inference forge

- **WHEN** `differentiable_forge_h` is called with `E = pinv(basis.W_dec) * scale_boost`
- **THEN** its `forged_h` SHALL equal the numpy `project_module → NativeModel → forward`
  forged hidden state (same host, same inputs) to numerical tolerance

#### Scenario: non-esm2 family is refused, not silently downgraded

- **WHEN** `differentiable_forge_h` is called with a `gpt2` / `llama` / `gemma2` / `whisper` host
- **THEN** it SHALL raise `NotImplementedError` naming the family — and SHALL NOT fall back to
  the activation-proxy path
