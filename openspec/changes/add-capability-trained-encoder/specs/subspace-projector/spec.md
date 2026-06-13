# subspace-projector Specification (delta)

## ADDED Requirements

### Requirement: Optional trained `encoder_override`

`SubspaceProjector` SHALL accept an optional `encoder_override: np.ndarray | None`
field (default `None`). When `None`, the projector's behavior SHALL be
byte-identical to the current `encode` path (`x @ pinv(W_dec) * scale_boost`) —
this change SHALL NOT alter any existing forge.

When `encoder_override` is not `None`:

- It SHALL be a 2-D array of shape exactly `(basis.d_model, basis.n_features)` —
  the same shape as `pinv(W_dec)` (matched capacity; the "tied" design). Construction
  SHALL raise `ValueError` on `ndim != 2` or any other shape, with a message naming
  the expected and observed shapes.
- It SHALL be cast to `basis.W_dec`'s dtype at construction time.
- It SHALL be treated as the **full** encode map: `encode(x)` SHALL return
  `x @ encoder_override` and SHALL NOT re-multiply by `scale_boost` (the override
  already absorbs scale; re-multiplying would double-apply it through a
  `train_encoder` round trip that initializes at `pinv(W_dec) * scale_boost`).
- `decode(z)` SHALL remain `z @ basis.W_dec`, unchanged.

The `scale_boost` field and its `"auto"` resolution SHALL continue to validate and
resolve as today, but SHALL have no effect on `encode` while `encoder_override` is set.

#### Scenario: `encoder_override=None` preserves the existing encode exactly

- **WHEN** a `SubspaceProjector` is constructed with `encoder_override=None` (the default)
- **THEN** `encode(x)` SHALL equal `x @ basis.pseudoinverse() * scale_boost` for all `x`,
  bit-for-bit with the pre-change implementation

#### Scenario: an override equal to `pinv(W_dec)*scale_boost` reproduces the default

- **WHEN** a `SubspaceProjector` is constructed with
  `encoder_override = basis.pseudoinverse() * resolved_scale_boost`
- **THEN** `encode(x)` SHALL equal the default (`encoder_override=None`) projector's
  `encode(x)` to within floating-point tolerance

#### Scenario: a mismatched-shape override is rejected

- **WHEN** a `SubspaceProjector` is constructed with an `encoder_override` whose shape
  is not `(d_model, n_features)` (e.g. transposed, or wrong feature count)
- **THEN** construction SHALL raise `ValueError` naming the expected shape
  `(d_model, n_features)` and the observed shape

#### Scenario: the override is the full map (scale not re-applied)

- **GIVEN** a basis with `scale_boost = 0.25`
- **WHEN** a `SubspaceProjector` is constructed with `encoder_override = E` for some `E`
- **THEN** `encode(x)` SHALL equal `x @ E` (NOT `x @ E * 0.25`)
