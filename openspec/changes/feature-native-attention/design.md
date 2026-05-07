# Design: feature-native attention

## The change in one diagram

```
                      v0 (shipped)                    v0.2 feature_native
                      ─────────────                   ─────────────────────
  residual              k                               k
  c_attn weight         (k, 3·d_host)                   (k, 3·k)
  Q, K, V each          d_host wide                     k wide
  head_dim              host head_dim                   k // num_heads
  attention scores      d_host inner space              k inner space
  c_proj weight         (d_host, k)                     (k, k)
  MLP intermediate      h_ff   (unchanged)              h_ff   (unchanged)
```

MLP genuinely doesn't change — see `Out of scope` in the proposal.

## Projection algebra

`D ≡ W_dec` shape `(k, d)`. `E ≡ pinv(W_dec)` shape `(d, k)`.

Three new identities for v0.2 attention:

```
W_c_attn_v1   (k, 3k)   = D @ W_c_attn_host @ block_diag(E, E, E)
b_c_attn_v1   (3k,)     = block_diag(E, E, E).T @ b_c_attn_host
                        = concat(b_q_v1, b_k_v1, b_v_v1)
                          where each b_*_v1 = E.T @ b_*_host
W_c_proj_v1   (k, k)    = D @ W_c_proj_host @ E
b_c_proj_v1   (k,)      = E.T @ b_c_proj_host    (already in v0)
```

The QKV bias decomposes into Q, K, V triple, each of which is a
residual-bias projection. The QKV weight needs the QKV-output
dimension projected three times because the c_attn weight stores Q,
K, V concatenated horizontally.

`project_qkv_full(W: (d, 3d)) -> (k, 3k)` implements the block-diagonal
construction:

```python
W_q, W_k, W_v = np.split(W, 3, axis=1)   # each (d, d)
return np.concatenate([
    self.project_residual_full(W_q),
    self.project_residual_full(W_k),
    self.project_residual_full(W_v),
], axis=1)
```

## Why MLP is genuinely out of scope

The spec's §4 formula `W_in_new = E.T @ W_in @ B.T` is dimensionally
inconsistent unless `h_ff = d`. For HF GPT-2, `h_ff = 4d`, so the
formula doesn't typecheck. The natural reading is that the MLP
intermediate is *not* residual-aligned and should not project — only
the residual-touching edges (`W_in`'s input side, `W_out`'s output
side) do. This is what v0 already ships and v0.2 keeps unchanged.

A separate `feature-native-mlp` change could explore making the MLP
intermediate k-wide by training a fresh `(k, k_intermediate)` and
`(k_intermediate, k)` pair against the host's MLP outputs. That's a
research question — does fine-tuning recover the MLP capacity at a
smaller intermediate? — independent of attention.

## The k-divisibility constraint

`num_heads × head_dim = k` is the standard transformer constraint
applied to the basis width. v0.2 enforces it at config-construction
time:

```
if attention_width == "feature_native" and hidden_size % num_heads != 0:
    raise ValueError(
        f"feature-native attention requires hidden_size ({hidden_size}) "
        f"to be divisible by num_heads ({num_heads}); set num_heads to "
        f"a divisor of hidden_size or pad the basis."
    )
```

Default behaviour: `num_heads` is inherited from the host. Users
working with awkward `n_features` (prime, near-prime) can override to
fewer heads or pad the basis with zero-norm rows during
`from_polygram_checkpoint`. The padding path is a follow-up
(`feature-basis-padding`) — for v0.2, the user picks compatible
values.

## Identity-basis sanity check still works

The v0.1 forge-pipeline test pins `KL(host || forged) < 1e-3` when
the basis is the d×d identity. Under feature-native attention with
`W_dec = I` (so `k = d`, `D = E = I`):

```
W_c_attn_v1 = I @ W_c_attn @ block_diag(I, I, I) = W_c_attn
W_c_proj_v1 = I @ W_c_proj @ I = W_c_proj
```

Both reduce to identity. The forged model is byte-identical to the
host modulo float64↔float32 conversions. So the sanity check keeps
working as a v0.2 correctness signal — and it strictly catches more
bugs than the v0 version, since now it covers the both-sides-projected
attention path too.

## Faithfulness expectations on non-trivial bases

When `W_dec` doesn't span the full residual (the realistic case after
SAE compression), feature-native attention will produce **higher** KL
than v0 host-mode on the same input. That's not a regression — it's
the cost of making attention scores live in feature space. The v0
host-mode preserves softmax-over-d-space exactly; v0.2 feature-native
rewrites it as softmax-over-k-space.

The fine-tune step in the FSM is what closes the gap. The v0.2
acceptance test does **not** assert "feature-native KL ≤ v0 KL"
because that's an empirical research question that depends on basis
quality. The acceptance test asserts:

1. The two modes produce *different* forged weights on a non-trivial
   basis (regression check that we actually changed something).
2. Identity-basis still holds (the algebra is right).
3. End-to-end runs to `done` without raising.

The interesting empirical comparison ("does feature-native attention
recover capability faster after N fine-tune steps?") lives in
`docs/research/`, not in the v0.2 acceptance criteria.

## Why default stays "host" in v0.2

Two reasons:

1. **Byte-equivalence with the v0.1 byte-equivalence test.** The
   imperative/FSM safety net in `forge-outer-loop-fsm` requires the
   default forge output to remain byte-stable across orchestrators.
   Flipping the default attention mode would invalidate every cached
   v0.1 forged checkpoint and force the safety-net test to be
   rewritten.
2. **The empirical question isn't settled.** Until we have a
   compression-quality story strong enough to claim feature-native
   attention recovers faster than host-attention under fine-tuning,
   the conservative default is the one with documented faithfulness
   guarantees. Flipping the default in v1.0 is conditional on that
   evidence — track it in a separate research write-up.

## What v1.0 looks like

The follow-up `feature-native-attention-default` change:

- Flips the default to `attention_width="feature_native"`.
- Adds an explicit `--legacy-host-attention` flag for one milestone.
- Updates the byte-equivalence test to compare orchestrators within
  each mode (host-vs-host, feature-native-vs-feature-native), not
  across modes.
- Removes the `host` mode in v1.1 once the deprecation window expires.

That's a separate proposal so this change can ship behind a flag
without committing to the default flip.

## Open questions deferred

1. **Padding the basis to a head-friendly size.** When `n_features`
   isn't divisible by any reasonable `num_heads`, do we pad with
   zero-norm rows or auto-reduce `num_heads`? Defer to
   `feature-basis-padding`; for v0.2, the user picks compatible
   values.
2. **MLP intermediate as k-wide.** Discussed in "Out of scope" —
   research question, not spec-conformance. If signal warrants, open
   `feature-native-mlp`.
3. **What does "fine-tune to convergence" mean for feature-native
   attention?** v0 fine-tune ships 4 AdamW steps as a smoke test.
   v0.2 will need a more serious fine-tune story to demonstrate
   capability recovery; that's `forge-finetune-recipe`.
