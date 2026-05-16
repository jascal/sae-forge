# M4 Axis-4 runbook — HEA_Rung2(n_qubits=9) + cross-encoding on real GPT-2

End-to-end recipe for running a real Axis-4 sweep on the M4 box. Targets
the **first empirical measurement** of whether
`rep_selection="kl_attribution"` (polygram 0.5.0) Pareto-dominates
`scale_aware` in structurally-feasible regimes — specifically, frontier
rows that the `quality_tier` diagnostic tags as `good` or `saturated`.

The Intel Mac validated the chain end-to-end (PR #32–#39 all merged) but
was capped at the `degenerate` regime because of compute limits. The M4
unblocks larger feature counts (N=512) where the basis can actually
span GPT-2-small's 768-dim residual.

## Clarification on rungs

Polygram's encoding ladder, with **max-features caps**:

| Encoding | Cap |
|---|---|
| `MPSRung1` | 8 |
| `Rung3` | 16 |
| `Rung4` | **32 — the ceiling, not negotiable** |
| `HEA_Rung2(n_qubits=N)` | `2^N` (e.g., `n_qubits=9` → 512) |

**There is no `Rung5`.** "Higher-than-Rung4" requires either
`HEA_Rung2(n_qubits=N)` with N ≥ 6 (cap=64+) or `Rung4(bond_dim=K)`
with K > 2 (more rank capacity at fixed N≤32).

The primary M4 protocol below uses `HEA_Rung2(n_qubits=9)` (cap=512)
because that's the only polygram encoding that can fit the N=512
feature count needed to escape the `degenerate` tier on GPT-2-small.
**Rung4 mechanically cannot reach the `good` tier on a 768-dim host** —
its cap of 32 features puts the best-case `quality_ratio` at 32/768 ≈
0.04, below the 0.0625 `degenerate` boundary.

Rung4 still has a role: see **Run #4 (optional)** for a Rung4-vs-
HEA_Rung2(n_qubits=5) capability-surface test at N=32 (both
encodings' native or near-native caps).

If you want to compare **`Rung4(bond_dim=2)` vs `Rung4(bond_dim=4)`** —
informally "Rung 5" — the CLI doesn't expose `--encoding-bond-dim`.
Use the Python API (*Python escape hatch* at the bottom).

## Prereqs (one-off)

```bash
# 1. Clone/sync sae-forge on the M4
git clone git@github.com:jascal/sae-forge.git ~/code/sae-forge
cd ~/code/sae-forge

# 2. Venv with the M4-appropriate extras (NOT [intel]; that's the Intel ceiling)
python3.11 -m venv .venv
.venv/bin/pip install -e ".[torch,polygram,recipe]"
.venv/bin/pip install huggingface_hub safetensors

# 3. Sanity check the version set
.venv/bin/python -c "
import saeforge, polygram, transformers, torch
print(f'sae-forge {saeforge.__version__} | polygram {polygram.__version__} | '
      f'transformers {transformers.__version__} | torch {torch.__version__} | '
      f'mps_built={torch.backends.mps.is_built()}')
"
# Expected: polygram >= 0.5.0; torch 2.x; mps_built=True
```

## Bootstrap (one-off, ~30s)

Slice the jbloom GPT-2 layer-8 SAE to N=512 features. The N=512 choice
is deliberate: to escape `quality_tier="degenerate"` (ratio < 0.0625
= 1/16) against GPT-2's 768-dim residual, we need `basis_rank >= 384`
(half-coverage = `quality_tier="good"`). N=512 gives the basis loop
enough headroom to land at least one row in the `good` tier after
compression.

```bash
mkdir -p /tmp/axis4_m4 && .venv/bin/python <<'PY'
"""Slice jbloom SAE to N=512 stride-sampled features."""
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file, save_file

sae_path = Path(hf_hub_download(
    repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
    filename="blocks.8.hook_resid_pre/sae_weights.safetensors",
))
state = load_file(str(sae_path))
N = 512
STRIDE = max(1, state["W_dec"].shape[0] // N)
fids = [i * STRIDE for i in range(N)]
out = Path("/tmp/axis4_m4/sae_N512.safetensors")
save_file({
    "W_dec": state["W_dec"][fids],
    "W_enc": state["W_enc"][:, fids] if state["W_enc"].shape[1] == state["W_dec"].shape[0] else state["W_enc"][fids],
    "b_enc": state["b_enc"][fids],
    "b_dec": state["b_dec"],
}, str(out))
print(f"sliced N={N} stride={STRIDE} → {out}")
PY
```

### Prompts (distinct files — the leakage firewall enforces this at parse time)

Write **two separate prompt files**. The CLI refuses if
`--validation-prompts` and `--eval-prompts` resolve to the same path
unless `--allow-validation-eval-overlap` is set (which then surfaces as
`validation_eval_overlap=true` in every frontier row).

```bash
# Validation prompts (60 lines — larger than the Intel smoke for higher signal)
.venv/bin/python <<'PY'
import textwrap
prompts = textwrap.dedent('''
The quick brown fox jumps over the lazy dog.
In a hole in the ground there lived a hobbit.
Newton's third law states that every action has an equal and opposite reaction.
Photosynthesis converts light energy into chemical energy stored in glucose.
The capital of France is Paris, a city renowned for its art and architecture.
All happy families are alike; each unhappy family is unhappy in its own way.
The mitochondrion is the powerhouse of the cell, generating most ATP.
To be or not to be, that is the question.
It was the best of times, it was the worst of times.
Four score and seven years ago our fathers brought forth a new nation.
Call me Ishmael. Some years ago, never mind how long.
She sells seashells by the seashore.
A long time ago in a galaxy far far away.
It is a truth universally acknowledged that a single man in possession of a good fortune.
When in the course of human events it becomes necessary to dissolve.
Two roads diverged in a yellow wood and I took the one less traveled.
The only thing we have to fear is fear itself.
Ask not what your country can do for you, ask what you can do for your country.
I have a dream that one day on the red hills of Georgia the sons of slaves.
We hold these truths to be self-evident that all men are created equal.
The earth orbits the sun once every 365.25 days completing a full revolution.
Water boils at 100 degrees Celsius at sea level under standard atmospheric pressure.
DNA carries the genetic instructions used in the growth and reproduction of organisms.
Light travels at approximately 299792 kilometers per second in a vacuum.
The Pythagorean theorem relates the lengths of the sides of a right triangle.
Shakespeare wrote thirty-seven plays and over one hundred and fifty sonnets.
The Pacific Ocean covers approximately one-third of the Earth surface area.
Beethoven composed nine symphonies including the famous Choral Symphony.
The Renaissance was a fervent period of European cultural artistic political revival.
The cat sat on the mat and watched the rain fall against the window.
The sun rises in the east and sets in the west every single day.
A journey of a thousand miles begins with a single step forward.
Time and tide wait for no man regardless of station or fortune.
Knowledge is power but wisdom is knowing how to wield that power.
The pen is mightier than the sword in the right hands of a writer.
Practice makes perfect when accompanied by genuine attention and care.
A picture is worth a thousand words to those who can read it.
Actions speak louder than words in the long run of any relationship.
Necessity is the mother of invention especially in dire circumstances.
Where there is smoke there is fire somewhere nearby waiting to flare.
The early bird catches the worm before the others have woken up.
Don't count your chickens before they hatch or your eggs in one basket.
A stitch in time saves nine when the fabric is still strong enough.
Beauty is in the eye of the beholder regardless of conventional standards.
Curiosity killed the cat but satisfaction brought it back to life.
Don't judge a book by its cover until you have read at least a chapter.
Every cloud has a silver lining if you are willing to look hard enough.
Fortune favors the bold who dare to step beyond the safety of routine.
Great minds think alike but fools rarely differ in their conclusions.
Honesty is the best policy in matters both small and consequential.
If you want something done right you have to do it yourself sometimes.
Just because you can does not always mean that you should do it.
Keep your friends close and your enemies closer for they teach you most.
Look before you leap into any decision that affects your future significantly.
Make hay while the sun shines and the conditions are favorable to growth.
No pain no gain in any worthwhile pursuit that demands true commitment.
Out of sight is not necessarily out of mind for those who truly care.
Practice what you preach if you want others to take you seriously.
Quality matters more than quantity in nearly every domain of human endeavor.
Rome was not built in a day and neither is any lasting achievement.
''').strip()
open('/tmp/axis4_m4/validation_prompts.jsonl', 'w').write(prompts + '\n')
print(f'wrote {len(prompts.splitlines())} validation prompts')
PY

# Eval prompts (10 distinct lines — content must NOT overlap validation)
.venv/bin/python <<'PY'
import textwrap
prompts = textwrap.dedent('''
She opened the book and began to read about the ancient history of magic.
The chef carefully selected ingredients for tonight's signature dish.
Quantum mechanics describes the behavior of matter at the smallest scales.
The mountain climber reached the summit just as the sun broke over the ridge.
Children's laughter echoed through the empty halls of the old schoolhouse.
The detective examined every clue with meticulous attention to detail.
Spring brought new life to the garden after the long cold winter months.
The orchestra rehearsed the final movement until each note rang true.
Astronomers discovered a new exoplanet in the habitable zone of its star.
The blacksmith hammered the glowing iron into the shape of a horseshoe.
''').strip()
open('/tmp/axis4_m4/eval_prompts.jsonl', 'w').write(prompts + '\n')
print(f'wrote {len(prompts.splitlines())} eval prompts')
PY
```

## Critical: encoding caps vs N

**Rung4 caps the Dictionary at 32 features. It cannot fit N=512.** This
runbook's earlier draft (pre-fix) incorrectly recommended `Rung4` at
N=512, which fails with:

```
ValueError: selected 512 features, but the Rung4 encoding caps a
Dictionary at 32 features.
```

The encoding cap ladder (polygram 0.5.0):

| Encoding | Cap |
|---|---|
| `MPSRung1` | 8 |
| `Rung3` | 16 |
| `Rung4` | **32** (this is the ceiling — not negotiable) |
| `HEA_Rung2(n_qubits=N)` | `2^N` (e.g., `n_qubits=9` → 512) |

To escape `quality_tier="degenerate"` on GPT-2-small's 768-dim residual
(`basis_rank ≥ 384` for the `good` tier), we need N ≥ ~512 features.
**The only polygram encoding that supports N=512 is `HEA_Rung2(n_qubits=9)`.**
Rung4 mechanically cannot reach the `good` tier on this host.

The runbook below uses `HEA_Rung2(n_qubits=9)` as the primary encoding.
Cross-encoding comparisons against `Rung4` happen at N=32 (Rung4's
native cap), where both rep_selection arms will land in `degenerate`
tier — informative as a capability-surface test, not a research-grade
comparison.

## Sanity-check the encoding BEFORE every run

A real-world footgun: if `--encoding LABEL:PATH` and `--encoding-qubits
LABEL:N` use different `LABEL` strings, the CLI silently falls back to
polygram's `HEA_Rung2()` default (`n_qubits=3`, cap=8) — no error.
**Always run `--plan-only` first and inspect the `encoding_kwargs=` line**
in the per-encoding plan block. If it shows `{'depth': 2}` or
`{'n_qubits': 3, 'depth': 2}`, the labels don't match.

Expected `encoding_kwargs=` line for the primary run below:
**`{'n_qubits': 9, 'depth': 2}`**.

## The runs

### Pre-flight (free)

```bash
.venv/bin/python -m saeforge.cli sweep-pareto --auto-materialise --plan-only \
    --encoding hea9:/tmp/axis4_m4/sae_N512.safetensors \
    --encoding-class hea9:HEA_Rung2 \
    --encoding-qubits hea9:9 \
    --host-model gpt2 --layer 8 \
    --pareto 64,128,256,512 \
    --validation-prompts /tmp/axis4_m4/validation_prompts.jsonl \
    --eval-prompts /tmp/axis4_m4/eval_prompts.jsonl \
    --validation-threshold 0.95 \
    --rep-selection scale_aware \
    --device mps --dtype float32 \
    --output-dir /tmp/axis4_m4/scale_aware/
```

Expected output keys:

- `cache_status=MISS (cold)` (first run only)
- `targets=[64, 128, 256, 512]`
- `encoding_class=HEA_Rung2`
- `encoding_kwargs={'n_qubits': 9, 'depth': 2}` ← **verify this**
- `validator_forward_count_estimate=~600` (60 prompts × ~10 tokens avg)

If `encoding_kwargs` is missing `n_qubits` or shows `n_qubits=3`,
DOUBLE-CHECK that the `--encoding` and `--encoding-qubits` flags use
the **identical** label string (case-sensitive, no whitespace).

### Run #1 — HEA_Rung2(n_qubits=9) with scale_aware (~15–25 min on M4)

```bash
.venv/bin/python -m saeforge.cli sweep-pareto --auto-materialise \
    --encoding hea9:/tmp/axis4_m4/sae_N512.safetensors \
    --encoding-class hea9:HEA_Rung2 \
    --encoding-qubits hea9:9 \
    --host-model gpt2 --layer 8 \
    --pareto 64,128,256,512 \
    --validation-prompts /tmp/axis4_m4/validation_prompts.jsonl \
    --eval-prompts /tmp/axis4_m4/eval_prompts.jsonl \
    --validation-threshold 0.95 \
    --rep-selection scale_aware \
    --device mps --dtype float32 \
    --output-dir /tmp/axis4_m4/scale_aware/
```

### Run #2 — HEA_Rung2(n_qubits=9) with kl_attribution (~15–25 min, fresh cache key)

`rep_selection` is part of the cache key, so this re-materialises
cleanly without contaminating Run #1's artifacts. **This is the
load-bearing comparison** — the K=512 row in this output, compared to
the K=512 row in Run #1, is the first empirical signal on whether
`kl_attribution` Pareto-dominates `scale_aware` in the good tier.

```bash
.venv/bin/python -m saeforge.cli sweep-pareto --auto-materialise \
    --encoding hea9:/tmp/axis4_m4/sae_N512.safetensors \
    --encoding-class hea9:HEA_Rung2 \
    --encoding-qubits hea9:9 \
    --host-model gpt2 --layer 8 \
    --pareto 64,128,256,512 \
    --validation-prompts /tmp/axis4_m4/validation_prompts.jsonl \
    --eval-prompts /tmp/axis4_m4/eval_prompts.jsonl \
    --validation-threshold 0.95 \
    --rep-selection kl_attribution \
    --device mps --dtype float32 \
    --output-dir /tmp/axis4_m4/kl_attribution/
```

### Run #3 (optional) — Cross-encoding HEA_Rung2(n_qubits=9) vs (n_qubits=10) at N=512 (~30–40 min)

The "Rung 4 vs Rung 5" intuition. Both encodings fit N=512; the
n_qubits=10 has more rank slack (cap=1024 vs cap=512). If the higher-
qubit encoding produces a meaningfully different validator-confirmed-
pair set, that's the Axis-4 cross-encoding signal.

You'll need to slice a second SAE (or pad the existing one with extra
features) for the n_qubits=10 path, since both encodings need their
input feature count to be within their cap. The simplest path: reuse
the N=512 slice for both labels — n_qubits=10 will have empty rank
headroom but the comparison is still valid.

```bash
.venv/bin/python -m saeforge.cli sweep-pareto --auto-materialise \
    --encoding hea9:/tmp/axis4_m4/sae_N512.safetensors \
    --encoding hea10:/tmp/axis4_m4/sae_N512.safetensors \
    --encoding-class hea9:HEA_Rung2 \
    --encoding-class hea10:HEA_Rung2 \
    --encoding-qubits hea9:9 \
    --encoding-qubits hea10:10 \
    --host-model gpt2 --layer 8 \
    --pareto 64,128,256,512 \
    --validation-prompts /tmp/axis4_m4/validation_prompts.jsonl \
    --eval-prompts /tmp/axis4_m4/eval_prompts.jsonl \
    --validation-threshold 0.95 \
    --rep-selection kl_attribution \
    --device mps --dtype float32 \
    --output-dir /tmp/axis4_m4/cross_encoding/
```

### Run #4 (optional) — Rung4 vs HEA_Rung2(n_qubits=5) at N=32 (capability-surface test)

This is the **only honest "Rung4 in the comparison" command**, run at
Rung4's native cap of 32. Both encodings will land in the `degenerate`
tier (32/768 ≈ 0.04 < 0.0625 threshold), so this is **not the
rep_selection research question** — it's a capability-surface test
that verifies multi-encoding sweep mechanics work on a real fixture.

Requires a separate N=32 slice:

```bash
.venv/bin/python <<'PY'
"""Slice jbloom SAE to N=32 (for the Rung4-native cross-encoding test)."""
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file, save_file

sae_path = Path(hf_hub_download(
    repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
    filename="blocks.8.hook_resid_pre/sae_weights.safetensors",
))
state = load_file(str(sae_path))
N = 32
STRIDE = max(1, state["W_dec"].shape[0] // N)
fids = [i * STRIDE for i in range(N)]
out = Path("/tmp/axis4_m4/sae_N32.safetensors")
save_file({
    "W_dec": state["W_dec"][fids],
    "W_enc": state["W_enc"][:, fids] if state["W_enc"].shape[1] == state["W_dec"].shape[0] else state["W_enc"][fids],
    "b_enc": state["b_enc"][fids],
    "b_dec": state["b_dec"],
}, str(out))
print(f"sliced N={N} → {out}")
PY

.venv/bin/python -m saeforge.cli sweep-pareto --auto-materialise \
    --encoding rung4:/tmp/axis4_m4/sae_N32.safetensors \
    --encoding hea5:/tmp/axis4_m4/sae_N32.safetensors \
    --encoding-class rung4:Rung4 \
    --encoding-class hea5:HEA_Rung2 \
    --encoding-qubits hea5:5 \
    --host-model gpt2 --layer 8 \
    --pareto 4,8,16,32 \
    --validation-prompts /tmp/axis4_m4/validation_prompts.jsonl \
    --eval-prompts /tmp/axis4_m4/eval_prompts.jsonl \
    --validation-threshold 0.95 \
    --rep-selection kl_attribution \
    --device mps --dtype float32 \
    --output-dir /tmp/axis4_m4/rung4_native/
```

## Reading the results

### Filter to the rows where rep_selection has signal

```bash
jq -r 'select(.quality_tier == "good" or .quality_tier == "saturated") |
    [.encoding_label, .target_n_features_kept, .n_features_kept_actual,
     .basis_rank, .quality_tier, .faithfulness_kl] | @tsv' \
    /tmp/axis4_m4/kl_attribution/frontier.jsonl | column -t
```

If this returns zero rows, none of your sweep landed in the
non-degenerate regime — bump N higher or check `quality_ratio` directly
to see how close you got.

### Side-by-side scale_aware vs kl_attribution (the load-bearing comparison)

```bash
.venv/bin/python <<'PY'
import json

def load(p):
    return {r["target_n_features_kept"]: r for r in (json.loads(l) for l in open(p))}

sa = load("/tmp/axis4_m4/scale_aware/frontier.jsonl")
kl = load("/tmp/axis4_m4/kl_attribution/frontier.jsonl")

print(f"{'K':>4} {'sa_kept':>8} {'sa_rank':>8} {'sa_tier':>11} {'sa_KL':>8}    "
      f"{'kl_kept':>8} {'kl_rank':>8} {'kl_tier':>11} {'kl_KL':>8}    {'Δ_KL':>8}")
print('-' * 110)
for k in sorted(sa.keys()):
    a, b = sa[k], kl[k]
    d = b["faithfulness_kl"] - a["faithfulness_kl"]
    print(f"{k:>4} {a['n_features_kept_actual']:>8} {a['basis_rank']:>8} "
          f"{a['quality_tier']:>11} {a['faithfulness_kl']:>8.4f}    "
          f"{b['n_features_kept_actual']:>8} {b['basis_rank']:>8} "
          f"{b['quality_tier']:>11} {b['faithfulness_kl']:>8.4f}    {d:+.4f}")
PY
```

## Expected frontier shape

| K target | likely `n_features_kept` | `basis_rank` | `quality_tier` | What this row tells you |
|---|---|---|---|---|
| 512 | 400–512 | 400–512 | `good` or `saturated` | **The load-bearing row.** rep_selection has room to matter. `Δ_KL` is the answer to the research question. |
| 256 | ~256 | ~256 | `undersized` | Boundary signal — does the rep_selection gap narrow as rank drops? |
| 128 | ~128 | ~128 | `undersized` | More compressed; KL rises noticeably. |
| 64 | ~64 | ~64 | `degenerate` | Both rep_selections converge here (Intel observed this empirically). |

**The K=512 row is the load-bearing one.** If `kl_attribution` KL is
meaningfully lower than `scale_aware` KL there, that's the first
empirical evidence the behavioural rep_selection Pareto-dominates the
geometric proxy in good regimes. If they're bit-identical or within
noise, that's evidence the proxy is sufficient and `scale_aware` should
stay the default.

## Caveats specifically for M4 / Apple Silicon

1. **`--dtype float32` not `bf16`.** PyTorch MPS has known kernel gaps
   on certain transformers ops in bf16. fp32 is the safe default for
   these runs.
2. **First Run #1 pays HuggingFace download cost** (~500 MB GPT-2 SAE
   + 500 MB GPT-2 weights). Subsequent runs hit the local cache.
3. **Run #2's cache key differs from Run #1** (rep_selection is part of
   the cache key) — re-materialise is correct.
4. **MPS fallback for unsupported ops.** If validation pass fails with
   an MPS-specific kernel error, prefix with
   `PYTORCH_ENABLE_MPS_FALLBACK=1`. Watch for CPU-fallback warnings —
   those ops run on CPU and may make the validator slower than
   advertised.

## If a row fails

Every row carries `host_d_model`, `basis_rank`, `quality_ratio`,
`quality_tier`, `validation_threshold`, `encoding_class`,
`validation_eval_overlap`, and `error_message`. When
`error_message != null`, the rest of the row's diagnostic fields
distinguish "structurally doomed setup" from "forge-mechanically broken."
Failures are isolated per row — the sweep continues and exits non-zero
at the end with `frontier.jsonl` still written.

## Python escape hatch — Rung4(bond_dim=4) etc.

The CLI's `--encoding-qubits` only configures HEA_Rung2's `n_qubits`.
For Rung3/Rung4's `bond_dim` (which is what people sometimes informally
call "Rung 5" when set higher than 2), use the Python API. The
auto-materialise driver is in `saeforge.auto_materialise` and the spec
is in `saeforge.auto_materialise.AutoMaterialiseSpec`:

```python
from pathlib import Path
from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector
from saeforge.auto_materialise import AutoMaterialiseSpec

specs = [
    AutoMaterialiseSpec(
        label="rung4_bd2",
        sae_checkpoint=Path("/tmp/axis4_m4/sae_N512.safetensors"),
        encoding_class="Rung4",
        encoding_kwargs={"bond_dim": 2},
    ),
    AutoMaterialiseSpec(
        label="rung4_bd4",  # the "Rung 5" interpretation
        sae_checkpoint=Path("/tmp/axis4_m4/sae_N512.safetensors"),
        encoding_class="Rung4",
        encoding_kwargs={"bond_dim": 4},
    ),
]

# Bootstrap a placeholder basis (the sweep driver hot-swaps per row).
basis = FeatureBasis.from_polygram_checkpoint(specs[0].sae_checkpoint)
projector = SubspaceProjector(basis)
pipeline = ForgePipeline(
    basis=basis, projector=projector,
    host_model_id="gpt2", dtype="float32", device="mps",
    eval_prompts=open("/tmp/axis4_m4/eval_prompts.jsonl").read().splitlines(),
)
pipeline.sweep_pareto(
    encodings=[(s.label, s.sae_checkpoint) for s in specs],
    output_dir=Path("/tmp/axis4_m4/bond_dim_compare/"),
    auto_materialise_specs=specs,
    validation_prompts=Path("/tmp/axis4_m4/validation_prompts.jsonl"),
    validation_threshold=0.95,
    layer=8,
    targets=[64, 128, 256, 512],
    score_field="polygram_overlap",
    rep_selection="kl_attribution",
)
```

## Known limitation that may bite you

**FSM compose-with-real-validation is partially broken on this
codebase**: if you wire `ForgePipeline(validation_report_path=...)`
through the FSM orchestrator (`orchestrator="fsm"`), the synth-basis
write only includes `W_dec`, but polygram's compress action needs
`W_enc` + `b_enc` + `b_dec` too. The auto-materialise sweep path above
**does not hit this** — it bypasses `_run_real_fsm` by feeding
pre-materialised SAEs back through the imperative path. But if you go
exploring with `orchestrator="fsm"` + adaptive-regrow + a real validation
report, you'll hit `ForgeFailed: no key aliasing to 'W_enc'`. The fix
is tracked under `openspec/changes/full-sae-keys-in-synth-basis/`.
