"""sweep_pareto_capability_progressive — smallest-n robust to data scale.

A multi-stage capability-aware Pareto sweep that progressively grows
protein count + narrows the active width set until the recommended
optimum *stops shifting* as data is added. The recommendation
contract is **smallest target_n_features_kept whose retained_mauc is
stable across the last K stages**, not "argmax retained_mauc on a
single sweep". The latter overfits to whatever subset of proteins
happened to be in the eval sample.

This is Occam's razor applied to forge basis selection: if a smaller
n keeps tying the larger n as you add data, the larger n's extra
features were noise-tuning, not signal-tuning — discard them. See
``openspec/changes/add-progressive-capability-sweep/proposal.md`` for
the empirical motivation (bio-sae's 10-vs-100-protein residue-feed
shift from n=16 to n=48 was the surfacing signal).

Pipeline per stage:

  1. Subsample ``dataset`` (cumulative: stage K+1 ⊇ stage K).
  2. Sweep ``sweep_pareto_capability`` over current active widths.
  3. Identify *plateau* of widths within ``plateau_tolerance`` of
     the stage's peak (always ≥ ``min_plateau_widths``).
  4. Expand to immediate ``candidate_widths`` neighbours so resolution
     can refine around the peak.
  5. Check convergence: smallest plateau-member unchanged for
     ``convergence_n_stages`` consecutive stages, retained_mauc
     variance within ``retained_mauc_tolerance``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from saeforge.sweep import ParetoFrontierRow


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvergenceTrajectoryEntry:
    """Per-stage record carried on
    :class:`ProgressiveRecommendation.convergence_trajectory`.

    External benchmarking (counting un-converged ratios across a
    corpus of progressive runs) reads this from
    ``progressive_summary.json`` without needing any in-library
    telemetry.
    """

    stage: int
    n_proteins: int
    argmin_plateau_width: int
    argmin_retained_mauc: float
    plateau_size: int
    neighbours_added: int
    shifted_from_prev_stage: bool


@dataclass(frozen=True)
class ProgressiveStageResult:
    """One stage's outcome inside a progressive sweep."""

    stage: int
    n_proteins: int
    active_widths: tuple[int, ...]
    rows: tuple[ParetoFrontierRow, ...]
    plateau_widths: tuple[int, ...]
    peak_n: int
    peak_retained_mauc: float


@dataclass(frozen=True)
class ProgressiveRecommendation:
    """The final recommendation from a progressive sweep.

    Attributes
    ----------
    target_n_features_kept:
        The smallest stable-plateau width. Pareto-optimal on
        (capability, parameter-cost).
    retained_mauc_vs_host:
        The converged width's retained_mauc on the LAST stage.
    stages_converged:
        How many consecutive stages this width has been a plateau
        member.
    converged:
        Whether ``convergence_n_stages`` was reached. False means
        the schedule was exhausted before convergence; the
        recommendation is the last-stage argmin-plateau-member but
        carries a warning.
    rationale:
        Human-readable explanation of why this width was picked.
    convergence_trajectory:
        Per-stage record (stage, n_proteins, argmin_plateau_width,
        argmin_retained_mauc, plateau_size, neighbours_added,
        shifted_from_prev_stage). On disk in
        ``progressive_summary.json`` — external benchmarking can
        count un-converged ratios without in-library telemetry.
    """

    target_n_features_kept: int
    retained_mauc_vs_host: float
    stages_converged: int
    converged: bool
    rationale: str
    convergence_trajectory: tuple[ConvergenceTrajectoryEntry, ...]


@dataclass(frozen=True)
class ProgressiveHistory:
    """Bundle of per-stage results + the final recommendation."""

    stages: tuple[ProgressiveStageResult, ...]
    recommendation: ProgressiveRecommendation

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary written to
        ``progressive_summary.json``."""
        return {
            "stages": [
                {
                    "stage": s.stage,
                    "n_proteins": s.n_proteins,
                    "active_widths": list(s.active_widths),
                    "plateau_widths": list(s.plateau_widths),
                    "peak_n": s.peak_n,
                    "peak_retained_mauc": s.peak_retained_mauc,
                    "n_rows": len(s.rows),
                }
                for s in self.stages
            ],
            "recommendation": {
                "target_n_features_kept": self.recommendation.target_n_features_kept,
                "retained_mauc_vs_host": self.recommendation.retained_mauc_vs_host,
                "stages_converged": self.recommendation.stages_converged,
                "converged": self.recommendation.converged,
                "rationale": self.recommendation.rationale,
                "convergence_trajectory": [
                    {
                        "stage": e.stage,
                        "n_proteins": e.n_proteins,
                        "argmin_plateau_width": e.argmin_plateau_width,
                        "argmin_retained_mauc": e.argmin_retained_mauc,
                        "plateau_size": e.plateau_size,
                        "neighbours_added": e.neighbours_added,
                        "shifted_from_prev_stage": e.shifted_from_prev_stage,
                    }
                    for e in self.recommendation.convergence_trajectory
                ],
            },
        }


# ---------------------------------------------------------------------------
# Pure helpers (testable without forge)
# ---------------------------------------------------------------------------


def _identify_plateau(
    rows: Sequence[ParetoFrontierRow],
    *,
    plateau_tolerance: float,
    min_plateau_widths: int,
) -> tuple[tuple[int, ...], float, int]:
    """Find the plateau widths for a stage's row set.

    Returns ``(plateau_widths_sorted, peak_retained_mauc, peak_n)``.
    Plateau is the set of widths within ``plateau_tolerance`` of the
    peak retained_mauc; if that set is smaller than
    ``min_plateau_widths``, widen the effective tolerance to include
    the top ``min_plateau_widths`` widths by retained_mauc.
    """
    successes = [
        r for r in rows
        if r.error_message is None and r.retained_mauc_vs_host is not None
    ]
    if not successes:
        return ((), float("nan"), -1)
    peak = max(successes, key=lambda r: r.retained_mauc_vs_host)
    peak_retained = float(peak.retained_mauc_vs_host)
    threshold = peak_retained - plateau_tolerance
    plateau = [
        r for r in successes if r.retained_mauc_vs_host >= threshold
    ]
    if len(plateau) < min_plateau_widths:
        # Widen: take the top ``min_plateau_widths`` rows by retained_mauc.
        plateau = sorted(
            successes, key=lambda r: -(r.retained_mauc_vs_host or 0.0),
        )[:min_plateau_widths]
    return (
        tuple(sorted(r.target_n_features_kept for r in plateau)),
        peak_retained,
        peak.target_n_features_kept,
    )


def _expand_neighbours(
    plateau: Sequence[int],
    candidate_widths: Sequence[int],
) -> tuple[int, ...]:
    """Active widths for the next stage: plateau + immediate
    ``candidate_widths`` neighbours of each plateau member.

    Returns sorted unique tuple. Neighbours are pulled ONLY from the
    user-supplied ``candidate_widths`` — the wrapper does not invent
    widths.
    """
    sorted_cands = sorted(set(int(w) for w in candidate_widths))
    cand_index = {w: i for i, w in enumerate(sorted_cands)}
    actives: set[int] = set(int(w) for w in plateau)
    for w in plateau:
        if w not in cand_index:
            continue
        idx = cand_index[w]
        if idx > 0:
            actives.add(sorted_cands[idx - 1])
        if idx < len(sorted_cands) - 1:
            actives.add(sorted_cands[idx + 1])
    return tuple(sorted(actives))


def _detect_convergence(
    trajectory: Sequence[ConvergenceTrajectoryEntry],
    *,
    convergence_n_stages: int,
    retained_mauc_tolerance: float,
) -> tuple[bool, int]:
    """Has the smallest-plateau-member width been stable for
    ``convergence_n_stages`` consecutive stages, with retained_mauc
    variance within tolerance?

    Returns ``(converged, stages_converged)`` where
    ``stages_converged`` counts the trailing run of stable stages
    (clamped to len(trajectory)).
    """
    if len(trajectory) < convergence_n_stages:
        # Can't have converged across N stages without N stages on
        # record. ``stages_converged`` is the run of consecutive
        # non-shifts at the trailing edge.
        return (False, _trailing_stable(trajectory, retained_mauc_tolerance))

    # Convergence requires the last ``convergence_n_stages`` transitions
    # (entries' shifted_from_prev_stage flags) ALL to be False, AND
    # retained_mauc variance across those entries to stay within
    # tolerance. trajectory[0] has shifted_from_prev_stage=False by
    # convention (no previous stage); convergence_n_stages=1 therefore
    # ALWAYS fires after 1+ stage, which is the documented degenerate
    # "single-shot via progressive surface" mode.
    tail = trajectory[-convergence_n_stages:]
    for entry in tail:
        if entry.shifted_from_prev_stage:
            return (False, _trailing_stable(trajectory, retained_mauc_tolerance))
    # retained_mauc variance check across the tail.
    if convergence_n_stages > 1:
        values = [e.argmin_retained_mauc for e in tail]
        if max(values) - min(values) > retained_mauc_tolerance:
            return (False, _trailing_stable(trajectory, retained_mauc_tolerance))
    return (True, _trailing_stable(trajectory, retained_mauc_tolerance))


def _trailing_stable(
    trajectory: Sequence[ConvergenceTrajectoryEntry],
    retained_mauc_tolerance: float,
) -> int:
    """Count consecutive trailing non-shifted stages with retained_mauc
    variance within tolerance."""
    if not trajectory:
        return 0
    stable = 1
    for i in range(len(trajectory) - 1, 0, -1):
        if trajectory[i].shifted_from_prev_stage:
            break
        if abs(trajectory[i].argmin_retained_mauc
               - trajectory[i - 1].argmin_retained_mauc) > retained_mauc_tolerance:
            break
        stable += 1
    return stable


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------


def sweep_pareto_capability_progressive(
    sae_checkpoint: "str | Path",
    host_model_id: str,
    dataset: Any,  # CapabilityDataset
    *,
    candidate_widths: Sequence[int],
    n_proteins_schedule: Sequence[int],
    output_dir: "str | Path",
    encodings: list[str] | None = None,
    scale_boosts: list["float | str"] | None = None,
    retained_mauc_tolerance: float = 0.005,
    plateau_tolerance: float = 0.01,
    min_plateau_widths: int = 3,
    convergence_n_stages: int = 2,
    cache_host: bool = True,
    max_seq_len: int = 512,
    device: str = "cpu",
) -> ProgressiveHistory:
    """Multi-stage capability sweep returning a stable recommendation.

    See module docstring for the contract. Returns a
    :class:`ProgressiveHistory` carrying per-stage results +
    the final :class:`ProgressiveRecommendation`.

    **Parameter guidance (production defaults):**

    - ``convergence_n_stages`` — **production users SHOULD leave
      this at the default of 2** (or set 3 for more conservative
      claims). The value controls how many consecutive non-shifting
      stages the wrapper requires before declaring convergence;
      higher values give stronger stability guarantees at the cost
      of one or two extra stages of compute.

      ``convergence_n_stages=1`` is supported as an *explicit
      opt-out* of the strict default, NOT as a recommended
      production value: it accepts a recommendation that's
      stable-vs-the-previous-single-stage but provides no
      multi-stage robustness attestation. Use only when the user
      has already separately verified the substrate's optimum is
      data-scale-stable, or for cheap exploratory probes before
      committing to a full sweep.

    - ``plateau_tolerance`` — 0.01 (default) defines a 1 % AUC
      band around the peak as "tied for first". Tighten to 0.005
      for substrates where retained_mauc separates cleanly;
      loosen to 0.02 for flat plateaus where many widths sit close
      together.

    - ``retained_mauc_tolerance`` — 0.005 (default) caps the
      max-pairwise-difference in retained_mauc across the
      trailing ``convergence_n_stages`` stages. Tighten only when
      you need extremely precise stability claims; the noise
      floor from protein-sample variation typically sits around
      this value.

    **Two opt-in modes that are NOT --accept-unconverged:**

    - ``convergence_n_stages=1``: looser data-scale check. Still
      asks "did the last stage shift from the previous?", just
      doesn't require K-in-a-row stability.
    - Single-element ``n_proteins_schedule=[N]``: degenerate to a
      single-shot ``sweep_pareto_capability`` at N proteins. Emits
      a progressive frontier with one stage; ``converged=True`` by
      definition (no prior stage exists to shift from).

    Both are *informed opt-outs* for users who don't want the strict
    default but also don't want to blanket-accept un-converged
    output via ``--accept-unconverged``.

    **Validation (raises ``ValueError`` with actionable messages):**

    - ``n_proteins_schedule`` SHALL be monotone non-decreasing
      (cumulative subsampling requires this).
    - ``n_proteins_schedule[-1]`` SHALL NOT exceed
      ``len(dataset.sequences)``.
    - ``candidate_widths`` SHALL be non-empty.
    """
    from saeforge.datasets.capability import CapabilityDataset
    from saeforge.sweep_capability import sweep_pareto_capability

    # ---- Validation ----
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule = [int(n) for n in n_proteins_schedule]
    if not schedule:
        raise ValueError(
            "sweep_pareto_capability_progressive: n_proteins_schedule must "
            "be non-empty. Pass a list like [10, 50, 200, 1000] (the "
            "fidelity ladder) or [200] for single-shot mode."
        )
    bad_transitions = [
        (i, schedule[i], schedule[i + 1])
        for i in range(len(schedule) - 1)
        if schedule[i] > schedule[i + 1]
    ]
    if bad_transitions:
        i, hi, lo = bad_transitions[0]
        raise ValueError(
            f"sweep_pareto_capability_progressive: n_proteins_schedule "
            f"must be monotone non-decreasing because each stage's "
            f"subsample is the previous stage's superset. Got "
            f"{schedule!r}; the transition at index {i} drops "
            f"{hi} -> {lo}. Sort the schedule ascending or check for "
            f"a typo."
        )
    if schedule[-1] > len(dataset.sequences):
        raise ValueError(
            f"sweep_pareto_capability_progressive: schedule's largest "
            f"stage ({schedule[-1]} proteins) exceeds the dataset's "
            f"available sequences ({len(dataset.sequences)}). Either "
            f"shrink the largest stage to <= {len(dataset.sequences)} "
            f"or load a larger dataset slice (e.g. raise "
            f"CapabilityDataset.from_bio_sae(n_proteins=...) at "
            f"construction time)."
        )
    if not candidate_widths:
        raise ValueError(
            "sweep_pareto_capability_progressive: candidate_widths must "
            "be non-empty. Pass a list of basis widths to consider, e.g. "
            "[16, 64, 128, 256, 512, 1024] for a typical SAE-of-1024 "
            "fixture."
        )

    if encodings is None or not encodings:
        encodings = ["raw_slice"]
    if scale_boosts is None or not scale_boosts:
        scale_boosts = [1.0]

    # ---- Per-stage loop ----
    active = tuple(sorted(set(int(w) for w in candidate_widths)))
    stage_results: list[ProgressiveStageResult] = []
    trajectory: list[ConvergenceTrajectoryEntry] = []
    aggregated_frontier_path = output_dir / "frontier.jsonl"
    if aggregated_frontier_path.exists():
        aggregated_frontier_path.unlink()

    prev_argmin: int | None = None
    converged_now = False

    for stage_idx, n_proteins in enumerate(schedule):
        stage_dir = output_dir / f"stage_{stage_idx}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        # Cumulative subsample (deterministic: take the first N).
        # The dataset is frozen; reconstruct a smaller view for this stage.
        stage_dataset = CapabilityDataset(
            sequences=list(dataset.sequences[:n_proteins]),
            labels=_slice_labels_for_stage(
                dataset, n_proteins=n_proteins,
            ),
            encoder=dataset.encoder,
            tokenizer_id=dataset.tokenizer_id,
            feed=dataset.feed,
            aggregator=dataset.aggregator,
            min_prevalence=dataset.min_prevalence,
            decode_via_basis=dataset.decode_via_basis,
            metadata={**dataset.metadata, "stage": stage_idx,
                      "n_proteins_in_stage": n_proteins},
        )
        rows = sweep_pareto_capability(
            sae_checkpoint=sae_checkpoint,
            host_model_id=host_model_id,
            dataset=stage_dataset,
            widths=list(active),
            encodings=encodings,
            scale_boosts=scale_boosts,
            output_dir=stage_dir,
            cache_host=cache_host,
            max_seq_len=max_seq_len,
            device=device,
        )
        # Stamp stage on each row and append to the aggregated frontier.
        stamped_rows = tuple(
            _row_with_stage(r, stage_idx) for r in rows
        )
        with aggregated_frontier_path.open("a") as fh:
            for r in stamped_rows:
                fh.write(json.dumps(r.to_json_dict()) + "\n")

        # Plateau + neighbours.
        plateau, peak_retained, peak_n = _identify_plateau(
            stamped_rows,
            plateau_tolerance=plateau_tolerance,
            min_plateau_widths=min_plateau_widths,
        )
        if not plateau:
            # Stage produced no successful rows; abort with an
            # informative trajectory entry.
            trajectory.append(ConvergenceTrajectoryEntry(
                stage=stage_idx,
                n_proteins=n_proteins,
                argmin_plateau_width=-1,
                argmin_retained_mauc=float("nan"),
                plateau_size=0,
                neighbours_added=0,
                shifted_from_prev_stage=False,
            ))
            stage_results.append(ProgressiveStageResult(
                stage=stage_idx, n_proteins=n_proteins,
                active_widths=active, rows=stamped_rows,
                plateau_widths=(), peak_n=-1,
                peak_retained_mauc=float("nan"),
            ))
            break

        argmin_width = min(plateau)
        argmin_row = next(r for r in stamped_rows
                          if r.target_n_features_kept == argmin_width)
        argmin_retained = float(argmin_row.retained_mauc_vs_host)
        shifted = (prev_argmin is not None and argmin_width != prev_argmin)
        next_active = _expand_neighbours(plateau, candidate_widths)
        neighbours_added = len(set(next_active) - set(plateau))

        trajectory.append(ConvergenceTrajectoryEntry(
            stage=stage_idx,
            n_proteins=n_proteins,
            argmin_plateau_width=argmin_width,
            argmin_retained_mauc=argmin_retained,
            plateau_size=len(plateau),
            neighbours_added=neighbours_added,
            shifted_from_prev_stage=shifted,
        ))
        stage_results.append(ProgressiveStageResult(
            stage=stage_idx, n_proteins=n_proteins,
            active_widths=active, rows=stamped_rows,
            plateau_widths=plateau, peak_n=peak_n,
            peak_retained_mauc=peak_retained,
        ))

        # Single-element schedule: degenerate single-shot mode.
        if len(schedule) == 1:
            converged_now = True
            break

        # Convergence check.
        converged_now, _ = _detect_convergence(
            trajectory,
            convergence_n_stages=convergence_n_stages,
            retained_mauc_tolerance=retained_mauc_tolerance,
        )
        if converged_now:
            break

        active = next_active
        prev_argmin = argmin_width

    # ---- Build recommendation ----
    if trajectory and trajectory[-1].argmin_plateau_width >= 0:
        last = trajectory[-1]
        _, stages_converged = _detect_convergence(
            trajectory,
            convergence_n_stages=convergence_n_stages,
            retained_mauc_tolerance=retained_mauc_tolerance,
        )
        rationale = _build_rationale(
            trajectory, converged_now,
            convergence_n_stages=convergence_n_stages,
            retained_mauc_tolerance=retained_mauc_tolerance,
        )
        recommendation = ProgressiveRecommendation(
            target_n_features_kept=last.argmin_plateau_width,
            retained_mauc_vs_host=last.argmin_retained_mauc,
            stages_converged=stages_converged,
            converged=converged_now,
            rationale=rationale,
            convergence_trajectory=tuple(trajectory),
        )
    else:
        recommendation = ProgressiveRecommendation(
            target_n_features_kept=-1,
            retained_mauc_vs_host=float("nan"),
            stages_converged=0,
            converged=False,
            rationale="All stages failed to produce a non-empty plateau.",
            convergence_trajectory=tuple(trajectory),
        )

    history = ProgressiveHistory(
        stages=tuple(stage_results),
        recommendation=recommendation,
    )

    # Write the summary JSON.
    (output_dir / "progressive_summary.json").write_text(
        json.dumps(history.to_json_dict(), indent=2)
    )
    return history


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _slice_labels_for_stage(dataset: Any, *, n_proteins: int):
    """Subsample the dataset's labels for the first N proteins.

    - Pooled feed: take the first N rows (each row is one protein).
    - Residue feed: take all rows whose residue belongs to one of the
      first N proteins. v1 assumes the dataset's labels were built
      from a protein-major-ordered residue_index (bio-sae's
      convention); a future revision could carry an explicit
      protein_id-per-row vector on CapabilityDataset.
    """
    import numpy as np

    if dataset.feed == "pooled":
        return np.ascontiguousarray(dataset.labels[:n_proteins])

    # Residue feed: we need to know which rows belong to the first N
    # proteins. Bio-sae's from_bio_sae populates dataset.labels by
    # taking residue_index[:,0] < n_proteins, so the labels are
    # already protein-major. Slicing rows here for the smaller
    # stage requires us to know how many residues each of the
    # first N proteins contributed.
    #
    # The dataset's sequences carry the ordering; the row count per
    # protein is recoverable from the bundle's residue_index. We
    # don't have access to residue_index here, so we read it from
    # metadata if present (bio-sae writes it as a sidecar) or fall
    # back to assuming the labels matrix is already aligned with
    # the truncated sequences (which is what bio-sae's from_bio_sae
    # produces).
    #
    # The CapabilityDataset already truncated to the requested
    # n_proteins at construction time, so dataset.labels already
    # corresponds to ALL its sequences. We just need the first N's
    # worth of rows. For that we need a residues-per-protein count.
    # bio-sae's from_bio_sae stores n_proteins in metadata; the row
    # count per protein is recoverable as labels.shape[0] /
    # len(sequences) ONLY IF every protein has the same residue
    # count (which is NOT true for real proteins).
    #
    # Practical resolution: stage subsampling in residue feed
    # requires the dataset to expose a residues_per_protein vector.
    # v1 supports this by checking dataset.metadata for
    # 'residues_per_protein' (bio-sae's from_bio_sae will populate
    # this in a follow-up); when absent, raise.
    rpp = dataset.metadata.get("residues_per_protein")
    if rpp is None:
        # Conservative fallback: if we're slicing to ALL proteins,
        # return the full labels as-is.
        if n_proteins >= len(dataset.sequences):
            return np.ascontiguousarray(dataset.labels)
        raise RuntimeError(
            "sweep_pareto_capability_progressive(feed='residue'): "
            "dataset.metadata['residues_per_protein'] is required to "
            "subsample residue-scope labels for a smaller stage. "
            "Bio-sae's from_bio_sae populates this in v0.8.2+; ensure "
            "the dataset was built with that version or supply the "
            "vector manually."
        )
    rpp = list(rpp)
    if len(rpp) < n_proteins:
        raise RuntimeError(
            f"residues_per_protein has {len(rpp)} entries, need "
            f"≥{n_proteins}"
        )
    cum = int(sum(rpp[:n_proteins]))
    return np.ascontiguousarray(dataset.labels[:cum])


def _row_with_stage(row: ParetoFrontierRow, stage: int) -> ParetoFrontierRow:
    """Return a copy of ``row`` with ``stage`` populated.

    ParetoFrontierRow is frozen; we round-trip through ``__init__``.
    """
    from dataclasses import replace

    return replace(row, stage=stage)


def _build_rationale(
    trajectory: Sequence[ConvergenceTrajectoryEntry],
    converged: bool,
    *,
    convergence_n_stages: int,
    retained_mauc_tolerance: float,
) -> str:
    """Human-readable explanation of why the recommended width was
    picked + whether convergence was reached."""
    if not trajectory:
        return "No stages ran."
    last = trajectory[-1]
    if converged:
        # Find the run of stable trailing stages.
        stable_stages = [
            e.stage for e in trajectory[-convergence_n_stages:]
        ]
        return (
            f"Smallest plateau-member n={last.argmin_plateau_width} "
            f"stable across stages {stable_stages} "
            f"(retained_mauc trajectory: "
            f"{[round(e.argmin_retained_mauc, 4) for e in trajectory[-convergence_n_stages:]]}, "
            f"max variance "
            f"{_trajectory_variance(trajectory[-convergence_n_stages:]):.4f} "
            f"within tolerance {retained_mauc_tolerance})."
        )
    else:
        # Identify which transition failed.
        failed = next(
            (i for i in range(1, len(trajectory))
             if trajectory[i].shifted_from_prev_stage),
            None,
        )
        if failed is not None:
            return (
                f"Schedule exhausted without convergence. Stage "
                f"{failed}'s argmin-plateau-member shifted from "
                f"n={trajectory[failed - 1].argmin_plateau_width} to "
                f"n={trajectory[failed].argmin_plateau_width}. Last "
                f"stage's recommendation "
                f"(n={last.argmin_plateau_width}) is the wrapper's "
                f"best guess but is NOT data-scale-robust. Consider "
                f"(a) extending the schedule, (b) loosening "
                f"plateau_tolerance, (c) "
                f"convergence_n_stages=1 for a less-strict mode, or "
                f"(d) --accept-unconverged if the recommendation is "
                f"acceptable as-is."
            )
        return (
            f"Schedule exhausted before convergence_n_stages "
            f"({convergence_n_stages}) of consecutive stable stages "
            f"could be observed. Recommendation "
            f"(n={last.argmin_plateau_width}) is the last stage's "
            f"argmin-plateau-member."
        )


def _trajectory_variance(
    trajectory: Sequence[ConvergenceTrajectoryEntry],
) -> float:
    """Max pairwise difference in argmin_retained_mauc across a
    trajectory slice. Used for the rationale string."""
    if len(trajectory) < 2:
        return 0.0
    values = [e.argmin_retained_mauc for e in trajectory]
    return max(values) - min(values)
