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
from dataclasses import dataclass, field
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
    """One stage's outcome inside a progressive sweep.

    Single-encoding sweeps populate ``plateau_widths`` (the legacy
    single-tuple shape) and leave ``per_encoding_plateau_widths`` empty.

    Multi-encoding sweeps populate ``per_encoding_plateau_widths`` (one
    entry per encoding label); ``plateau_widths`` carries the
    plateau of the winning encoding (per the recommendation
    tiebreaker) for back-compat consumers that read the legacy
    field.
    """

    stage: int
    n_proteins: int
    active_widths: tuple[int, ...]
    rows: tuple[ParetoFrontierRow, ...]
    plateau_widths: tuple[int, ...]
    peak_n: int
    peak_retained_mauc: float
    per_encoding_plateau_widths: dict[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressiveRecommendation:
    """The final recommendation from a progressive sweep.

    Attributes
    ----------
    target_n_features_kept:
        The smallest stable-plateau width. Pareto-optimal on
        (capability, parameter-cost). For multi-encoding sweeps,
        this belongs to the winning encoding per the tiebreaker
        (see ``per_encoding_recommendations``).
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
        For multi-encoding sweeps, names the winning encoding and
        the tiebreaker that selected it.
    convergence_trajectory:
        Per-stage record. On disk in ``progressive_summary.json`` —
        external benchmarking can count un-converged ratios without
        in-library telemetry.
    per_encoding_recommendations:
        For multi-encoding sweeps, a dict mapping encoding label to
        the per-encoding ``ProgressiveRecommendation``. None for
        single-encoding sweeps (back-compat preserved).

        The top-level recommendation belongs to the winning
        encoding per the tiebreaker chain:
          1. Among encodings whose recommendation converged.
          2. Pick smallest stable n at retained_mauc >=
             cross-encoding median of converged retained_mauc.
          3. Ties broken by lowest argmin-retained-mauc variance.
          4. Final tiebreak by encoding-list order.

        If NO encoding converged, top-level falls back to the
        encoding with lowest variance + names this in rationale.
    """

    target_n_features_kept: int
    retained_mauc_vs_host: float
    stages_converged: int
    converged: bool
    rationale: str
    convergence_trajectory: tuple[ConvergenceTrajectoryEntry, ...]
    per_encoding_recommendations: dict[str, "ProgressiveRecommendation"] | None = None
    winning_encoding: str | None = None


@dataclass(frozen=True)
class ProgressiveHistory:
    """Bundle of per-stage results + the final recommendation."""

    stages: tuple[ProgressiveStageResult, ...]
    recommendation: ProgressiveRecommendation

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary written to
        ``progressive_summary.json``."""
        def _rec_to_dict(rec: ProgressiveRecommendation) -> dict[str, Any]:
            d = {
                "target_n_features_kept": rec.target_n_features_kept,
                "retained_mauc_vs_host": rec.retained_mauc_vs_host,
                "stages_converged": rec.stages_converged,
                "converged": rec.converged,
                "rationale": rec.rationale,
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
                    for e in rec.convergence_trajectory
                ],
            }
            if rec.winning_encoding is not None:
                d["winning_encoding"] = rec.winning_encoding
            return d

        top_level_rec = _rec_to_dict(self.recommendation)
        if self.recommendation.per_encoding_recommendations is not None:
            top_level_rec["per_encoding_recommendations"] = {
                label: _rec_to_dict(per_rec)
                for label, per_rec in
                self.recommendation.per_encoding_recommendations.items()
            }

        stage_dicts: list[dict[str, Any]] = []
        for s in self.stages:
            d = {
                "stage": s.stage,
                "n_proteins": s.n_proteins,
                "active_widths": list(s.active_widths),
                "plateau_widths": list(s.plateau_widths),
                "peak_n": s.peak_n,
                "peak_retained_mauc": s.peak_retained_mauc,
                "n_rows": len(s.rows),
            }
            if s.per_encoding_plateau_widths:
                d["per_encoding_plateau_widths"] = {
                    label: list(widths)
                    for label, widths in s.per_encoding_plateau_widths.items()
                }
            stage_dicts.append(d)

        return {
            "stages": stage_dicts,
            "recommendation": top_level_rec,
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
    sae_checkpoint: "str | Path | None" = None,
    host_model_id: str | None = None,
    dataset: Any = None,  # CapabilityDataset
    *,
    candidate_widths: Sequence[int],
    n_proteins_schedule: Sequence[int],
    output_dir: "str | Path",
    encodings: "list[tuple[str, str | Path]] | list[str] | None" = None,
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

    if scale_boosts is None or not scale_boosts:
        scale_boosts = [1.0]

    # ---- Encoding-list normalization ----
    # Two input shapes:
    #   (a) legacy single-encoding: encodings=None + sae_checkpoint=PATH,
    #       or encodings=['label_a', 'label_b'] (informational labels
    #       — v0.8.x back-compat).
    #   (b) multi-encoding: encodings=[(label, path), ...] (new shape).
    #
    # Internally, we always model state per-encoding-label. Single-
    # encoding sweeps have one entry in the per-encoding state dict;
    # multi-encoding sweeps have one entry per encoding label.
    is_multi_encoding = (
        encodings is not None
        and len(encodings) > 0
        and not isinstance(encodings[0], str)
    )
    if is_multi_encoding:
        # Multi-encoding mode — encodings is list[tuple[str, path]].
        if sae_checkpoint is not None:
            raise ValueError(
                "sweep_pareto_capability_progressive: pass either "
                "`encodings=[(label, path), ...]` (multi-encoding) OR "
                "`sae_checkpoint=PATH` (single-encoding), not both."
            )
        encoding_labels = [str(label) for label, _ in encodings]
        # Validate uniqueness — same as sweep_pareto_capability.
        if len(set(encoding_labels)) != len(encoding_labels):
            raise ValueError(
                f"sweep_pareto_capability_progressive: duplicate encoding "
                f"label in encodings. Labels SHALL be unique; got "
                f"{encoding_labels!r}."
            )
        encodings_arg_for_sweep = encodings  # tuple list passes through
    else:
        # Single-encoding mode (legacy informational labels OR
        # sae_checkpoint).
        if encodings is None or not encodings:
            encoding_labels = ["raw_slice"]
            encodings_arg_for_sweep = None
        else:
            # Legacy informational labels.
            encoding_labels = [str(e) for e in encodings]
            encodings_arg_for_sweep = list(encoding_labels)

    # ---- Per-stage loop ----
    # Per-encoding state tracked across stages.
    initial_active = tuple(sorted(set(int(w) for w in candidate_widths)))
    per_encoding_active: dict[str, tuple[int, ...]] = {
        label: initial_active for label in encoding_labels
    }
    per_encoding_trajectory: dict[str, list[ConvergenceTrajectoryEntry]] = {
        label: [] for label in encoding_labels
    }
    per_encoding_prev_argmin: dict[str, int | None] = {
        label: None for label in encoding_labels
    }
    per_encoding_converged: dict[str, bool] = {
        label: False for label in encoding_labels
    }
    stage_results: list[ProgressiveStageResult] = []
    aggregated_frontier_path = output_dir / "frontier.jsonl"
    if aggregated_frontier_path.exists():
        aggregated_frontier_path.unlink()

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
        # Union of all encodings' active widths for this stage. Each
        # encoding's own plateau-based active set is a subset of this
        # union; we sweep the union and then partition rows by encoding.
        stage_active_union = sorted(
            set().union(*per_encoding_active.values())
        )
        rows = sweep_pareto_capability(
            sae_checkpoint=sae_checkpoint if not is_multi_encoding else None,
            host_model_id=host_model_id,
            dataset=stage_dataset,
            widths=list(stage_active_union),
            encodings=encodings_arg_for_sweep,
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

        # Per-encoding plateau identification + trajectory update.
        per_encoding_plateau_widths_this_stage: dict[str, tuple[int, ...]] = {}
        any_encoding_succeeded = False
        for label in encoding_labels:
            # Filter to this encoding's active widths AND this
            # encoding's rows.
            enc_active = per_encoding_active[label]
            enc_rows = tuple(
                r for r in stamped_rows
                if r.encoding_label == label
                and r.target_n_features_kept in enc_active
            )
            plateau, peak_retained, peak_n = _identify_plateau(
                enc_rows,
                plateau_tolerance=plateau_tolerance,
                min_plateau_widths=min_plateau_widths,
            )
            if not plateau:
                # This encoding produced no successful rows at this
                # stage; record a degenerate trajectory entry but
                # don't abort other encodings.
                per_encoding_trajectory[label].append(
                    ConvergenceTrajectoryEntry(
                        stage=stage_idx, n_proteins=n_proteins,
                        argmin_plateau_width=-1,
                        argmin_retained_mauc=float("nan"),
                        plateau_size=0, neighbours_added=0,
                        shifted_from_prev_stage=False,
                    )
                )
                per_encoding_plateau_widths_this_stage[label] = ()
                continue
            any_encoding_succeeded = True
            argmin_width = min(plateau)
            argmin_row = next(
                r for r in enc_rows
                if r.target_n_features_kept == argmin_width
            )
            argmin_retained = float(argmin_row.retained_mauc_vs_host)
            prev = per_encoding_prev_argmin[label]
            shifted = (prev is not None and argmin_width != prev)
            next_active = _expand_neighbours(plateau, candidate_widths)
            neighbours_added = len(set(next_active) - set(plateau))
            per_encoding_trajectory[label].append(
                ConvergenceTrajectoryEntry(
                    stage=stage_idx, n_proteins=n_proteins,
                    argmin_plateau_width=argmin_width,
                    argmin_retained_mauc=argmin_retained,
                    plateau_size=len(plateau),
                    neighbours_added=neighbours_added,
                    shifted_from_prev_stage=shifted,
                )
            )
            per_encoding_plateau_widths_this_stage[label] = plateau
            # Advance per-encoding state for next stage.
            per_encoding_active[label] = next_active
            per_encoding_prev_argmin[label] = argmin_width

        # Top-level (winning-encoding) plateau for ProgressiveStageResult.
        # Pick the encoding with the largest plateau at this stage as the
        # representative for the legacy plateau_widths field — back-compat
        # for single-encoding consumers. (Single-encoding sweeps have one
        # entry; this picks that entry.)
        if per_encoding_plateau_widths_this_stage:
            top_label = max(
                per_encoding_plateau_widths_this_stage.keys(),
                key=lambda L: len(per_encoding_plateau_widths_this_stage[L]),
            )
            top_plateau = per_encoding_plateau_widths_this_stage[top_label]
            top_traj = per_encoding_trajectory[top_label][-1]
            top_peak_n = top_traj.argmin_plateau_width
            top_peak_retained = top_traj.argmin_retained_mauc
        else:
            top_plateau = ()
            top_peak_n = -1
            top_peak_retained = float("nan")

        stage_results.append(ProgressiveStageResult(
            stage=stage_idx, n_proteins=n_proteins,
            active_widths=tuple(stage_active_union),
            rows=stamped_rows,
            plateau_widths=top_plateau,
            peak_n=top_peak_n,
            peak_retained_mauc=top_peak_retained,
            per_encoding_plateau_widths=per_encoding_plateau_widths_this_stage,
        ))

        if not any_encoding_succeeded:
            break

        # Single-element schedule: degenerate single-shot mode for
        # all encodings.
        if len(schedule) == 1:
            for label in encoding_labels:
                per_encoding_converged[label] = True
            break

        # Per-encoding convergence detection. Loop exits when ALL
        # encodings have converged.
        for label in encoding_labels:
            if per_encoding_converged[label]:
                continue
            converged, _ = _detect_convergence(
                per_encoding_trajectory[label],
                convergence_n_stages=convergence_n_stages,
                retained_mauc_tolerance=retained_mauc_tolerance,
            )
            per_encoding_converged[label] = converged
        if all(per_encoding_converged[label] for label in encoding_labels):
            break

    # ---- Build per-encoding recommendations ----
    per_encoding_recs: dict[str, ProgressiveRecommendation] = {}
    for label in encoding_labels:
        trajectory = per_encoding_trajectory[label]
        converged = per_encoding_converged[label]
        if trajectory and trajectory[-1].argmin_plateau_width >= 0:
            last = trajectory[-1]
            _, stages_converged = _detect_convergence(
                trajectory,
                convergence_n_stages=convergence_n_stages,
                retained_mauc_tolerance=retained_mauc_tolerance,
            )
            rationale = _build_rationale(
                trajectory, converged,
                convergence_n_stages=convergence_n_stages,
                retained_mauc_tolerance=retained_mauc_tolerance,
            )
            per_encoding_recs[label] = ProgressiveRecommendation(
                target_n_features_kept=last.argmin_plateau_width,
                retained_mauc_vs_host=last.argmin_retained_mauc,
                stages_converged=stages_converged,
                converged=converged,
                rationale=rationale,
                convergence_trajectory=tuple(trajectory),
            )
        else:
            per_encoding_recs[label] = ProgressiveRecommendation(
                target_n_features_kept=-1,
                retained_mauc_vs_host=float("nan"),
                stages_converged=0,
                converged=False,
                rationale=(
                    f"Encoding {label!r}: all stages failed to produce a "
                    f"non-empty plateau."
                ),
                convergence_trajectory=tuple(trajectory),
            )

    # ---- Pick winning encoding + build top-level recommendation ----
    if is_multi_encoding:
        winning_label, winner_rationale = _pick_winning_encoding(
            per_encoding_recs,
            encoding_order=encoding_labels,
        )
        winning_rec = per_encoding_recs[winning_label]
        recommendation = ProgressiveRecommendation(
            target_n_features_kept=winning_rec.target_n_features_kept,
            retained_mauc_vs_host=winning_rec.retained_mauc_vs_host,
            stages_converged=winning_rec.stages_converged,
            converged=winning_rec.converged,
            rationale=f"Winning encoding {winning_label!r}: {winner_rationale}",
            convergence_trajectory=winning_rec.convergence_trajectory,
            per_encoding_recommendations=per_encoding_recs,
            winning_encoding=winning_label,
        )
    else:
        # Single-encoding sweep — just use the only recommendation,
        # leaving per_encoding_recommendations=None for back-compat.
        only_label = encoding_labels[0]
        recommendation = per_encoding_recs[only_label]

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


def _pick_winning_encoding(
    per_encoding_recs: dict[str, ProgressiveRecommendation],
    *,
    encoding_order: list[str],
) -> tuple[str, str]:
    """Pick the winning encoding from per-encoding recommendations.

    Tiebreaker chain per spec
    ``add-multi-encoding-capability-sweep/specs/pareto-sweep/spec.md``
    "ProgressiveRecommendation.per_encoding_recommendations"
    (design.md Decision 4 — see openspec for the full rationale):

    1. Filter to encodings whose recommendation converged.
    2. Among those, pick smallest ``target_n_features_kept`` at
       ``retained_mauc >= cross-encoding median`` of converged
       encodings' retained_mauc values.
    3. Tiebreak by lowest argmin-retained-mauc variance across stages.
    4. Final tiebreak by ``encoding_order`` index (CLI flag order /
       Python encodings list order — explicit user-supplied
       priority).

    If NO encoding converged, fall back to encoding with lowest
    argmin-retained-mauc variance (most data-scale-stable, even if
    non-converged); ``rationale`` names the fallback explicitly.

    Returns ``(winning_label, rationale_string)``.
    """
    converged = {
        label: rec for label, rec in per_encoding_recs.items()
        if rec.converged
    }
    if converged:
        # Pick smallest n at retained_mauc >= median(converged retained_mauc).
        retained_values = [
            rec.retained_mauc_vs_host for rec in converged.values()
        ]
        # numpy median; use sorted-pick to avoid the dep here.
        sorted_retained = sorted(retained_values)
        n = len(sorted_retained)
        median = (
            sorted_retained[n // 2] if n % 2 == 1
            else (sorted_retained[n // 2 - 1] + sorted_retained[n // 2]) / 2
        )
        eligible = {
            label: rec for label, rec in converged.items()
            if rec.retained_mauc_vs_host >= median
        }
        # Tiebreak chain.
        def _variance(rec: ProgressiveRecommendation) -> float:
            return _trajectory_variance(rec.convergence_trajectory)
        def _order_idx(label: str) -> int:
            return encoding_order.index(label)
        winner = min(
            eligible.keys(),
            key=lambda L: (
                eligible[L].target_n_features_kept,
                _variance(eligible[L]),
                _order_idx(L),
            ),
        )
        winner_rec = eligible[winner]
        rationale = (
            f"smallest stable n={winner_rec.target_n_features_kept} at "
            f"retained_mauc={winner_rec.retained_mauc_vs_host:.4f} "
            f">= cross-encoding median ({median:.4f}); converged with "
            f"trajectory variance {_variance(winner_rec):.4f}. "
            f"Other converged encodings: "
            f"{[k for k in converged if k != winner]!r}."
        )
        return winner, rationale

    # No encoding converged. Fall back to lowest-variance.
    def _variance(rec: ProgressiveRecommendation) -> float:
        return _trajectory_variance(rec.convergence_trajectory)
    def _order_idx(label: str) -> int:
        return encoding_order.index(label)
    winner = min(
        per_encoding_recs.keys(),
        key=lambda L: (
            _variance(per_encoding_recs[L]),
            _order_idx(L),
        ),
    )
    winner_rec = per_encoding_recs[winner]
    rationale = (
        f"NO encoding converged; fell back to lowest-variance encoding "
        f"(trajectory variance {_variance(winner_rec):.4f}). Top-level "
        f"recommendation is NOT data-scale-robust. Consider "
        f"--accept-unconverged, a longer schedule, or "
        f"convergence_n_stages=1 (see add-progressive-capability-sweep "
        f"design.md Decision 6)."
    )
    return winner, rationale


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
