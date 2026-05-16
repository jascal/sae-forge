"""Tests for the Pareto sweep driver.

Covers: ParetoFrontierRow validation + round-trip, manifest parsing,
checkpoint enumeration, multi-K sweeps, resumability, multi-encoding,
per-row failure isolation, truncated-JSONL recovery, frontier-only mode
(with and without manifest), and CLI smoke.

Most tests use a stub pipeline with monkey-patched ``_basis_swap`` so they
don't require torch or full forge runs. The byte-equivalence scenario from
the spec is exercised at integration level (gated behind torch availability);
unit tests verify the driver's orchestration contract.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from saeforge.sweep import (
    ParetoFrontierRow,
    _enumerate_checkpoints,
    _load_completed_rows,
    _parse_pareto_manifest,
    sweep_pareto,
)


# ---------------------------------------------------------------------------
# Stub pipeline — replaces ForgePipeline for orchestration unit tests
# ---------------------------------------------------------------------------


@dataclass
class _StubResult:
    output_dir: Path
    faithfulness_kl: float | None = 0.5
    extras: dict = field(default_factory=lambda: {"perplexity": 1.5, "final_loss": 2.7})


@dataclass
class _StubPipeline:
    """A minimal stand-in for ForgePipeline.

    The sweep driver's ``_basis_swap`` is monkey-patched to a no-op for tests
    that use this stub, so we never touch the real ``FeatureBasis`` /
    ``SubspaceProjector`` factories.
    """

    raise_on_calls: tuple[int, ...] = ()  # 1-indexed call numbers that raise
    basis: object = None
    projector: object = None
    _call_count: int = 0
    _calls: list[tuple[Path, Path]] = field(default_factory=list)

    def run(self, output_dir, **kwargs):
        self._call_count += 1
        # In the real driver, the SAE is already swapped into self.basis. We
        # don't have access to that here, so we record output_dir and rely on
        # the caller's `_basis_swap` monkey-patch for assertions.
        self._calls.append((Path(output_dir),))
        if self._call_count in self.raise_on_calls:
            raise RuntimeError(f"stub forge failure on call {self._call_count}")
        result_dir = Path(output_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        return _StubResult(output_dir=result_dir)


@pytest.fixture
def stub_basis_swap(monkeypatch):
    """No-op the ``_basis_swap`` context manager for orchestration tests.

    The real ``_basis_swap`` rebuilds basis + projector from a checkpoint,
    which requires ``FeatureBasis.from_polygram_checkpoint``. For tests that
    don't care about the swap itself (most), this fixture replaces it with a
    no-op so we can use the ``_StubPipeline`` without real SAE files.
    """
    @contextlib.contextmanager
    def _noop(pipeline, sae_checkpoint):
        yield

    monkeypatch.setattr("saeforge.sweep._basis_swap", _noop)


# ---------------------------------------------------------------------------
# Fixtures: per-K SAE directories + manifests
# ---------------------------------------------------------------------------


def _make_pareto_dir(
    root: Path,
    sae_template: Path,
    *,
    targets: list[int],
    actuals: list[int] | None = None,
    reached: list[bool] | None = None,
    write_manifest: bool = True,
    layout: str = "pareto_subdir",
) -> Path:
    """Materialise a fake polygram-pareto output directory.

    ``layout="pareto_subdir"`` mimics ``polygram compress --pareto-materialize``:
    ``<root>/pareto.json`` + ``<root>/pareto/k_<K>.safetensors``.
    ``layout="flat"`` puts ``k_<K>.safetensors`` at the root for the
    single-directory caller variant.
    """
    root.mkdir(parents=True, exist_ok=True)
    actuals = actuals if actuals is not None else targets
    reached = reached if reached is not None else [True] * len(targets)

    per_k_dir = root / "pareto" if layout == "pareto_subdir" else root
    per_k_dir.mkdir(parents=True, exist_ok=True)
    for k in targets:
        shutil.copy(sae_template, per_k_dir / f"k_{k}.safetensors")

    if write_manifest:
        # Polygram's actual ParetoReport JSON schema: each outcome carries
        # a flat `clusters` list (one entry per cluster representative),
        # `feature_ids`, `target_k`, `reached_target`. `n_features_kept`
        # is `len(clusters)`, not a stored field. See sweep.py
        # `_parse_pareto_manifest` for the parser this matches.
        manifest = {
            "schema_version": 1,
            "sae_checkpoint": str(sae_template),
            "sae_checkpoint_sha256": "0" * 64,
            "score_field": "polygram_overlap",
            "targets": targets,
            "outcomes": [
                {
                    "target_k": k,
                    "reached_target": r,
                    "clusters": [
                        {
                            "cluster_id": cid,
                            "members": [cid],
                            "representative": cid,
                            "zeroed": [],
                            "cluster_norm_mean": None,
                            "cluster_norm_std": None,
                            "merged_norm": None,
                        }
                        for cid in range(a)
                    ],
                    "feature_ids": list(range(a)),
                }
                for k, a, r in zip(targets, actuals, reached)
            ],
        }
        (root / "pareto.json").write_text(json.dumps(manifest))

    return root


# ---------------------------------------------------------------------------
# ParetoFrontierRow
# ---------------------------------------------------------------------------


class TestParetoFrontierRow:
    def test_importable_from_saeforge(self):
        from saeforge import ParetoFrontierRow as Top

        assert Top is ParetoFrontierRow

    def test_rejects_zero_target(self):
        with pytest.raises(ValueError, match="target_n_features_kept"):
            ParetoFrontierRow(
                encoding_label="x",
                target_n_features_kept=0,
                n_features_kept_actual=0,
                pareto_reached_target=False,
                faithfulness_kl=None,
                perplexity=None,
                final_fine_tune_loss=None,
                sae_checkpoint="x",
                forged_model_path=None,
                elapsed_seconds=0.0,
                error_message=None,
            )

    def test_rejects_negative_elapsed(self):
        with pytest.raises(ValueError, match="elapsed_seconds"):
            ParetoFrontierRow(
                encoding_label="x",
                target_n_features_kept=1,
                n_features_kept_actual=0,
                pareto_reached_target=False,
                faithfulness_kl=None,
                perplexity=None,
                final_fine_tune_loss=None,
                sae_checkpoint="x",
                forged_model_path=None,
                elapsed_seconds=-1.0,
                error_message=None,
            )

    def test_json_round_trip(self):
        row = ParetoFrontierRow(
            encoding_label="rung4",
            target_n_features_kept=200,
            n_features_kept_actual=180,
            pareto_reached_target=True,
            faithfulness_kl=0.42,
            perplexity=1.5,
            final_fine_tune_loss=2.7,
            sae_checkpoint="/tmp/k_200.safetensors",
            forged_model_path="/tmp/out",
            elapsed_seconds=12.5,
            error_message=None,
        )
        rt = ParetoFrontierRow.from_json_dict(json.loads(json.dumps(row.to_json_dict())))
        assert rt == row

    def test_non_finite_floats_become_null(self):
        row = ParetoFrontierRow(
            encoding_label="x",
            target_n_features_kept=1,
            n_features_kept_actual=1,
            pareto_reached_target=True,
            faithfulness_kl=float("inf"),
            perplexity=float("nan"),
            final_fine_tune_loss=float("-inf"),
            sae_checkpoint="x",
            forged_model_path="x",
            elapsed_seconds=0.0,
            error_message=None,
        )
        d = row.to_json_dict()
        assert d["faithfulness_kl"] is None
        assert d["perplexity"] is None
        assert d["final_fine_tune_loss"] is None


# ---------------------------------------------------------------------------
# Manifest + checkpoint enumeration
# ---------------------------------------------------------------------------


class TestEnumerateCheckpoints:
    def test_pareto_subdir_layout(self, tmp_path, synthetic_compressed_sae):
        d = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2, 5, 8],
        )
        result = _enumerate_checkpoints(d)
        ks = [k for k, _ in result]
        assert ks == [2, 5, 8]  # ascending

    def test_flat_layout(self, tmp_path, synthetic_compressed_sae):
        d = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2, 5, 8],
            write_manifest=False,
            layout="flat",
        )
        result = _enumerate_checkpoints(d)
        ks = [k for k, _ in result]
        assert ks == [2, 5, 8]

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _enumerate_checkpoints(tmp_path / "nonexistent")

    def test_directory_with_no_kN_files_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="no k_<K>"):
            _enumerate_checkpoints(tmp_path / "empty")


class TestParetoManifest:
    def test_parses_outcomes(self, tmp_path):
        manifest = {
            "schema_version": 1,
            "sae_checkpoint": "x",
            "sae_checkpoint_sha256": "0" * 64,
            "score_field": "polygram_overlap",
            "targets": [2, 5],
            "outcomes": [
                {
                    "target_k": 2,
                    "reached_target": True,
                    "clusters": [{"cluster_id": 0, "members": [0], "representative": 0, "zeroed": []},
                                 {"cluster_id": 1, "members": [1], "representative": 1, "zeroed": []}],
                    "feature_ids": [0, 1],
                },
                {
                    "target_k": 5,
                    "reached_target": False,
                    "clusters": [{"cluster_id": i, "members": [i], "representative": i, "zeroed": []} for i in range(4)],
                    "feature_ids": [0, 1, 2, 3],
                },
            ],
        }
        p = tmp_path / "pareto.json"
        p.write_text(json.dumps(manifest))
        result = _parse_pareto_manifest(p)
        assert set(result.keys()) == {2, 5}
        assert result[2].n_features_kept == 2
        assert result[2].reached_target is True
        assert result[5].reached_target is False


# ---------------------------------------------------------------------------
# Resumability scan
# ---------------------------------------------------------------------------


class TestLoadCompletedRows:
    def test_missing_file_returns_empty(self, tmp_path):
        assert _load_completed_rows(tmp_path / "missing.jsonl") == set()

    def test_reads_labelled_pairs(self, tmp_path):
        p = tmp_path / "frontier.jsonl"
        p.write_text(
            json.dumps({"encoding_label": "mps", "target_n_features_kept": 200, "error_message": None}) + "\n"
            + json.dumps({"encoding_label": "mps", "target_n_features_kept": 500, "error_message": None}) + "\n"
        )
        assert _load_completed_rows(p) == {("mps", 200), ("mps", 500)}

    def test_skips_error_rows(self, tmp_path):
        """Failure rows are retryable — not marked as completed."""
        p = tmp_path / "frontier.jsonl"
        p.write_text(
            json.dumps({"encoding_label": "mps", "target_n_features_kept": 200, "error_message": None}) + "\n"
            + json.dumps({"encoding_label": "mps", "target_n_features_kept": 500, "error_message": "boom"}) + "\n"
        )
        assert _load_completed_rows(p) == {("mps", 200)}

    def test_truncated_last_line_is_dropped(self, tmp_path):
        p = tmp_path / "frontier.jsonl"
        p.write_text(
            json.dumps({"encoding_label": "mps", "target_n_features_kept": 200, "error_message": None}) + "\n"
            + '{"encoding_label": "mps", "target_n_'  # truncated mid-write
        )
        assert _load_completed_rows(p) == {("mps", 200)}
        # File should be rewritten without the bad line.
        remaining = p.read_text().strip().splitlines()
        assert len(remaining) == 1
        json.loads(remaining[0])  # parses cleanly


# ---------------------------------------------------------------------------
# Driver: multi-K, multi-encoding, failure isolation, resumability
# ---------------------------------------------------------------------------


class TestSweepMultiK:
    def test_emits_one_row_per_k(
        self, tmp_path, synthetic_compressed_sae, stub_basis_swap
    ):
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2, 5, 8],
        )
        pipeline = _StubPipeline()
        out = tmp_path / "out"
        frontier = sweep_pareto(
            pipeline,
            encodings=[("rung4", encoding_dir)],
            output_dir=out,
        )
        assert frontier == out / "frontier.jsonl"
        rows = [json.loads(line) for line in frontier.read_text().splitlines()]
        assert len(rows) == 3
        assert [r["target_n_features_kept"] for r in rows] == [2, 5, 8]
        assert all(r["encoding_label"] == "rung4" for r in rows)
        assert all(r["error_message"] is None for r in rows)
        assert all(r["faithfulness_kl"] == 0.5 for r in rows)
        # Each row's actual count comes from the manifest.
        assert [r["n_features_kept_actual"] for r in rows] == [2, 5, 8]
        # Manifest reached_target propagates.
        assert all(r["pareto_reached_target"] is True for r in rows)
        # pipeline.run was invoked once per row.
        assert pipeline._call_count == 3


class TestSweepResumability:
    def test_skips_completed_rows(
        self, tmp_path, synthetic_compressed_sae, stub_basis_swap
    ):
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2, 5, 8],
        )
        out = tmp_path / "out"
        out.mkdir()
        # Pre-populate frontier.jsonl with two completed rows.
        (out / "frontier.jsonl").write_text(
            json.dumps({
                "encoding_label": "rung4",
                "target_n_features_kept": 2,
                "n_features_kept_actual": 2,
                "pareto_reached_target": True,
                "faithfulness_kl": 0.1,
                "perplexity": 1.0,
                "final_fine_tune_loss": 0.2,
                "sae_checkpoint": "x",
                "forged_model_path": "x",
                "elapsed_seconds": 1.0,
                "error_message": None,
            }) + "\n"
            + json.dumps({
                "encoding_label": "rung4",
                "target_n_features_kept": 5,
                "n_features_kept_actual": 5,
                "pareto_reached_target": True,
                "faithfulness_kl": 0.2,
                "perplexity": 1.1,
                "final_fine_tune_loss": 0.3,
                "sae_checkpoint": "x",
                "forged_model_path": "x",
                "elapsed_seconds": 1.0,
                "error_message": None,
            }) + "\n"
        )
        pipeline = _StubPipeline()
        sweep_pareto(
            pipeline,
            encodings=[("rung4", encoding_dir)],
            output_dir=out,
        )
        # Only K=8 should be forged.
        assert pipeline._call_count == 1
        rows = [json.loads(line) for line in (out / "frontier.jsonl").read_text().splitlines()]
        assert [r["target_n_features_kept"] for r in rows] == [2, 5, 8]


class TestSweepMultiEncoding:
    def test_two_encodings_two_k_each(
        self, tmp_path, synthetic_compressed_sae, stub_basis_swap
    ):
        mps_dir = _make_pareto_dir(
            tmp_path / "mps", synthetic_compressed_sae["checkpoint"], targets=[3, 6]
        )
        rung4_dir = _make_pareto_dir(
            tmp_path / "rung4", synthetic_compressed_sae["checkpoint"], targets=[3, 6]
        )
        pipeline = _StubPipeline()
        out = tmp_path / "out"
        sweep_pareto(
            pipeline,
            encodings=[("mps", mps_dir), ("rung4", rung4_dir)],
            output_dir=out,
        )
        rows = [json.loads(line) for line in (out / "frontier.jsonl").read_text().splitlines()]
        assert len(rows) == 4
        labels = [r["encoding_label"] for r in rows]
        assert labels.count("mps") == 2
        assert labels.count("rung4") == 2


class TestSweepFailures:
    def test_one_row_failure_does_not_abort(
        self, tmp_path, synthetic_compressed_sae, stub_basis_swap
    ):
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2, 5, 8],
        )
        pipeline = _StubPipeline(raise_on_calls=(2,))  # second row raises
        out = tmp_path / "out"
        with pytest.raises(RuntimeError, match="1 row"):
            sweep_pareto(
                pipeline,
                encodings=[("rung4", encoding_dir)],
                output_dir=out,
            )
        rows = [json.loads(line) for line in (out / "frontier.jsonl").read_text().splitlines()]
        assert len(rows) == 3
        assert rows[0]["error_message"] is None
        assert rows[1]["error_message"] is not None
        assert "stub forge failure" in rows[1]["error_message"]
        assert rows[1]["faithfulness_kl"] is None
        assert rows[2]["error_message"] is None
        # Subsequent rows still produced finite metrics.
        assert rows[2]["faithfulness_kl"] == 0.5

    def test_failure_row_is_retried_on_next_sweep(
        self, tmp_path, synthetic_compressed_sae, stub_basis_swap
    ):
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2, 5],
        )
        out = tmp_path / "out"
        # First sweep: K=5 fails.
        pipeline_1 = _StubPipeline(raise_on_calls=(2,))
        with pytest.raises(RuntimeError):
            sweep_pareto(pipeline_1, encodings=[("rung4", encoding_dir)], output_dir=out)
        # Second sweep: K=5 retries (and succeeds with a fresh pipeline).
        pipeline_2 = _StubPipeline()
        sweep_pareto(pipeline_2, encodings=[("rung4", encoding_dir)], output_dir=out)
        # K=2 was completed first time, so pipeline_2 only forges K=5.
        assert pipeline_2._call_count == 1


# ---------------------------------------------------------------------------
# Frontier-only mode
# ---------------------------------------------------------------------------


class TestFrontierOnly:
    def test_no_forge_calls(
        self, tmp_path, synthetic_compressed_sae, stub_basis_swap
    ):
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2, 5, 8],
        )
        pipeline = _StubPipeline()
        out = tmp_path / "out"
        sweep_pareto(
            pipeline,
            encodings=[("rung4", encoding_dir)],
            output_dir=out,
            frontier_only=True,
        )
        assert pipeline._call_count == 0
        rows = [json.loads(line) for line in (out / "frontier.jsonl").read_text().splitlines()]
        assert len(rows) == 3
        for r in rows:
            assert r["faithfulness_kl"] is None
            assert r["perplexity"] is None
            assert r["forged_model_path"] is None
            assert r["n_features_kept_actual"] is not None  # from manifest
            assert r["pareto_reached_target"] is True

    def test_manifest_fallback(
        self, tmp_path, synthetic_compressed_sae, stub_basis_swap
    ):
        """Without pareto.json, n_features_kept_actual falls back to SAE counting."""
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2, 5],
            write_manifest=False,
        )
        pipeline = _StubPipeline()
        out = tmp_path / "out"
        sweep_pareto(
            pipeline,
            encodings=[("rung4", encoding_dir)],
            output_dir=out,
            frontier_only=True,
        )
        rows = [json.loads(line) for line in (out / "frontier.jsonl").read_text().splitlines()]
        # All rows use the same synthetic SAE template, so n_features_kept_actual
        # is the same non-zero count and pareto_reached_target is None
        # (undeterminable without manifest).
        for r in rows:
            assert r["n_features_kept_actual"] is not None
            assert r["pareto_reached_target"] is None


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class TestCLI:
    def test_parse_encoding_specs(self):
        from saeforge.cli import _parse_encoding_specs

        result = _parse_encoding_specs(["mps:/path/to/mps", "rung4:/other/path"])
        assert result == [("mps", Path("/path/to/mps")), ("rung4", Path("/other/path"))]

    def test_parse_encoding_rejects_no_colon(self):
        from saeforge.cli import _parse_encoding_specs

        with pytest.raises(ValueError, match="no colon found"):
            _parse_encoding_specs(["bogus"])

    def test_parse_encoding_rejects_empty_label(self):
        from saeforge.cli import _parse_encoding_specs

        with pytest.raises(ValueError, match="empty label"):
            _parse_encoding_specs([":/path"])

    def test_parse_encoding_rejects_empty_path(self):
        from saeforge.cli import _parse_encoding_specs

        with pytest.raises(ValueError, match="empty path"):
            _parse_encoding_specs(["label:"])

    def test_parse_encoding_accepts_path_with_colon(self):
        """Paths with internal colons (e.g. Windows drive letters) split on the first."""
        from saeforge.cli import _parse_encoding_specs

        result = _parse_encoding_specs(["mps:C:/path/to/sae"])
        assert result == [("mps", Path("C:/path/to/sae"))]

    def test_cli_frontier_only_smoke(
        self, tmp_path, synthetic_compressed_sae, monkeypatch
    ):
        """End-to-end: argv → frontier.jsonl with --frontier-only.

        Doesn't construct a real ForgePipeline.run path because the bootstrap
        basis (the first K SAE) is loaded via FeatureBasis.from_polygram_checkpoint
        — that's fine, the fixture is a real safetensors file. --frontier-only
        skips the actual forge call.
        """
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2, 5],
        )
        out = tmp_path / "out"
        from saeforge.cli import main

        rc = main([
            "sweep-pareto",
            "--encoding", f"rung4:{encoding_dir}",
            "--host-model", "gpt2",  # unused with --frontier-only
            "--output-dir", str(out),
            "--frontier-only",
        ])
        assert rc == 0
        assert (out / "frontier.jsonl").is_file()
        rows = [json.loads(line) for line in (out / "frontier.jsonl").read_text().splitlines()]
        assert len(rows) == 2
        assert {r["target_n_features_kept"] for r in rows} == {2, 5}


# ---------------------------------------------------------------------------
# Forge-quality diagnostics
# ---------------------------------------------------------------------------


class TestForgeQualityDiagnostics:
    def test_rows_carry_diagnostics_when_d_model_resolved(
        self, tmp_path, synthetic_compressed_sae, stub_basis_swap, monkeypatch
    ):
        """When `host_d_model_override` is supplied, every row populates the
        four diagnostic fields (host_d_model, basis_rank, quality_ratio,
        quality_tier).
        """
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2, 5, 8],
        )
        pipeline = _StubPipeline()
        out = tmp_path / "out"

        from saeforge.sweep import sweep_pareto

        sweep_pareto(
            pipeline,
            encodings=[("rung4", encoding_dir)],
            output_dir=out,
            host_d_model_override=16,  # synthetic SAE has d_model=16
        )
        rows = [json.loads(line) for line in (out / "frontier.jsonl").read_text().splitlines()]
        assert len(rows) == 3
        for r in rows:
            assert r["host_d_model"] == 16
            assert r["basis_rank"] is not None
            assert r["quality_ratio"] is not None
            assert r["quality_tier"] in {"degenerate", "undersized", "good", "saturated"}

    def test_rows_diagnostic_fields_null_when_d_model_unresolvable(
        self, tmp_path, synthetic_compressed_sae, stub_basis_swap
    ):
        """When `resolve_host_d_model` returns None and no override given,
        diagnostic fields are all None across rows.
        """
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2],
        )
        pipeline = _StubPipeline()
        # The stub pipeline has no host_model_id, so resolution is short-
        # circuited to None.
        from saeforge.sweep import sweep_pareto

        sweep_pareto(
            pipeline,
            encodings=[("rung4", encoding_dir)],
            output_dir=tmp_path / "out",
        )
        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "frontier.jsonl").read_text().splitlines()
        ]
        for r in rows:
            assert r["host_d_model"] is None
            assert r["basis_rank"] is None
            assert r["quality_ratio"] is None
            assert r["quality_tier"] is None

    def test_quality_floor_refuses_before_any_forge(
        self, tmp_path, synthetic_compressed_sae, stub_basis_swap
    ):
        """`quality_floor=0.5` against a tiny synthetic SAE refuses before
        any forge call (mocked pipeline.run sees zero invocations).
        """
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2],
        )
        pipeline = _StubPipeline()
        from saeforge.sweep import sweep_pareto

        with pytest.raises(RuntimeError, match="quality_floor"):
            sweep_pareto(
                pipeline,
                encodings=[("rung4", encoding_dir)],
                output_dir=tmp_path / "out",
                quality_floor=0.5,
                host_d_model_override=768,  # synthetic SAE is ~6 features
            )
        assert pipeline._call_count == 0

    def test_quality_floor_accepts_good_setup(
        self, tmp_path, synthetic_compressed_sae, stub_basis_swap
    ):
        """When the smallest-K basis is in the good/saturated tier under the
        override, the floor passes and the sweep proceeds.
        """
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2],
        )
        pipeline = _StubPipeline()
        from saeforge.sweep import sweep_pareto

        # host_d_model=8 means ratio = 6/8 = 0.75 → good; floor 0.5 passes.
        sweep_pareto(
            pipeline,
            encodings=[("rung4", encoding_dir)],
            output_dir=tmp_path / "out",
            quality_floor=0.5,
            host_d_model_override=8,
        )
        assert pipeline._call_count == 1

    def test_advisory_does_not_refuse_by_default(
        self, tmp_path, synthetic_compressed_sae, stub_basis_swap, capsys
    ):
        """Degenerate setup without `--quality-floor` prints advisory but the
        sweep runs all rows.
        """
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2],
        )
        pipeline = _StubPipeline()
        from saeforge.sweep import sweep_pareto

        sweep_pareto(
            pipeline,
            encodings=[("rung4", encoding_dir)],
            output_dir=tmp_path / "out",
            host_d_model_override=768,  # synthetic SAE → degenerate
        )
        captured = capsys.readouterr()
        # The advisory goes to stderr.
        assert "forge-quality advisory" in captured.err
        assert "degenerate" in captured.err.lower()
        assert pipeline._call_count == 1  # sweep still ran

    def test_failure_rows_carry_diagnostics(
        self, tmp_path, synthetic_compressed_sae, stub_basis_swap
    ):
        """Per-row failures still emit rows with the four diagnostic fields
        populated. The diagnostic is computed pre-forge so it survives even
        when the forge raises.
        """
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2],
        )
        pipeline = _StubPipeline(raise_on_calls=(1,))
        from saeforge.sweep import sweep_pareto

        with pytest.raises(RuntimeError, match="1 row"):
            sweep_pareto(
                pipeline,
                encodings=[("rung4", encoding_dir)],
                output_dir=tmp_path / "out",
                host_d_model_override=16,
            )
        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "frontier.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]["error_message"] is not None
        assert rows[0]["faithfulness_kl"] is None
        # Diagnostics still populated.
        assert rows[0]["host_d_model"] == 16
        assert rows[0]["basis_rank"] is not None
        assert rows[0]["quality_tier"] in {"degenerate", "undersized", "good", "saturated"}


# ---------------------------------------------------------------------------
# CLI: quality flags
# ---------------------------------------------------------------------------


class TestCLIQualityFlags:
    def test_quality_floor_out_of_range_exits_2(
        self, tmp_path, synthetic_compressed_sae
    ):
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2],
        )
        from saeforge.cli import main

        rc = main([
            "sweep-pareto",
            "--encoding", f"rung4:{encoding_dir}",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
            "--frontier-only",
            "--quality-floor", "1.5",
        ])
        assert rc == 2

    def test_quality_tier_thresholds_malformed_exits_2(
        self, tmp_path, synthetic_compressed_sae
    ):
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2],
        )
        from saeforge.cli import main

        rc = main([
            "sweep-pareto",
            "--encoding", f"rung4:{encoding_dir}",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
            "--frontier-only",
            "--quality-tier-thresholds", "bogus",
        ])
        assert rc == 2

    def test_quality_tier_thresholds_ordering_violation_exits_2(
        self, tmp_path, synthetic_compressed_sae
    ):
        encoding_dir = _make_pareto_dir(
            tmp_path / "rung4",
            synthetic_compressed_sae["checkpoint"],
            targets=[2],
        )
        from saeforge.cli import main

        rc = main([
            "sweep-pareto",
            "--encoding", f"rung4:{encoding_dir}",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
            "--frontier-only",
            "--quality-tier-thresholds", "saturated:0.5,good:1.0,undersized:0.25",
        ])
        assert rc == 2


# ---------------------------------------------------------------------------
# Auto-materialise CLI validation
# ---------------------------------------------------------------------------


class TestAutoMaterialiseCLIValidation:
    """Refusal scenarios for the new --auto-materialise CLI flags.

    These tests don't run materialisation end-to-end — that requires real
    torch + a host model. They focus on the validation/refusal contract
    spelled out in the spec.
    """

    def _common_args(self, tmp_path, sae_file):
        return [
            "sweep-pareto",
            "--encoding", f"mps:{sae_file}",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
        ]

    def _make_sae_file(self, tmp_path, synthetic_compressed_sae):
        """Provide a single .safetensors file path (no dir layout)."""
        return synthetic_compressed_sae["checkpoint"]

    def test_validator_flags_without_auto_materialise_refused(
        self, tmp_path, synthetic_compressed_sae
    ):
        sae_file = self._make_sae_file(tmp_path, synthetic_compressed_sae)
        from saeforge.cli import main

        # Build a directory layout for the non-auto-materialise mode.
        encoding_dir = _make_pareto_dir(
            tmp_path / "mps", sae_file, targets=[2],
        )
        rc = main([
            "sweep-pareto",
            "--encoding", f"mps:{encoding_dir}",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
            "--frontier-only",
            "--validation-threshold", "0.95",
        ])
        assert rc == 2

    def test_auto_materialise_with_dir_path_refused(
        self, tmp_path, synthetic_compressed_sae, capsys
    ):
        """--auto-materialise + --encoding LABEL:DIR is mixed mode → refuse."""
        sae_file = self._make_sae_file(tmp_path, synthetic_compressed_sae)
        encoding_dir = _make_pareto_dir(tmp_path / "mps", sae_file, targets=[2])
        validation = tmp_path / "v.txt"
        validation.write_text("hello\n")
        eval_p = tmp_path / "e.txt"
        eval_p.write_text("world\n")
        from saeforge.cli import main

        rc = main([
            "sweep-pareto",
            "--auto-materialise",
            "--encoding", f"mps:{encoding_dir}",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
            "--validation-prompts", str(validation),
            "--eval-prompts", str(eval_p),
            "--pareto", "2",
            "--layer", "8",
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "single .safetensors file" in err or "Mixed mode" in err

    def test_same_path_validation_eval_refused_by_default(
        self, tmp_path, synthetic_compressed_sae, capsys
    ):
        """--validation-prompts == --eval-prompts → refused unless --allow-...overlap."""
        sae_file = self._make_sae_file(tmp_path, synthetic_compressed_sae)
        shared = tmp_path / "shared.txt"
        shared.write_text("hello\n")
        from saeforge.cli import main

        rc = main([
            "sweep-pareto",
            "--auto-materialise",
            "--encoding", f"mps:{sae_file}",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
            "--validation-prompts", str(shared),
            "--eval-prompts", str(shared),
            "--pareto", "2",
            "--layer", "8",
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "leakage" in err.lower()

    def test_missing_required_flags_refused(
        self, tmp_path, synthetic_compressed_sae, capsys
    ):
        """--auto-materialise without --pareto/--layer/--validation-prompts → refuse."""
        sae_file = self._make_sae_file(tmp_path, synthetic_compressed_sae)
        from saeforge.cli import main

        rc = main([
            "sweep-pareto",
            "--auto-materialise",
            "--encoding", f"mps:{sae_file}",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "requires" in err.lower()

    def test_plan_only_and_frontier_only_mutually_exclusive(
        self, tmp_path, synthetic_compressed_sae, capsys
    ):
        sae_file = self._make_sae_file(tmp_path, synthetic_compressed_sae)
        validation = tmp_path / "v.txt"
        validation.write_text("hello\n")
        eval_p = tmp_path / "e.txt"
        eval_p.write_text("world\n")
        from saeforge.cli import main

        rc = main([
            "sweep-pareto",
            "--auto-materialise",
            "--encoding", f"mps:{sae_file}",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
            "--validation-prompts", str(validation),
            "--eval-prompts", str(eval_p),
            "--pareto", "2",
            "--layer", "8",
            "--frontier-only",
            "--plan-only",
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "mutually exclusive" in err.lower()

    def test_plan_only_without_auto_materialise_refused(
        self, tmp_path, synthetic_compressed_sae
    ):
        sae_file = self._make_sae_file(tmp_path, synthetic_compressed_sae)
        encoding_dir = _make_pareto_dir(tmp_path / "mps", sae_file, targets=[2])
        from saeforge.cli import main

        rc = main([
            "sweep-pareto",
            "--encoding", f"mps:{encoding_dir}",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
            "--plan-only",
        ])
        assert rc == 2

    def test_unknown_encoding_class_refused(
        self, tmp_path, synthetic_compressed_sae, capsys
    ):
        sae_file = self._make_sae_file(tmp_path, synthetic_compressed_sae)
        validation = tmp_path / "v.txt"
        validation.write_text("hello\n")
        eval_p = tmp_path / "e.txt"
        eval_p.write_text("world\n")
        from saeforge.cli import main

        rc = main([
            "sweep-pareto",
            "--auto-materialise",
            "--encoding", f"mps:{sae_file}",
            "--encoding-class", "mps:Bogus",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
            "--validation-prompts", str(validation),
            "--eval-prompts", str(eval_p),
            "--pareto", "2",
            "--layer", "8",
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "Bogus" in err or "supported" in err.lower()

    def test_plan_only_on_cold_cache_prints_miss(
        self, tmp_path, synthetic_compressed_sae, capsys
    ):
        """--plan-only with no prior materialisation prints MISS (cold) to stderr."""
        sae_file = self._make_sae_file(tmp_path, synthetic_compressed_sae)
        validation = tmp_path / "v.txt"
        validation.write_text("hello\n")
        eval_p = tmp_path / "e.txt"
        eval_p.write_text("world\n")
        from saeforge.cli import main

        rc = main([
            "sweep-pareto",
            "--auto-materialise",
            "--encoding", f"mps:{sae_file}",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
            "--validation-prompts", str(validation),
            "--eval-prompts", str(eval_p),
            "--pareto", "2,4",
            "--layer", "8",
            "--plan-only",
        ])
        assert rc == 0
        err = capsys.readouterr().err
        assert "label=mps" in err
        assert "cache_status=MISS" in err
        assert "cold" in err
        # No frontier.jsonl written.
        assert not (tmp_path / "out" / "frontier.jsonl").is_file()

    def test_force_rematerialise_without_auto_materialise_refused(
        self, tmp_path, synthetic_compressed_sae
    ):
        sae_file = self._make_sae_file(tmp_path, synthetic_compressed_sae)
        encoding_dir = _make_pareto_dir(tmp_path / "mps", sae_file, targets=[2])
        from saeforge.cli import main

        rc = main([
            "sweep-pareto",
            "--encoding", f"mps:{encoding_dir}",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
            "--force-rematerialise",
        ])
        assert rc == 2

    def test_assign_phase_knobs_without_auto_materialise_refused(
        self, tmp_path, synthetic_compressed_sae, capsys
    ):
        """--assign-phase-knobs is auto-materialise-only (polygram 0.6.0
        flag is only meaningful at materialisation time)."""
        sae_file = self._make_sae_file(tmp_path, synthetic_compressed_sae)
        encoding_dir = _make_pareto_dir(tmp_path / "mps", sae_file, targets=[2])
        from saeforge.cli import main

        rc = main([
            "sweep-pareto",
            "--encoding", f"mps:{encoding_dir}",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
            "--assign-phase-knobs",
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "--assign-phase-knobs" in err
        assert "--auto-materialise" in err

    def test_assign_phase_knobs_surfaces_in_plan_only(
        self, tmp_path, synthetic_compressed_sae, capsys
    ):
        """--plan-only stderr block should expose the flag so users can see
        why the cache will MISS."""
        sae_file = self._make_sae_file(tmp_path, synthetic_compressed_sae)
        validation = tmp_path / "v.txt"
        validation.write_text("hello\n")
        eval_p = tmp_path / "e.txt"
        eval_p.write_text("world\n")
        from saeforge.cli import main

        rc = main([
            "sweep-pareto",
            "--auto-materialise",
            "--encoding", f"mps:{sae_file}",
            "--host-model", "gpt2",
            "--output-dir", str(tmp_path / "out"),
            "--validation-prompts", str(validation),
            "--eval-prompts", str(eval_p),
            "--pareto", "2",
            "--layer", "8",
            "--plan-only",
            "--assign-phase-knobs",
        ])
        assert rc == 0
        err = capsys.readouterr().err
        assert "assign_phase_knobs=True" in err


# ---------------------------------------------------------------------------
# Provenance row population (live regression for the indent-mismatch bug
# the first MBP Axis-4 smoke surfaced: the success-path ParetoFrontierRow
# construction was missing the three provenance kwargs because an
# earlier replace_all only matched 8-space-indent sites and missed the
# 4-space-indent function-level return.)
# ---------------------------------------------------------------------------


class TestProvenanceRowPopulation:
    """Pin that provenance fields flow through to ALL three _process_row
    exit paths: success, failure, frontier-only. The first MBP smoke
    revealed the success path was missing them.
    """

    def test_success_path_carries_provenance(
        self, tmp_path, synthetic_compressed_sae, stub_basis_swap, monkeypatch
    ):
        from saeforge.sweep import _process_row

        pipeline = _StubPipeline()  # .run succeeds, returns _StubResult
        result = _process_row(
            pipeline=pipeline,
            label="rung4",
            target_k=2,
            ckpt_path=tmp_path / "fake.safetensors",
            manifest_entry=None,
            sweep_output_dir=tmp_path,
            frontier_only=False,
            forge_kwargs={},
            host_d_model=768,
            basis_rank=4,
            quality_ratio=0.005,
            quality_tier="degenerate",
            provenance_validation_threshold=0.95,
            provenance_encoding_class="Rung4",
            provenance_validation_eval_overlap=False,
        )
        # Pre-fix, validation_threshold/encoding_class/validation_eval_overlap
        # would all be None here. Post-fix, they carry through.
        assert result.validation_threshold == 0.95
        assert result.encoding_class == "Rung4"
        assert result.validation_eval_overlap is False

    def test_failure_path_carries_provenance(self, tmp_path, stub_basis_swap):
        from saeforge.sweep import _process_row

        pipeline = _StubPipeline(raise_on_calls=(1,))
        result = _process_row(
            pipeline=pipeline,
            label="rung4",
            target_k=2,
            ckpt_path=tmp_path / "fake.safetensors",
            manifest_entry=None,
            sweep_output_dir=tmp_path,
            frontier_only=False,
            forge_kwargs={},
            provenance_validation_threshold=0.7,
            provenance_encoding_class="MPSRung1",
            provenance_validation_eval_overlap=True,
        )
        assert result.error_message is not None  # forge raised
        assert result.validation_threshold == 0.7
        assert result.encoding_class == "MPSRung1"
        assert result.validation_eval_overlap is True

    def test_frontier_only_path_carries_provenance(self, tmp_path, stub_basis_swap):
        from saeforge.sweep import _process_row

        pipeline = _StubPipeline()  # never called under frontier_only
        result = _process_row(
            pipeline=pipeline,
            label="rung4",
            target_k=2,
            ckpt_path=tmp_path / "fake.safetensors",
            manifest_entry=None,
            sweep_output_dir=tmp_path,
            frontier_only=True,
            forge_kwargs={},
            provenance_validation_threshold=0.95,
            provenance_encoding_class="HEA_Rung2",
            provenance_validation_eval_overlap=False,
        )
        assert pipeline._call_count == 0
        assert result.validation_threshold == 0.95
        assert result.encoding_class == "HEA_Rung2"
        assert result.validation_eval_overlap is False
