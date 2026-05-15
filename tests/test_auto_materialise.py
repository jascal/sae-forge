"""Tests for `saeforge.auto_materialise` — auto-materialise pre-step.

Coverage:
- AutoMaterialiseSpec validation
- Encoding-class registry
- Cache key: same inputs → same key; threshold change → miss; content-hash
  not path-hash
- is_cache_hit: cold / miss-on-diff / hit / missing per-K files
- format_plan_only_block output shape

End-to-end materialise() against a real polygram chain (validator +
plan_pareto + apply) requires torch + a host and is gated as integration
work; the cache-key and CLI-validation paths exercise the full surface
this PR adds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

# polygram is an optional extra. The cache-key / spec-validation paths
# are pure-Python and don't need it, but the encoding-class registry
# resolution + the `HEA_Rung2` default tests do. Per-test
# `pytest.importorskip("polygram")` lives on the affected classes.

from saeforge.auto_materialise import (
    AutoMaterialiseSpec,
    _ENCODING_CLASS_REGISTRY,
    _resolve_encoding_class,
    compute_cache_key,
    format_plan_only_block,
    is_cache_hit,
)


def _write_dummy_sae(path: Path, *, n_features: int = 8, d_model: int = 16):
    """Write a small safetensors with the keys polygram expects."""
    rng = np.random.default_rng(0)
    save_file(
        {
            "W_dec": rng.standard_normal((n_features, d_model)).astype(np.float32),
            "W_enc": rng.standard_normal((d_model, n_features)).astype(np.float32),
            "b_dec": np.zeros(d_model, dtype=np.float32),
            "b_enc": np.zeros(n_features, dtype=np.float32),
        },
        str(path),
    )


# ---------------------------------------------------------------------------
# AutoMaterialiseSpec
# ---------------------------------------------------------------------------


class TestAutoMaterialiseSpec:
    def test_defaults_to_mps_rung1(self, tmp_path):
        sae = tmp_path / "sae.safetensors"
        _write_dummy_sae(sae)
        spec = AutoMaterialiseSpec(label="mps", sae_checkpoint=sae)
        assert spec.encoding_class == "MPSRung1"
        assert spec.encoding_kwargs == {}

    def test_rejects_unknown_encoding_class(self, tmp_path):
        sae = tmp_path / "sae.safetensors"
        _write_dummy_sae(sae)
        with pytest.raises(ValueError, match="encoding_class"):
            AutoMaterialiseSpec(label="x", sae_checkpoint=sae, encoding_class="Bogus")

    def test_rejects_empty_label(self, tmp_path):
        sae = tmp_path / "sae.safetensors"
        _write_dummy_sae(sae)
        with pytest.raises(ValueError, match="label"):
            AutoMaterialiseSpec(label="", sae_checkpoint=sae)

    def test_accepts_hea_rung2_with_qubits(self, tmp_path):
        sae = tmp_path / "sae.safetensors"
        _write_dummy_sae(sae)
        spec = AutoMaterialiseSpec(
            label="rung2",
            sae_checkpoint=sae,
            encoding_class="HEA_Rung2",
            encoding_kwargs={"n_qubits": 5},
        )
        assert spec.encoding_kwargs["n_qubits"] == 5

    def test_hea_rung2_polygram_default_is_n_qubits_3(self, tmp_path):
        """Pins polygram's HEA_Rung2 default n_qubits=3 (cap=8). The CLI
        help text describes the fallback users hit when --encoding-qubits
        is omitted; this test guards against a silent polygram default
        change.

        Note: HEA_Rung2 also requires `depth`. The CLI handler defaults
        depth=2 (standard) when --encoding-class HEA_Rung2 is set, so
        callers only need to pass --encoding-qubits to fully configure
        the encoding.
        """
        pytest.importorskip("polygram")
        from polygram import HEA_Rung2

        # depth=2 is the CLI's internal default; n_qubits=3 is polygram's.
        default_instance = HEA_Rung2(depth=2)
        assert default_instance.max_features == 8


# ---------------------------------------------------------------------------
# Encoding class registry
# ---------------------------------------------------------------------------


class TestEncodingClassRegistry:
    @pytest.mark.parametrize("name", list(_ENCODING_CLASS_REGISTRY))
    def test_resolves_supported_classes(self, name):
        pytest.importorskip("polygram")
        cls = _resolve_encoding_class(name)
        assert cls.__name__ == name

    def test_rejects_unknown_class_name(self):
        pytest.importorskip("polygram")
        with pytest.raises(ValueError, match="unknown"):
            _resolve_encoding_class("Bogus")


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class TestCacheKey:
    def _build_inputs(self, tmp_path):
        sae = tmp_path / "sae.safetensors"
        _write_dummy_sae(sae)
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("hello world\n")
        spec = AutoMaterialiseSpec(label="mps", sae_checkpoint=sae)
        return spec, prompts

    def _common_kwargs(self):
        return dict(
            validation_threshold=0.7,
            jaccard_threshold=0.3,
            layer=8,
            model_name="gpt2",
            targets=[2, 4],
            score_field="polygram_overlap",
            rep_selection="scale_aware",
        )

    def test_same_inputs_same_key(self, tmp_path):
        spec, prompts = self._build_inputs(tmp_path)
        k1 = compute_cache_key(spec=spec, validation_prompts_path=prompts, **self._common_kwargs())
        k2 = compute_cache_key(spec=spec, validation_prompts_path=prompts, **self._common_kwargs())
        assert k1 == k2

    def test_threshold_change_yields_different_key(self, tmp_path):
        spec, prompts = self._build_inputs(tmp_path)
        kwargs = self._common_kwargs()
        k1 = compute_cache_key(spec=spec, validation_prompts_path=prompts, **kwargs)
        kwargs["validation_threshold"] = 0.95
        k2 = compute_cache_key(spec=spec, validation_prompts_path=prompts, **kwargs)
        assert k1 != k2
        assert k1["validation_threshold"] == 0.7
        assert k2["validation_threshold"] == 0.95

    def test_score_field_change_yields_different_key(self, tmp_path):
        spec, prompts = self._build_inputs(tmp_path)
        kwargs = self._common_kwargs()
        k1 = compute_cache_key(spec=spec, validation_prompts_path=prompts, **kwargs)
        kwargs["score_field"] = "jaccard"
        k2 = compute_cache_key(spec=spec, validation_prompts_path=prompts, **kwargs)
        assert k1 != k2

    def test_content_hash_not_path_hash(self, tmp_path):
        """Renaming a file with identical content should NOT change the key."""
        spec, prompts = self._build_inputs(tmp_path)
        k1 = compute_cache_key(spec=spec, validation_prompts_path=prompts, **self._common_kwargs())

        # Copy SAE to a new path with identical content; build a new spec.
        new_sae = tmp_path / "subdir" / "renamed.safetensors"
        new_sae.parent.mkdir()
        new_sae.write_bytes(spec.sae_checkpoint.read_bytes())
        new_spec = AutoMaterialiseSpec(label="mps", sae_checkpoint=new_sae)
        k2 = compute_cache_key(
            spec=new_spec, validation_prompts_path=prompts, **self._common_kwargs()
        )
        # SHA matches, path differs.
        assert k1["sae_checkpoint_sha256"] == k2["sae_checkpoint_sha256"]
        assert k1["sae_checkpoint_path"] != k2["sae_checkpoint_path"]

    def test_targets_normalised_via_sort(self, tmp_path):
        """`targets=[4, 2]` should produce the same key as `targets=[2, 4]`."""
        spec, prompts = self._build_inputs(tmp_path)
        kwargs = self._common_kwargs()
        kwargs["targets"] = [4, 2]
        k_unsorted = compute_cache_key(spec=spec, validation_prompts_path=prompts, **kwargs)
        kwargs["targets"] = [2, 4]
        k_sorted = compute_cache_key(spec=spec, validation_prompts_path=prompts, **kwargs)
        assert k_unsorted == k_sorted


# ---------------------------------------------------------------------------
# is_cache_hit
# ---------------------------------------------------------------------------


class TestIsCacheHit:
    def _setup(self, tmp_path):
        sae = tmp_path / "sae.safetensors"
        _write_dummy_sae(sae)
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("hello\n")
        spec = AutoMaterialiseSpec(label="mps", sae_checkpoint=sae)
        key = compute_cache_key(
            spec=spec,
            validation_prompts_path=prompts,
            validation_threshold=0.7,
            jaccard_threshold=0.3,
            layer=8,
            model_name="gpt2",
            targets=[2, 4],
            score_field="polygram_overlap",
            rep_selection="scale_aware",
        )
        materialised_dir = tmp_path / "_materialised" / "mps"
        materialised_dir.mkdir(parents=True)
        return materialised_dir, key

    def test_cold_cache(self, tmp_path):
        materialised_dir, key = self._setup(tmp_path)
        hit, diff = is_cache_hit(materialised_dir, key)
        assert hit is False
        assert diff == ["cold"]

    def test_hit_when_meta_and_files_match(self, tmp_path):
        materialised_dir, key = self._setup(tmp_path)
        (materialised_dir / "auto_materialise_meta.json").write_text(json.dumps(key))
        (materialised_dir / "pareto").mkdir()
        for k in key["targets"]:
            (materialised_dir / "pareto" / f"k_{k}.safetensors").write_text("")
        hit, diff = is_cache_hit(materialised_dir, key)
        assert hit is True
        assert diff == []

    def test_miss_when_meta_field_differs(self, tmp_path):
        materialised_dir, key = self._setup(tmp_path)
        # Write meta with a different threshold.
        on_disk = dict(key)
        on_disk["validation_threshold"] = 0.95
        (materialised_dir / "auto_materialise_meta.json").write_text(json.dumps(on_disk))
        hit, diff = is_cache_hit(materialised_dir, key)
        assert hit is False
        assert "validation_threshold" in diff

    def test_miss_when_per_k_file_absent(self, tmp_path):
        materialised_dir, key = self._setup(tmp_path)
        (materialised_dir / "auto_materialise_meta.json").write_text(json.dumps(key))
        (materialised_dir / "pareto").mkdir()
        # Only write k_2 — k_4 is missing.
        (materialised_dir / "pareto" / "k_2.safetensors").write_text("")
        hit, diff = is_cache_hit(materialised_dir, key)
        assert hit is False
        assert any("missing_k_4" in d for d in diff)

    def test_hit_when_only_paths_differ(self, tmp_path):
        """The cache is content-addressed: renaming or moving an input file
        with identical content does NOT invalidate the cache. The
        ``*_path`` fields are recorded in the meta for human inspection
        but excluded from the cache-hit diff.
        """
        materialised_dir, key = self._setup(tmp_path)
        # Write meta with different paths but same SHAs / other fields.
        on_disk = dict(key)
        on_disk["sae_checkpoint_path"] = "/somewhere/else/sae.safetensors"
        on_disk["validation_prompts_path"] = "/elsewhere/prompts.txt"
        (materialised_dir / "auto_materialise_meta.json").write_text(json.dumps(on_disk))
        (materialised_dir / "pareto").mkdir()
        for k in key["targets"]:
            (materialised_dir / "pareto" / f"k_{k}.safetensors").write_text("")
        hit, diff = is_cache_hit(materialised_dir, key)
        assert hit is True, (
            f"expected cache hit when only path fields differ; got diff={diff}"
        )


# ---------------------------------------------------------------------------
# format_plan_only_block
# ---------------------------------------------------------------------------


class TestFormatPlanOnlyBlock:
    def test_hit_block_says_hit(self, tmp_path):
        sae = tmp_path / "sae.safetensors"
        _write_dummy_sae(sae)
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("hello\nworld\n")
        spec = AutoMaterialiseSpec(label="mps", sae_checkpoint=sae)
        key = compute_cache_key(
            spec=spec,
            validation_prompts_path=prompts,
            validation_threshold=0.7,
            jaccard_threshold=0.3,
            layer=8,
            model_name="gpt2",
            targets=[2, 4],
            score_field="polygram_overlap",
            rep_selection="scale_aware",
        )
        block = format_plan_only_block(
            spec=spec, cache_key=key, diff_fields=[], cache_hit=True,
            n_prompts=2, avg_prompt_tokens=1.0,
        )
        assert "label=mps" in block
        assert "cache_status=HIT" in block
        assert key["sae_checkpoint_sha256"][:10] in block

    def test_miss_block_lists_diff_fields(self, tmp_path):
        sae = tmp_path / "sae.safetensors"
        _write_dummy_sae(sae)
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("hello\n")
        spec = AutoMaterialiseSpec(label="mps", sae_checkpoint=sae)
        key = compute_cache_key(
            spec=spec,
            validation_prompts_path=prompts,
            validation_threshold=0.7,
            jaccard_threshold=0.3,
            layer=8,
            model_name="gpt2",
            targets=[2],
            score_field="polygram_overlap",
            rep_selection="scale_aware",
        )
        block = format_plan_only_block(
            spec=spec, cache_key=key, diff_fields=["validation_threshold"], cache_hit=False,
            n_prompts=1, avg_prompt_tokens=1.0,
        )
        assert "MISS (validation_threshold)" in block


# ---------------------------------------------------------------------------
# assign_phase_knobs plumbing (polygram >=0.6.0)
# ---------------------------------------------------------------------------


class TestAssignPhaseKnobs:
    """Cover the polygram 0.6.0 `assign_phase_knobs` kwarg plumbing.

    Three load-bearing properties:
    - cache_key always includes the flag (default False), so toggling busts
      the cache deterministically;
    - format_plan_only_block surfaces it for human-readable plan dumps;
    - _run_materialisation_chain forwards it to polygram.from_sae_lens
      only when True (omitted when False for forward-compat with older
      polygrams that don't accept the kwarg).
    """

    def _inputs(self, tmp_path):
        sae = tmp_path / "sae.safetensors"
        _write_dummy_sae(sae)
        prompts = tmp_path / "p.txt"
        prompts.write_text("hello\n")
        spec = AutoMaterialiseSpec(label="mps", sae_checkpoint=sae)
        return spec, prompts

    def _common(self):
        return dict(
            validation_threshold=0.7,
            jaccard_threshold=0.3,
            layer=8,
            model_name="gpt2",
            targets=[2],
            score_field="polygram_overlap",
            rep_selection="scale_aware",
        )

    def test_cache_key_default_false(self, tmp_path):
        spec, prompts = self._inputs(tmp_path)
        key = compute_cache_key(
            spec=spec, validation_prompts_path=prompts, **self._common()
        )
        assert key["assign_phase_knobs"] is False

    def test_cache_key_flip_invalidates(self, tmp_path):
        spec, prompts = self._inputs(tmp_path)
        k_off = compute_cache_key(
            spec=spec, validation_prompts_path=prompts, **self._common()
        )
        k_on = compute_cache_key(
            spec=spec, validation_prompts_path=prompts,
            assign_phase_knobs=True, **self._common()
        )
        assert k_off != k_on
        assert k_off["assign_phase_knobs"] is False
        assert k_on["assign_phase_knobs"] is True

    def test_plan_only_block_surfaces_flag(self, tmp_path):
        spec, prompts = self._inputs(tmp_path)
        key = compute_cache_key(
            spec=spec, validation_prompts_path=prompts,
            assign_phase_knobs=True, **self._common()
        )
        block = format_plan_only_block(
            spec=spec, cache_key=key, diff_fields=[], cache_hit=True,
            n_prompts=1, avg_prompt_tokens=1.0,
        )
        assert "assign_phase_knobs=True" in block

    def test_run_chain_forwards_true_to_from_sae_lens(self, tmp_path, monkeypatch):
        """assign_phase_knobs=True reaches polygram.from_sae_lens."""
        pytest.importorskip("polygram")
        import polygram

        from saeforge.auto_materialise import _run_materialisation_chain

        captured: dict[str, object] = {}

        def fake_from_sae_lens(records, slot_ids, **kw):
            captured.update(kw)
            # Short-circuit before BehaviouralValidator (needs torch + a real host).
            raise RuntimeError("captured-and-stop")

        monkeypatch.setattr(polygram, "from_sae_lens", fake_from_sae_lens)

        spec, prompts = self._inputs(tmp_path)
        materialised_dir = tmp_path / "out"
        materialised_dir.mkdir()

        with pytest.raises(RuntimeError, match="captured-and-stop"):
            _run_materialisation_chain(
                spec=spec,
                validation_prompts_path=prompts,
                materialised_dir=materialised_dir,
                assign_phase_knobs=True,
                **self._common(),
            )
        assert captured.get("assign_phase_knobs") is True
        assert "encoding" in captured

    def test_run_chain_omits_kwarg_when_false(self, tmp_path, monkeypatch):
        """Default (False) omits the kwarg entirely so older polygrams
        without ``assign_phase_knobs`` parameter still work."""
        pytest.importorskip("polygram")
        import polygram

        from saeforge.auto_materialise import _run_materialisation_chain

        captured: dict[str, object] = {}

        def fake_from_sae_lens(records, slot_ids, **kw):
            captured.update(kw)
            raise RuntimeError("captured-and-stop")

        monkeypatch.setattr(polygram, "from_sae_lens", fake_from_sae_lens)

        spec, prompts = self._inputs(tmp_path)
        materialised_dir = tmp_path / "out2"
        materialised_dir.mkdir()

        with pytest.raises(RuntimeError, match="captured-and-stop"):
            _run_materialisation_chain(
                spec=spec,
                validation_prompts_path=prompts,
                materialised_dir=materialised_dir,
                # assign_phase_knobs left at default False
                **self._common(),
            )
        assert "assign_phase_knobs" not in captured
        assert "encoding" in captured
