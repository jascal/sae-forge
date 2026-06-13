"""CLI tests for the capability-trained encoder flags (change add-capability-trained-encoder, task 4)."""
import json

from saeforge.cli import _build_parser, main
from saeforge.sweep import ParetoFrontierRow


def test_sweep_capability_flags_parse():
    parser = _build_parser()
    args = parser.parse_args([
        "sweep-capability", "--dataset-config", "d.yaml", "--host", "h",
        "--widths", "8", "--output-dir", "out",
        "--train-encoder", "--basis-order", "readout_aligned",
        "--readout-fallback", "downstream_decode", "--train-steps", "120",
    ])
    assert args.train_encoder is True
    assert args.basis_order == "readout_aligned"
    assert args.readout_fallback == "downstream_decode"
    assert args.train_steps == 120


def test_sweep_capability_flag_defaults():
    parser = _build_parser()
    args = parser.parse_args([
        "sweep-capability", "--dataset-config", "d.yaml", "--host", "h",
        "--widths", "8", "--output-dir", "out",
    ])
    assert args.train_encoder is False
    assert args.basis_order == "row_norm"
    assert args.readout_fallback is None


def test_recommend_trained_margin_default():
    parser = _build_parser()
    args = parser.parse_args(["recommend", "--frontier", "f.jsonl", "--target", "retained-mauc-vs-host>=0.5"])
    assert args.trained_margin == 0.02


def _trained_row(delta, overfit):
    return ParetoFrontierRow(
        encoding_label="e", target_n_features_kept=8, n_features_kept_actual=8,
        pareto_reached_target=True, faithfulness_kl=None, perplexity=None, final_fine_tune_loss=None,
        sae_checkpoint="x", forged_model_path=None, elapsed_seconds=1.0, error_message=None,
        host_baseline_mauc=0.9, forge_mauc=0.8, retained_mauc_vs_host=0.88,
        retained_mauc_trained=0.88 + delta, retained_mauc_pinv_baseline=0.88, delta_heldout=delta,
        encoder_trained=True, overfit_flag=overfit, encoder_artifact_path="/tmp/e.npy",
    )


def _write_frontier(tmp_path, row):
    p = tmp_path / "frontier.jsonl"
    p.write_text(json.dumps(row.to_json_dict()) + "\n")
    return p


def test_recommend_prefers_trained_when_margin_cleared(tmp_path, capsys):
    p = _write_frontier(tmp_path, _trained_row(delta=0.05, overfit=False))
    rc = main(["recommend", "--frontier", str(p), "--target", "retained-mauc-vs-host>=0.5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "recommended_encoder:      trained" in out
    assert "effective_retained_mauc:  0.9300" in out
    assert "/tmp/e.npy" in out


def test_recommend_keeps_pinv_below_margin(tmp_path, capsys):
    p = _write_frontier(tmp_path, _trained_row(delta=0.005, overfit=False))
    main(["recommend", "--frontier", str(p), "--target", "retained-mauc-vs-host>=0.5"])
    out = capsys.readouterr().out
    assert "recommended_encoder:      pinv" in out
    assert "effective_retained_mauc:  0.8800" in out


def test_recommend_keeps_pinv_when_overfit(tmp_path, capsys):
    p = _write_frontier(tmp_path, _trained_row(delta=0.05, overfit=True))
    main(["recommend", "--frontier", str(p), "--target", "retained-mauc-vs-host>=0.5"])
    out = capsys.readouterr().out
    assert "recommended_encoder:      pinv" in out


def test_recommend_custom_trained_margin(tmp_path, capsys):
    # delta 0.03 clears the default 0.02 but not a custom 0.05
    p = _write_frontier(tmp_path, _trained_row(delta=0.03, overfit=False))
    main(["recommend", "--frontier", str(p), "--target", "retained-mauc-vs-host>=0.5",
          "--trained-margin", "0.05"])
    out = capsys.readouterr().out
    assert "recommended_encoder:      pinv" in out


def test_recommend_json_carries_trained_fields(tmp_path, capsys):
    p = _write_frontier(tmp_path, _trained_row(delta=0.05, overfit=False))
    main(["recommend", "--frontier", str(p), "--target", "retained-mauc-vs-host>=0.5", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["recommended_encoder"] == "trained"
    assert payload["effective_retained_mauc"] == 0.93
    assert payload["trained_margin"] == 0.02
