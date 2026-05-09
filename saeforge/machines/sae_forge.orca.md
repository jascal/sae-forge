# machine SaeForge

The v0.2 forge outer-loop FSM with continual-learning support.
Drives load → activations_scanned → compress↔regrow basis loop →
project → fine-tune → evaluate → optional next-task / refine /
done. All v0.2 fields default to values that recover v0.1
single-shard byte-identical behavior.

The three loops:
- **Stream loop** (per shard): `evaluated → loaded` when
  `ctx.advance_stream == true`.
- **Refine loop** (per-shard convergence): `evaluated → compressed`
  when `ctx.should_continue == true` (existing v0.1 behavior).
- **Basis loop** (compress↔regrow refinement): `compressed → regrown`
  / `regrown → compressed` driven by `ctx.inner_refine_idx` versus
  `ctx.inner_refine_passes`.

All loop predicates are encoded as rich orca guards against ctx
fields directly — no precomputed flat-bool flags.

## context

| Field | Type | Default |
|-------|------|---------|
| sae_checkpoint | string |  |
| host_model_id | string |  |
| output_dir | string |  |
| iterations | int | 1 |
| regrow_count | int | 0 |
| current_iter | int | 0 |
| current_sae_path | string |  |
| compressed_sae_path | string |  |
| regrown_sae_path | string |  |
| current_feature_count | int | 0 |
| projected_weights_path | string |  |
| finetuned_model_path | string |  |
| faithfulness | float | 0.0 |
| min_faithfulness | float | 0.0 |
| perplexity | float | 1000000.0 |
| best_perplexity | float | 1000000.0 |
| final_model_path | string |  |
| error_message | string |  |
| quantum_aware | bool | false |
| n_params | int | 0 |
| transitions_log | list | [] |
| should_continue | bool | false |
| compression | dict |  |
| epoch_compression | dict |  |
| regrow | dict |  |
| n_tasks | int | 1 |
| task_idx | int | 0 |
| task_trigger | string | labeled |
| token_budget_per_task | int | 0 |
| tokens_seen_in_task | int | 0 |
| loss_delta_threshold | float | 0.0 |
| recent_eval_losses | list | [] |
| advance_stream | bool | false |
| inner_refine_passes | int | 1 |
| inner_refine_idx | int | 0 |
| protect_top_k | int | 0 |
| protect_score | string | mean_act |
| protected_features | list | [] |
| activation_buffer_size | int | 4096 |
| feature_usage | list | [] |
| replay_ratio | float | 0.0 |
| replay_policy | string | reservoir |
| replay_buffer_size | int | 0 |
| task_iterator_id | string |  |

## state init [initial]
## state loaded
## state activations_scanned
## state compressed
## state regrown
## state projected
## state finetuned
## state evaluated
## state done [final]
## state failed [final]

## guards

| Name | Expression |
| should_regrow | ctx.regrow_count > 0 |
| no_regrow_more_passes | ctx.regrow_count == 0 and ctx.inner_refine_idx < ctx.inner_refine_passes |
| no_regrow_done | ctx.regrow_count == 0 and ctx.inner_refine_idx >= ctx.inner_refine_passes |
| basis_loop_continue | ctx.inner_refine_idx < ctx.inner_refine_passes |
| basis_loop_done | ctx.inner_refine_idx >= ctx.inner_refine_passes |
| stream_advance | ctx.advance_stream == true |
| refine_same_shard | ctx.advance_stream == false and ctx.should_continue == true |
| terminate_run | ctx.advance_stream == false and ctx.should_continue == false |

## transitions

| Source | Event | Guard | Target | Action |
| init | start |  | loaded | load_sae_and_corpus |
| loaded | load_done |  | activations_scanned | scan_activations |
| loaded | error |  | failed | log_error |
| activations_scanned | scan_done |  | compressed | compress_with_polygram |
| activations_scanned | error |  | failed | log_error |
| compressed | compress_done | should_regrow | regrown | perform_regrowth |
| compressed | compress_done | no_regrow_more_passes | compressed | compress_with_polygram |
| compressed | compress_done | no_regrow_done | projected | project_to_subspace |
| compressed | error |  | failed | log_error |
| regrown | regrowth_done | basis_loop_continue | compressed | compress_with_polygram |
| regrown | regrowth_done | basis_loop_done | projected | project_to_subspace |
| regrown | error |  | failed | log_error |
| projected | projection_done |  | finetuned | fine_tune_model |
| projected | error |  | failed | log_error |
| finetuned | finetune_done |  | evaluated | evaluate_faithfulness |
| finetuned | error |  | failed | log_error |
| evaluated | eval_done | stream_advance | loaded | advance_to_next_task |
| evaluated | eval_done | refine_same_shard | compressed | rotate_for_next_iter |
| evaluated | eval_done | terminate_run | done | save_final_model |
| evaluated | error |  | failed | log_error |

## actions

| Name | Signature |
| load_sae_and_corpus | (ctx) -> Context |
| scan_activations | (ctx) -> Context |
| compress_with_polygram | (ctx) -> Context |
| perform_regrowth | (ctx) -> Context |
| project_to_subspace | (ctx) -> Context |
| fine_tune_model | (ctx) -> Context |
| evaluate_faithfulness | (ctx) -> Context |
| advance_to_next_task | (ctx) -> Context |
| rotate_for_next_iter | (ctx) -> Context |
| save_final_model | (ctx) -> Context |
| log_error | (ctx) -> Context |
