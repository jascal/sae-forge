# machine SaeForge

The v0.1 forge outer-loop FSM. Drives load → compress → optional
regrow → project → fine-tune → evaluate → optional iterate → done|failed.
Compress, regrow, and fine-tune are no-op pass-throughs in v0.1; the
v0.2 milestone replaces them with calls into Polygram and an HF
trainer.

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

## state init [initial]
## state loaded
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
| no_regrow | ctx.regrow_count == 0 |
| should_continue_loop | ctx.should_continue == true |
| done_iterating | ctx.should_continue == false |

## transitions

| Source | Event | Guard | Target | Action |
| init | start |  | loaded | load_sae_and_corpus |
| loaded | load_done |  | compressed | compress_with_polygram |
| loaded | error |  | failed | log_error |
| compressed | compress_done | should_regrow | regrown | perform_regrowth |
| compressed | compress_done | no_regrow | projected | project_to_subspace |
| compressed | error |  | failed | log_error |
| regrown | regrowth_done |  | projected | project_to_subspace |
| regrown | error |  | failed | log_error |
| projected | projection_done |  | finetuned | fine_tune_model |
| projected | error |  | failed | log_error |
| finetuned | finetune_done |  | evaluated | evaluate_faithfulness |
| finetuned | error |  | failed | log_error |
| evaluated | eval_done | done_iterating | done | save_final_model |
| evaluated | eval_done | should_continue_loop | compressed | rotate_for_next_iter |
| evaluated | error |  | failed | log_error |

## actions

| Name | Signature |
| load_sae_and_corpus | (ctx) -> Context |
| compress_with_polygram | (ctx) -> Context |
| perform_regrowth | (ctx) -> Context |
| project_to_subspace | (ctx) -> Context |
| fine_tune_model | (ctx) -> Context |
| evaluate_faithfulness | (ctx) -> Context |
| rotate_for_next_iter | (ctx) -> Context |
| save_final_model | (ctx) -> Context |
| log_error | (ctx) -> Context |
