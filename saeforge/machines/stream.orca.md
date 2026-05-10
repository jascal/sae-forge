# machine StreamMachine

The outermost forge sub-machine: handles the multi-shard stream
loop. Drives `init → streaming → done` for single-shard runs (v0.1
default) and `init → streaming → next_shard → streaming → ... →
done` for continual-learning multi-shard runs.

`streaming` is a compound state that invokes `RefineMachine`. When
`RefineMachine` reaches its `[final]` (`exiting`), this machine
fires `refine_done` and arbitrates between `next_shard` (with
`advance_to_next_task`) and `done` (with `save_final_model`),
based on guards that read `ctx.advance_stream` and
`ctx.should_continue`. Both ctx fields are set by
`evaluate_faithfulness` inside `RefineMachine` and are visible here
because the orchestrator shares one ctx dict across the hierarchy.

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
| error_origin_machine | string |  |
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
| _machine_path | string | stream |

## state init [initial]
## state streaming

- invoke: RefineMachine
- on_done: -> refine_done

## state next_shard
## state done [final]
## state failed [final]

## guards

| Name | Expression |
|------|------------|
| stream_advance | ctx.advance_stream == true |
| terminate_run | ctx.advance_stream == false and ctx.should_continue == false |

## transitions

| Source | Event | Guard | Target | Action |
|--------|-------|-------|--------|--------|
| init | start |  | streaming |  |
| init | error |  | failed | log_error |
| streaming | refine_done | stream_advance | next_shard | advance_to_next_task |
| streaming | refine_done | terminate_run | done | save_final_model |
| streaming | error |  | failed | log_error |
| next_shard | shard_loaded |  | streaming |  |
| next_shard | error |  | failed | log_error |

## actions

| Name | Signature |
|------|-----------|
| advance_to_next_task | (ctx) -> Context |
| save_final_model | (ctx) -> Context |
| log_error | (ctx) -> Context |
