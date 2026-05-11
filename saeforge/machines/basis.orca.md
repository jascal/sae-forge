# machine BasisMachine

The innermost forge sub-machine: drives the compress↔regrow basis
loop, then projects to subspace and fine-tunes. Reaches `done`
when fine-tune completes; the parent (RefineMachine) then fires
its own `basis_done` event and runs `evaluate_faithfulness`.

This machine is invoked by `RefineMachine.refining` per the
hierarchical-fsm capability. Action handlers and ctx are shared
with the parent at orchestration time (the orchestrator passes
the parent's ctx dict into the child by reference and registers
the same ACTION_TABLE on every spawned machine).

## state starting [initial]
## state compressed
## state regrown
## state projected
## state finetuned
## state done [final]
## state failed [final]

## guards

| Name | Expression |
|------|------------|
| should_regrow | ctx.regrow_count > 0 |
| no_regrow_more_passes | ctx.regrow_count == 0 and ctx.inner_refine_idx < ctx.inner_refine_passes |
| no_regrow_done | ctx.regrow_count == 0 and ctx.inner_refine_idx >= ctx.inner_refine_passes |
| basis_loop_continue | ctx.inner_refine_idx < ctx.inner_refine_passes |
| basis_loop_done | ctx.inner_refine_idx >= ctx.inner_refine_passes |

## transitions

| Source | Event | Guard | Target | Action |
|--------|-------|-------|--------|--------|
| starting | start |  | compressed | compress_with_polygram |
| starting | error |  | failed | log_error |
| compressed | compress_done | should_regrow | regrown | adapt_and_regrow |
| compressed | compress_done | no_regrow_more_passes | compressed | compress_with_polygram |
| compressed | compress_done | no_regrow_done | projected | project_to_subspace |
| compressed | error |  | failed | log_error |
| regrown | regrowth_done | basis_loop_continue | compressed | compress_with_polygram |
| regrown | regrowth_done | basis_loop_done | projected | project_to_subspace |
| regrown | error |  | failed | log_error |
| projected | projection_done |  | finetuned | fine_tune_model |
| projected | error |  | failed | log_error |
| finetuned | finetune_done |  | done |  |
| finetuned | error |  | failed | log_error |

## actions

| Name | Signature |
|------|-----------|
| compress_with_polygram | (ctx) -> Context |
| perform_regrowth | (ctx) -> Context |
| adapt_and_regrow | (ctx) -> Context |
| project_to_subspace | (ctx) -> Context |
| fine_tune_model | (ctx) -> Context |
| log_error | (ctx) -> Context |
