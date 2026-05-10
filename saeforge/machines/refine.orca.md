# machine RefineMachine

The middle forge sub-machine: handles per-shard convergence. Loads
the SAE + corpus and scans activations on entry (`load_and_scan`,
the composed action that replaces v0.2's `loaded`/
`activations_scanned` state pair), then invokes `BasisMachine` for
the compress↔regrow loop. After basis returns, `evaluate_faithfulness`
runs and the refine loop either re-enters `refining` (same shard,
inner-refine-passes pattern) or exits to let `StreamMachine`
arbitrate between the next shard and termination.

Invoked by `StreamMachine.streaming`. The transitions out of
`evaluating` use guards over `ctx.advance_stream` and
`ctx.should_continue` — both set by `evaluate_faithfulness` and
visible because the orchestrator shares one ctx dict across all
machines in the hierarchy.

## state entering [initial]
## state refining

- invoke: BasisMachine
- on_done: -> basis_done

## state evaluating
## state exiting [final]
## state failed [final]

## guards

| Name | Expression |
|------|------------|
| refine_continue | ctx.advance_stream == false and ctx.should_continue == true |
| refine_exit | ctx.advance_stream == true or ctx.should_continue == false |

## transitions

| Source | Event | Guard | Target | Action |
|--------|-------|-------|--------|--------|
| entering | start |  | refining | load_and_scan |
| entering | error |  | failed | log_error |
| refining | basis_done |  | evaluating | evaluate_faithfulness |
| refining | error |  | failed | log_error |
| evaluating | eval_done | refine_continue | refining | rotate_for_next_iter |
| evaluating | eval_done | refine_exit | exiting |  |
| evaluating | error |  | failed | log_error |

## actions

| Name | Signature |
|------|-----------|
| load_and_scan | (ctx) -> Context |
| evaluate_faithfulness | (ctx) -> Context |
| rotate_for_next_iter | (ctx) -> Context |
| log_error | (ctx) -> Context |
