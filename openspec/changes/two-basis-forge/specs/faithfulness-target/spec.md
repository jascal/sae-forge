# faithfulness-target Specification

## ADDED Requirements

### Requirement: Circuit-restricted faithfulness KL

The forge faithfulness report SHALL, when circuit-faithfulness is
requested, report `KL(host ‖ forged)` not only globally but also
restricted to a circuit-driven token mask and to that mask's
complement. v1 SHALL support at least the `induction_predictable`
mask (the next token equals what followed the current token's previous
in-context occurrence) and SHALL expose `in_context_repeat` as a second
mask.

This is an *additional* reported target. It SHALL NOT alter the global
KL computation, thresholds, or acceptance gates already defined for
single-basis forge; global KL remains the primary faithfulness target
with its existing contract intact. Circuit faithfulness SHALL default
off; when not requested, no circuit mask is computed and only the
existing global targets are reported.

The rationale pinned by this requirement: global KL is dominated by the
common, assertion-driven next-token mass and is nearly blind to circuit
breakage (induction-predictable tokens are a single-digit percentage of
tokens). A mechanism that targets circuit fidelity — such as
composition-subspace preserve — must be judged on the masked KL, not
only the aggregate.

#### Scenario: masked and complement KL are both reported

- **GIVEN** a forge run requesting circuit faithfulness with the `induction_predictable` mask
- **WHEN** the faithfulness report is produced
- **THEN** it contains a global KL, a masked KL, a complement KL, and the masked-token count
- **AND** the global KL value and its acceptance gate are unchanged from the single-basis report

#### Scenario: circuit_kl is zero for an identical forge

- **GIVEN** forged logits equal to host logits
- **WHEN** `circuit_kl(host_logits, forged_logits, mask=induction_predictable)` is computed
- **THEN** both `masked_kl` and `complement_kl` are `0.0`

#### Scenario: circuit faithfulness defaults off

- **GIVEN** a forge run that does not request circuit faithfulness
- **WHEN** the report is produced
- **THEN** only the existing global KL targets are reported
- **AND** no circuit mask is computed
