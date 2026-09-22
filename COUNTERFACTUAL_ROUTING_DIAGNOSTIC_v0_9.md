# Counterfactual routing diagnostic v0.9

## Decision

**NO-GO for the current hand-written routing-superiority claim.** The routing
rule is retained only as a documented diagnostic baseline. It must not be
presented as a validated contribution or tuned on these stream outcomes.

This is an engineering diagnostic, not a paper result. It uses the compact
character-fusion feasibility host and the already inspected public stream.

## Question and leakage control

For every transition from chunk `t` to chunk `t+1`, five actions branch from
the identical pre-update model state:

- no update;
- heads only;
- gate and heads;
- gate, heads, BiGRU, and TextCNN;
- the preceding modules plus the top encoder block.

Each branch updates only on labeled chunk `t` and is evaluated on chunk
`t+1`. The oracle is computed afterward from the labels of `t+1`; it is used
only to measure diagnostic regret and never makes an online decision. After
the branches, a fixed medium update advances the common anchor, so a routing
choice cannot contaminate a later branch point. The sentinel, reference, and
null calibration remain frozen.

## Full-stream result

The evaluation covers 86 transitions for each of seeds 42, 52, and 62 (258
paired transition evaluations). Values below are means across the three
seeds.

| Policy | Oracle-action accuracy | Mean one-step benefit | Mean regret | Positive-benefit rate | Harm rate |
|---|---:|---:|---:|---:|---:|
| Current routing | 0.2558 | 0.000178 | 0.014800 | 0.1008 | 0.0930 |
| Severity only | 0.2597 | 0.000356 | 0.014623 | 0.1279 | 0.0969 |
| Route shuffle | 0.3062 | 0.001495 | 0.013483 | 0.1395 | 0.0775 |

Paired routing-minus-control differences by seed:

| Control | Seed | Benefit difference | Regret difference |
|---|---:|---:|---:|
| Severity only | 42 | -0.000216 | +0.000216 |
| Severity only | 52 | -0.000097 | +0.000097 |
| Severity only | 62 | -0.000219 | +0.000219 |
| Route shuffle | 42 | -0.001746 | +0.001746 |
| Route shuffle | 52 | +0.000183 | -0.000183 |
| Route shuffle | 62 | -0.002388 | +0.002388 |

Lower regret is better. Routing has lower benefit and higher regret than
severity-only in all three seeds. It loses to route shuffle on benefit and
regret in two of three seeds. Its decisions equal the severity-only action on
93.0% of transitions, so the current signature logic rarely creates a
meaningfully distinct policy.

The oracle mean one-step benefit is 0.014979, while the current routing mean
benefit is 0.000178. Thus useful action heterogeneity exists in this diagnostic,
but the hand-written signature rule does not identify it. Oracle action ties
are resolved by fixed action order, so regret—not exact action accuracy—is the
primary diagnostic measure.

## Consequence for the manuscript

1. Do not claim that the current tri-view signature router improves predictive
   performance, selects the correct module, or beats severity-only adaptation.
2. Keep the tri-view monitor as an independently testable measurement layer;
   do not equate detection quality with adaptation quality.
3. Do not add explainability, memory, or another component to rescue this
   result.
4. Any learned replacement router requires a new, predeclared temporal
   development/holdout protocol. It must be fitted only on development
   transitions, frozen, and evaluated once on untouched future days against
   severity-only and cost-matched shuffle controls.

## Reproduction

```bash
PYTHONPATH=src python src/counterfactual_routing_diagnostic.py \
  --seed 42 --outdir outputs/counterfactual_seed_42
PYTHONPATH=src python src/counterfactual_routing_diagnostic.py \
  --seed 52 --outdir outputs/counterfactual_seed_52
PYTHONPATH=src python src/counterfactual_routing_diagnostic.py \
  --seed 62 --outdir outputs/counterfactual_seed_62
```

The output metadata labels every run `diagnostic_not_paper_result`.
