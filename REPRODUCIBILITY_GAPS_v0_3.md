# Reproducibility gaps affecting the source-model reconstruction

These gaps matter because the new paper must not claim an exact reproduction when the source does not provide enough information.

## 1. Exact lightweight encoder checkpoint is not identified
The paper names **RoBERTa-wwm-ext-small** and says its parameter count/layer count are reduced, but it does not provide an exact public checkpoint identifier or full layer configuration.

The public HFL `chinese-roberta-wwm-ext` checkpoint is reproducible but is a 12-layer, 768-hidden model, so it cannot silently be treated as the unspecified small variant.

**Decision:** use the neutral term `fusion host`. Log the exact encoder checkpoint used. Treat a reproducible small Chinese encoder as a sensitivity experiment, not as the source study's exact model.

## 2. Financial incremental pre-training is underspecified
The paper describes incremental masked-language-model pre-training on financial news, announcements, and market comments, but does not give enough information to recreate the exact corpus, schedule, number of steps/epochs, sampling, or checkpoint.

**Decision:** do not fabricate this stage. The primary reproducible study uses a named public encoder; optional domain-adaptive pre-training must use a fully specified public corpus and a separate ablation.

## 3. Streaming/incremental update policy is underspecified
The paper states that an incremental updating strategy is introduced for streaming data, but the exact trigger, chunk definition, parameters updated, label timing, and update schedule are not specified in sufficient algorithmic detail.

**Decision:** this is precisely the methodological gap addressed by TRIAS. Do not present the source paper's unspecified update mechanism as an implemented baseline unless an explicit defensible surrogate policy is defined.

## 4. Risk calibration is not fully reproducible
The paper describes a validation-distribution-derived threshold and empirical adjustment according to risk preference, but not a complete fixed algorithm that would allow exact independent reproduction.

**Decision:** Data 1 experiments focus on sentiment. Any risk calibration experiment must have explicit formulas, thresholds, and label provenance.

## 5. Data 2 is self-collected and not publicly linked
Risk level/type labels are described for Data 2, but the row-level dataset is not made available in the manuscript.

**Decision:** no Data-2 numeric result will be represented as independently reproduced.

## 6. Primary evaluation is not a streaming-valid protocol
The source paper reports an 8:1:1 train/validation/test split and repeated random partition/cross-validation-style evaluation. This does not establish performance under chronological non-stationarity.

**Decision:** chronological prequential evaluation is the primary protocol for TRIAS.
