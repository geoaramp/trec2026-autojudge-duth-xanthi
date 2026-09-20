# DUTH XANTHI at TREC 2026 AutoJudge

Submission of the DUTH XANTHI team to the TREC 2026 AutoJudge track.

## DUTH Quality Judge

`DuthQualityJudge` is a pointwise LLM-based evaluator for generated
research/report answers.

Each response is evaluated on a 0-4 scale considering relevance,
completeness, specificity, coherence, apparent factual reliability,
repetition, irrelevant material, and unsupported claims.

The score is normalized to [0,1] and reported as `DUTH_QUALITY`.

The implementation uses the AutoJudge `LlmConfigProtocol` and
`minima-llm`, so the LLM endpoint and model are supplied by the
execution environment.

## Files

Judge:
`judges/duth_quality/duth_quality.py`

Workflow:
`judges/duth_quality/workflow.yml`

## Team

DUTH XANTHI
Democritus University of Thrace
Greece
