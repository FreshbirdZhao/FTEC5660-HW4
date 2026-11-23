# Experiment versions

The current Task 5 submission uses `single-factor-online/` (15 configurations, B0–V14), with matching retrieval-only results in `single-factor-offline/`. Reproduce it with the commands in the project README. `configurations.json` records every feature switch; adjacent rows differ in one switch. V14 is checked against the submitted retriever for all 60 public questions.

`online/` and `offline/` preserve the original seven-configuration exploratory experiment. They are superseded because some transitions grouped multiple changes. Their V1–V6 labels do not identify the same configurations as the current study. They are retained for provenance, not used as the submission's ablation table.

`generator-attempt.log` records an early network-failed run. The repository-root `results.csv` has since been regenerated successfully by the original official runner and contains 60 correct answers.
