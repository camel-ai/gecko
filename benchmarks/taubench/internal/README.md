# tau2 Internal Runtime

This package is the in-repository tau2 subset used by Gecko/GATS.

Scope:

- Domains: `airline`, `retail`
- Data: `data/taubench/domains/{airline,retail}`
- Runtime: domain DB models, native tool semantics, environment replay, task/message models
- Evaluation: DB, action, communicate, and optional LLM-backed NL assertions

Non-goals for this migration slice:

- No external `tau2` package imports.
- No unrelated tau2 domains.
- No official tau2 CLI, registry, voice/audio stack, or orchestrator code.

Runner:

- `run_taubench.py` is the self-contained text-mode runner. It uses the
  internal loader, internal half-duplex orchestrator, internal LLM user
  simulator, internal tool-calling assistant, and internal evaluator.

Validation commands:

Start Gecko before the runner command.

```bash
python -m unittest tests.taubench.test_internal_runtime -v
python run_taubench.py --domain airline --task-ids 0 --num-trials 1
```
