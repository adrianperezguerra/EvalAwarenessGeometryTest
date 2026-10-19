# Evaluation Awareness Geometry

Preliminary code for examining the geometry of evaluation-framed versus deployment-framed activation differences, and their overlap with matched neutral-control differences.

**Status: research scaffold, not a validated pipeline.** The synthetic `--mock` path checks the math and output plumbing only. It does not validate the hypothesis, prompt design, activation extraction, or causal interpretation.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m pip install torch transformers
```

Choose a PyTorch build appropriate for your hardware using the official PyTorch installation instructions.

## Smoke test

```bash
python eval_awareness_prelim.py --mock
pytest -q
```

Mock values are synthetic and must not be reported as experimental results.

## Prepare prompt pairs

Edit `data/pairs.example.json`, replacing every `REPLACE:` value with actual prompts. Each pair's `a` and `b` should express the same underlying task/content while varying only the intended framing. The JSON must contain non-empty `eval_deploy` and `control` arrays, each with string fields `a` and `b`.

To run the tiny built-in placeholders, opt in explicitly:

```bash
python eval_awareness_prelim.py --demo-prompts --model Qwen/Qwen2.5-0.5B-Instruct --layer 8
```

Demo results are not evidence for the research hypothesis. For an actual run:

```bash
python eval_awareness_prelim.py \\
  --pairs-json data/pairs.example.json \\
  --model Qwen/Qwen2.5-0.5B-Instruct \\
  --revision <model-commit-or-tag> \\
  --layer 8 --device cpu --pooling mean \\
  --output outputs/run.json
```

`--layer` indexes the Hugging Face `hidden_states` tuple: index 0 is the embedding output, index N is the output of decoder block N. Confirm this convention and module behavior for the chosen architecture. The code records the supplied model revision but does not force one; for reproducible research, pass a fixed commit/tag. `--pooling mean` averages token activations; `--pooling last` uses the last token activation. These choices are not interchangeable and should be pre-registered or justified.

## Metrics and interpretation

- `PR_*_uncentered`: participation ratio from the raw difference matrix, preserving the original scaffold's uncentered calculation.
- `PR_*_centered`: participation ratio after subtracting each cloud's mean. This measures variation around the mean and is a different diagnostic.
- `cosine_eval_control`: cosine similarity between unit-normalized mean-difference directions.
- `residual_norm_ratio`: norm remaining after projecting the evaluation direction away from the rank-one control direction, divided by the original norm.
- `residual_energy_fraction`: squared residual norm ratio. For normalized directions and one nonzero control direction, this is `1 - cosine_eval_control**2`, up to floating-point precision.
- `within_*_cosine_reference_mean`: descriptive mean pairwise cosine among individual difference vectors. These pairwise values are dependent and are **not** a formal inferential null, confidence interval, or p-value.

The rank-one projection removes only the component aligned with one mean control direction. It does not establish that the residual is evaluation-specific or causally meaningful. A stronger analysis should compare against a multi-dimensional control subspace (for example, a properly fit LEACE/linear erasure approach) and test behavior under ablation. Do not claim confound removal based on the residual norm alone.

## Known limitations / next steps

- Handles one evaluation/deployment contrast and one control contrast at a time; it does not aggregate multiple benchmarks or estimate uncertainty.
- No LEACE fit, permutation/bootstrap inference, confidence intervals, multiple-comparison correction, surface-feature classifier, or SVG/judge behavioral evaluation.
- The ablation hook is a mechanism stub only. You must choose the correct module, verify its output convention, test generation behavior, and compare against sham/random-direction controls before interpreting results.
- No batching, activation caching, automatic model/module introspection, or full run-manifest capture.
- Prompt matching and dataset quality dominate interpretation; placeholder examples are deliberately inadequate for research conclusions.
- Verify dataset/model licenses and access terms. Do not commit restricted data, credentials, or model weights. No project license has been selected yet; choose one before public release if appropriate.
