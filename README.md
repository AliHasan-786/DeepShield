# DeepShield

DeepShield is a research prototype for adding adversarial perturbations to an
image before it reaches an AI editing pipeline. It combines multi-model VAE
objectives, expectation over transformations (EOT), frequency regularization,
and optional perceptual constraints.

The important question is not whether adversarial noise can make an image
universally “AI-proof.” It cannot make that guarantee. The useful question is
where a fixed protection transfers, where it fails, and what visual cost it
imposes. This repository includes the protection engine and an offline-tested
evaluation harness for running that paired experiment.

> **Evidence boundary:** the engine and evaluation plumbing are implemented;
> six local harness tests pass. No current GPU transfer matrix, effectiveness
> rate, or claim of protection against a proprietary service is published.

## What is implemented

- iterative adversarial optimization against one or more diffusion VAEs;
- JPEG, WebP, resizing, and color-jitter EOT transformations;
- momentum and translation-invariant gradient options;
- encoder, pixel-space, frequency, denoising, and optional LPIPS objectives;
- CLI and FastAPI entry points;
- a versioned manifest for paired protected/unprotected evaluation;
- deterministic JPEG upload preprocessing, lazy Diffusers adapters, fixed
  metric interfaces, append-only results, and explicit failure cells;
- a planner that calculates run count and estimated compute from user-supplied
  timing and price assumptions without loading a model.

## What is not established

- that DeepShield stops arbitrary, proprietary, or future image editors;
- that perturbations survive every crop, resize, compression, purification, or
  screenshot path;
- a measured comparison against BlurGuard or the third-party baselines stored
  locally for research;
- calibrated metric thresholds, a frozen sample set, or confirmatory results;
- a hosted public demo or production service.

The proposed “What Survives?” comparison is a separate future experiment. It
has not been run and is not part of this checkpoint.

## Architecture

For each optimization step, DeepShield applies sampled transformations to the
candidate image, evaluates configured surrogate objectives, aggregates the
gradients, and projects the perturbation back into the selected pixel budget.
Model diversity is intended to test transfer; it is not evidence that transfer
occurred.

The evaluation harness then pairs the same source, mask, prompt, pipeline,
preprocessing, and random seed for unprotected and protected inputs. Empty,
failed, and 100% attack-success cells remain visible rather than being silently
discarded.

## Quick start

Install the protection dependencies:

```bash
python -m pip install -r requirements_deepshield.txt
```

Run a local protection job (model downloads and suitable hardware may be
required):

```bash
python run_protection.py \
  --input photo.jpg \
  --output photo_protected.png \
  --preset demo
```

Start the API:

```bash
./scripts/run_api.sh
```

The API exposes `GET /health` and `POST /process`. It is a prototype and should
not be exposed as an unauthenticated public service.

## Offline evaluation checks

The following checkpoint checks require no GPU and download no models:

```bash
python -m unittest discover -s tests -v
python scripts/plan_transfer_eval.py \
  configs/evaluation.example.json \
  --seconds-per-run 120 \
  --gpu-hourly-usd 0 \
  --json
```

The example manifest is deliberately `draft`. The runner refuses to present a
draft protocol as evidence unless the calibration-only `--allow-draft` flag is
supplied. See [the evaluation protocol](docs/EVALUATION_PROTOCOL.md).

## Authorship and provenance

DeepShield began as a group extension of
[BlurGuard](https://github.com/jsu-kim/BlurGuard), whose original authors and
MIT notice remain credited in [LICENSE](LICENSE). Git history records Ali Hasan
as the author of the initial enhanced protection pipeline, later momentum/TI,
WebP/color-jitter and pixel-loss work, and the original demo script. Other
commits in this repository were authored by the BlurGuard team and former
project collaborators; this is not represented as a solely authored codebase.

Local checkouts of Anti-DreamBooth, PhotoGuard, Mist, MMA-Diffusion, and
DiffAdvPerturbationBench are research references only. They are intentionally
untracked, are not part of the DeepShield distribution, and must retain their
own upstream licenses. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Portfolio framing

The public-facing contribution is the implemented engine plus a reproducible,
failure-visible evaluation design—not a universal defense claim. The bounded
case-study draft is in [docs/PORTFOLIO_CASE_STUDY.md](docs/PORTFOLIO_CASE_STUDY.md).

## Security

Never commit credentials, private keys, `.env` files, or local databases. See
[SECURITY.md](SECURITY.md) and the redacted audit in
[DECISION_BRIEF.md](DECISION_BRIEF.md). No history rewrite or key rotation was
performed because the audit found no Git exposure evidence.

## License

The tracked repository is distributed under the existing MIT license. Keep the
copyright and permission notice in [LICENSE](LICENSE) with substantial copies.
