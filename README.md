# DeepShield

DeepShield is a research prototype for adding bounded adversarial
perturbations to an image before it reaches a diffusion-based editing pipeline.
The implementation combines BlurGuard-style frequency regularization with
expectation over transformations (EOT), encoder objectives, and optional
denoising objectives.

> **Evidence status:** this repository contains an implemented prototype. It
> does not establish that protected images are impossible to edit, that the
> method transfers to a proprietary service, or that it survives arbitrary
> preprocessing. No defense-success percentage is claimed from this public
> source snapshot.

## The useful question

The project is best treated as an evaluation problem:

**Under a fixed visual budget, when does a protection transfer through realistic
preprocessing to a held-out editor, when does it fail, and what does it cost in
image quality?**

A fixed adversarial perturbation is not permanent or universal protection.
Model changes, compression, cropping, reconstruction, purification, and other
processing can weaken or remove it.

## What this source implements

- iterative perturbation optimization against diffusion components;
- JPEG-oriented EOT, resizing, blur, and related augmentations;
- encoder and optional denoising objectives;
- BlurGuard-derived frequency regularization;
- configurable pixel budgets and optimization steps;
- single-image and batch CLI entry points;
- a FastAPI worker with health and processing endpoints.

These are code capabilities, not measured robustness results. In particular,
sampling JPEG qualities during optimization is not proof that protection
survives JPEG or an unknown service.

## Quick start

Installing and running the protection engine may download model weights and may
require suitable GPU memory:

```bash
python -m pip install -r requirements_deepshield.txt

python run_protection.py \
  --input photo.jpg \
  --output photo_protected.png
```

Batch processing:

```bash
python run_protection.py \
  --input-dir ./photos/ \
  --output-dir ./protected/
```

Local API worker:

```bash
./scripts/run_api.sh
curl http://127.0.0.1:8000/health
```

The API is a prototype and should not be exposed as an unauthenticated public
service.

## Reviewed source checkpoint

The intentionally reviewed prototype checkpoint is commit
[`3b2703e49f8e9919b6c29543ce2de3f5b662dc97`](https://github.com/AliHasan-786/DeepShield/tree/3b2703e49f8e9919b6c29543ce2de3f5b662dc97).
It adds the security audit, explicit evidence boundaries, an offline-tested
paired-evaluation harness, and public case-study material without running a GPU
experiment.

- [Bounded portfolio case study](https://github.com/AliHasan-786/DeepShield/blob/3b2703e49f8e9919b6c29543ce2de3f5b662dc97/docs/PORTFOLIO_CASE_STUDY.md)
- [Evaluation protocol](https://github.com/AliHasan-786/DeepShield/blob/3b2703e49f8e9919b6c29543ce2de3f5b662dc97/docs/EVALUATION_PROTOCOL.md)
- [Security decision brief](https://github.com/AliHasan-786/DeepShield/blob/3b2703e49f8e9919b6c29543ce2de3f5b662dc97/DECISION_BRIEF.md)

At that reviewed revision, six deterministic offline harness tests passed. That
number describes software checks only; it is not an effectiveness metric. No
GPU transfer matrix, frozen thresholds, approved sample pack, or comparison
against a proprietary editor was produced.

## Authorship and attribution

DeepShield began as a group extension of
[BlurGuard](https://github.com/jsu-kim/BlurGuard). BlurGuard's original authors
and MIT notice remain credited in [LICENSE](LICENSE).

Git history records Ali Hasan as the author of the initial enhanced DeepShield
pipeline. The repository also contains meaningful work by the original
BlurGuard authors and former project collaborators, including API/deployment
work and later model/optimization changes. The project is not represented as a
solely authored codebase.

This tree also contains inherited datasets, papers, and third-party reference
implementations. They are not counted as Ali's original work, benchmark
evidence, or proof that their licenses permit reuse in another product. The
reviewed checkpoint records the current attribution and license boundaries in
[`THIRD_PARTY.md`](https://github.com/AliHasan-786/DeepShield/blob/3b2703e49f8e9919b6c29543ce2de3f5b662dc97/THIRD_PARTY.md).

## Limits

- Surrogate-model optimization does not prove black-box transfer.
- Stronger perturbations may create unacceptable visual degradation.
- LPIPS and embedding-based measures are proxies, not human guarantees.
- Proprietary pipelines can change without notice.
- Negative or null experimental outcomes must remain visible.
- Any future experiment needs rights-cleared images, fixed preprocessing,
  frozen thresholds, exact model revisions, and a separate compute cap.

## License

The tracked DeepShield source retains the repository's existing MIT license.
Third-party materials remain subject to their own notices and restrictions.
