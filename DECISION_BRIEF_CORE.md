# Decision brief: DeepShield core evaluation gate

**Date:** 2026-07-20
**Status (updated 2026-11-07):** executable CPU-tested harness implemented;
no paid GPU run started; checkpoint is intentionally no-spend

## Recommended call

Keep DeepShield as a **bounded supporting portfolio case study**. Checkpoint the
implemented engine, security work, and failure-visible evaluation design, but
do not spend on a GPU pilot or expand into a broad product build. A later pilot
is optional and requires a new evidence/budget decision.

## What exists now

- A strict versioned manifest for samples, protection variants, attack
  pipelines, seeds, JPEG preprocessing, and provisional metric thresholds.
- A deterministic, non-differentiable JPEG upload round-trip.
- A matrix renderer that preserves 100% attack-success cells and refuses to
  turn dry runs into evidence.
- A dependency-light planning CLI that reports run count, estimated GPU hours,
  and estimated cost from an explicit hourly price.
- A draft evaluation protocol and an honest newest-models/arms-race limits doc.
- Lazy Diffusers inpainting/img2img adapters that load one pipeline at a time.
- Frozen FaceNet identity, CLIP edit-fidelity, and LPIPS visual-cost backends.
- An append-only paired-run orchestrator and CLI with a draft-protocol guard.
- A manual AWS/Hugging Face/GPU checklist for Ali.
- Six passing unit tests, including a fake-adapter end-to-end paired run.

## Deferred GPU decision

The example manifest contains 1 sample × 1 protection variant × 2 pipelines ×
3 seeds = **6 paired experimental rows**. At a provisional two minutes per row,
that is 0.2 GPU-hours, excluding model downloads, protection generation, and
metric loading. That number is planning arithmetic, not a benchmark.

If a later checkpoint approves an experiment, the earlier safe upper bound was:

- one 24 GB NVIDIA GPU (the existing deployment class is AWS `g5.xlarge`);
- hard cap: **2 instance-hours**;
- hard dollar cap: **Ali chooses after checking the live AWS console price**;
- stop after the first SD1.5 and SDXL paired pilot if either pipeline does not
  fit, metrics produce excessive `NA` values, or runtime projects beyond the
  cap.

AWS documents that G5 uses an NVIDIA A10G with 24 GB per GPU. Hugging Face
documents model CPU offload for SDXL/Flux memory pressure, but warns that
offloading can be substantially slower. The full Flux-inclusive matrix must
therefore be estimated from the pilot rather than guessed.

## Current gates

1. **Track:** bounded supporting case study; startup claims remain out of scope.
2. **Pilot spend:** not approved or needed for this checkpoint.
3. **Evaluation subjects:** no sample pack is approved. A future pilot should
   use synthetic or explicitly licensed subjects and requires a fresh gate.

## Not complete

- No stock/synthetic sample pack has been approved or generated.
- No GPU experiment, measured result, protection matrix, or interactive result
  demo exists yet.
- Vendored baseline adapters are not wired in.

The old operational plan is preserved in `docs/MANUAL_GPU_PILOT.md` but is
paused and should not be executed under this checkpoint.

These are next implementation steps, not claims of completion.
