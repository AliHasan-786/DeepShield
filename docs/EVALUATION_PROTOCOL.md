# DeepShield transfer-evaluation protocol

Status: **draft — no result produced under this protocol is publishable yet**

## Falsifiable question

For the same source image, edit prompt, mask, pipeline, preprocessing, and
random seed, does an edit that succeeds on the unprotected image still succeed
on the protected image?

Each pipeline must receive a real JPEG round-trip before inference. The
canonical implementation is
`deepshield.evaluation.preprocessing.jpeg_reencode`; this is deliberately
non-differentiable and separate from the EOT approximation used to create the
protection.

## Unit of analysis

One row is:

`sample × protection variant × attack pipeline × seed`

The unprotected and protected attacks use the same prompt, mask, dimensions,
inference settings, and seed. A manifest records every planned row before GPU
execution. Missing and failed runs remain missing; they are not silently
replaced.

## Fixed metrics

1. **Identity similarity:** cosine similarity between a frozen face embedding
   for the identity reference and the attacked output. Runs with no detected
   face, multiple ambiguous faces, or a detector failure are `NA`, not zero.
   The embedding model and detector version must be added to the frozen
   manifest before execution.
2. **CLIP edit fidelity:** cosine similarity between the attacked output and
   the edit prompt using a frozen CLIP checkpoint recorded in the manifest.
3. **Visual cost:** LPIPS between the source and protected image, measured
   before the editing attack with the frozen LPIPS checkpoint.
4. **Attack success:** true only when both identity similarity and CLIP edit
   fidelity clear thresholds calibrated on an unprotected pilot set.

The numerical thresholds currently in `configs/evaluation.example.json` are
placeholders. The pilot establishes their provenance; the manifest then changes
from `draft` to `frozen` before the confirmatory run. Thresholds must not be
retuned after looking at protected-image outcomes.

## Matrix reporting

The primary matrix reports attack success rate by protection variant and
pipeline. Lower is better for DeepShield. A 100% cell is printed explicitly
and means the protection failed on every recorded run. Empty cells display
`—`; dry-run plans never appear as evidence.

Secondary tables report identity similarity, CLIP edit fidelity, LPIPS, runtime,
and errors. Results are stratified by perturbation budget and pipeline rather
than pooled into a single headline number.

## Baselines and attribution

PhotoGuard, Anti-DreamBooth, Mist, MMA-Diffusion, DiffAdvPerturbationBench, and
BlurGuard remain third-party baselines. Their adapters and outputs must retain
upstream names and licenses and must never be counted as Ali's original code.

## Human spot-check gate

The confirmatory sample is rendered as paired, blinded panels for an independent
model review. Reviewers answer only whether the requested edit is present,
whether identity is preserved, and whether obvious corruption is visible.
Agreement and disagreements are reported. This does not replace the fixed
metrics.
