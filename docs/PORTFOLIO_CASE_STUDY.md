# DeepShield: a bounded portfolio case study

## The question

Can a perturbation created against known diffusion components continue to
disrupt an image-editing pipeline after realistic preprocessing and a change of
model—and what visible cost does that perturbation impose?

This is more useful and falsifiable than claiming an image can be permanently
“AI-proof.” A sufficiently changed or unknown editor may defeat a fixed
perturbation.

## Ali Hasan's role

DeepShield began as a group extension of BlurGuard. According to the preserved
Git history, Ali authored the initial enhanced protection pipeline and later
added momentum and translation-invariant gradients, WebP/color-jitter EOT,
pixel-space loss, and the original demonstration script. Former teammates added
API/deployment work, ensemble/model support, stability fixes, and other
changes. BlurGuard's authorship and license remain prominent.

For the portfolio checkpoint, Ali also owns the security audit, evidence
boundary, and reproducible evaluation framing. The whole repository is not
presented as solely authored work.

## What the system does

The protection engine optimizes a bounded image perturbation against configured
surrogate VAE/denoising objectives. It can sample JPEG, WebP, resize, and color
jitter transformations during optimization, use a model ensemble, and add
frequency, pixel, and LPIPS-related constraints.

The separate evaluation harness defines paired runs in a versioned manifest.
For a fixed source, mask, prompt, pipeline, JPEG round-trip, and seed, it records
outputs for both the original and protected image. It deliberately preserves
missing runs and 100% failure cells.

## November 2026 landscape and gap

This checkpoint reviewed current primary sources through November 7, 2026.
Recent work makes a universal defense claim especially untenable:

- *Is Perturbation-Based Image Protection Disruptive to Image Editing?* reports
  that several perturbation defenses often did not fully disrupt editing and
  could increase prompt alignment in its tested settings.
- *Towards Transferable Defense Against Malicious Image Edits* directly targets
  the acknowledged cross-model transfer problem with image- and text-level
  optimization.
- *DCT-Shield* evaluates a frequency-domain defense designed around JPEG, while
  *DIA* attacks the full inversion trajectory rather than only an encoder.
- *Purify Once, Edit Freely* evaluates model mismatch plus purification and
  reports a practical route for restoring editability across tested defenses.

Primary sources:

- https://arxiv.org/abs/2506.04394
- https://arxiv.org/abs/2512.14341
- https://openaccess.thecvf.com/content/ICCV2025/html/Bala_DCT-Shield_A_Robust_Frequency_Domain_Defense_against_Malicious_Image_Editing_ICCV_2025_paper.html
- https://openaccess.thecvf.com/content/ICCV2025/html/Hong_DIA_The_Adversarial_Exposure_of_Deterministic_Inversion_in_Diffusion_Models_ICCV_2025_paper.html
- https://arxiv.org/abs/2603.13028

The defensible open question is therefore narrower: under a fixed visual budget,
does DeepShield's implemented ensemble/EOT configuration change paired edit
success after specified preprocessing when the editor architecture is held out,
and does it do better than a matched BlurGuard baseline? The answer may be no.
No novelty claim is made until an exact protocol and closest-comparator audit
survive review.

## What was actually checked

- six local unit tests pass, including deterministic JPEG preprocessing,
  manifest validation, failure-visible matrices, and a synthetic fake-adapter
  end-to-end run that writes paired results;
- the planner validates the draft example manifest and reports six planned
  rows from explicit assumptions without loading a model;
- Python sources compile locally;
- Git history and the intended checkpoint files pass a redacted secret scan;
- the repository's ownership, attribution, and third-party license boundaries
  are documented.

These are software and reproducibility checks. They are not a measured defense
success rate.

## Result

The defensible result is an implemented protection prototype plus a testable
protocol that makes negative results visible. No GPU transfer experiment has
been run for this checkpoint, no metric thresholds are frozen, and no
effectiveness percentage is claimed.

## Failure modes and limits

- Proprietary editors hide weights, preprocessing, masks, and sometimes the
  base architecture.
- Cropping, screenshots, purification, or changed codecs can erase or alter a
  perturbation.
- A diverse surrogate ensemble may still fail to transfer.
- Stronger perturbations can create unacceptable visual cost.
- The current manifest uses placeholder thresholds and has no approved sample
  pack.
- Local baseline repositories have separate licenses and are not integrated.

## The next falsifiable contribution

Only if a later portfolio decision justifies the cost, freeze a rights-cleared
synthetic sample pack and metric thresholds, then compare DeepShield with
BlurGuard on the same paired edits across one matched and one architecturally
held-out licensed local pipeline. Apply fixed JPEG plus at least one additional
preprocessing condition, publish every failed/missing cell, and report visual
cost with uncertainty. The hypothesis is contradicted if DeepShield does not
reduce paired attack success versus BlurGuard at a matched visual budget on the
held-out pipeline.

Gates before execution are: current model/pipeline availability and license,
sample rights, a reviewed metric/threshold calibration plan, a frozen manifest,
and a separate compute cap. Paid proprietary APIs and broad baseline wiring are
out of scope.

## Why it belongs in a portfolio

DeepShield demonstrates adversarial optimization, ML systems engineering,
experiment design, security hygiene, and the discipline to separate implemented
capability from empirical proof. It should appear as a supporting technical
case study, not as a startup claim or a universal safety product.
