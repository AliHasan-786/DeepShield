# Current-transfer protocol preparation

**Review cutoff:** 2026-11-09

**Protocol:** `configs/transfer_protocol.2026-11-09.v1.json`

**Status:** draft, unrun, and blocked from execution

## Decision this work is meant to support

The useful decision is whether the current DeepShield approach has enough
residual value against a modern open editor to justify further engineering, or
whether it should be retired as an image-protection direction.

The proposed question is deliberately narrower than a robustness or novelty
claim:

> At a visual cost matched to BlurGuard on development images, does DeepShield
> reduce successful edits more than BlurGuard after fixed preprocessing when
> evaluated on a held-out FLUX.2 editor?

The answer may be no. The source checkpoint, validator, and dry-run plan are not
experimental evidence.

## Current primary-source landscape

The following records were reviewed as primary sources available through the
cutoff date. Descriptions below summarize what each source claims; DeepShield
has not reproduced those claims.

| Work | What it already contributes | Relevance to this protocol |
|---|---|---|
| [BlurGuard, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/29806b2e4dcb35e42abed21d43e2b7d8-Abstract-Conference.html) | Adaptive regional blur of protective noise; evaluates reversal/purification robustness and perceptual cost. Official code is [MIT](https://github.com/jsu-kim/BlurGuard) at reviewed commit `e1b2d5e03500320e4bad32a8ae04e6dd64eb0495`. | Required baseline and ancestry, not Ali's original work. The official workflow also needs SD 1.4, SAM, and separate purification environments; no DeepShield adapter is currently verified. |
| [Is Perturbation-Based Image Protection Disruptive to Image Editing?, 2025](https://arxiv.org/abs/2506.04394) | Reports settings where tested perturbation defenses did not fully disrupt editing and sometimes increased prompt alignment. | Makes favorable effects unsafe to assume and supports retaining adverse cells. |
| [DIA, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Hong_DIA_The_Adversarial_Exposure_of_Deterministic_Inversion_in_Diffusion_Models_ICCV_2025_paper.html) | Optimizes against an integrated DDIM inversion trajectory rather than only a VAE bottleneck. | DeepShield's VAE-centric ensemble does not cover this objective. |
| [DCT-Shield, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Bala_DCT-Shield_A_Robust_Frequency_Domain_Defense_against_Malicious_Image_Editing_ICCV_2025_paper.html) | Frequency-domain protection designed around JPEG and a protection/visual-quality tradeoff. | EOT over JPEG is not the only current JPEG-aware defense and cannot be called novel here. |
| [Purify Once, Edit Freely, 2026](https://arxiv.org/abs/2603.13028) | Evaluates purification under model mismatch and reports restored editability across tested defenses. | Requires a model-mismatch threat model and prevents treating preprocessing robustness as settled. |
| [VETO, July 2026](https://arxiv.org/abs/2607.27292) | Proposes a joint-attention-targeted defense for modern unified editors and VetoBench; compares Pareto frontiers over perturbation budgets and uses human validation of edit success. | This is the closest known work. It likely covers the broader scientific premise better than DeepShield. The remaining value is an implementation-specific obsolescence test, not a supported novelty claim. |

### What this study could add

If executed later, the study would provide a reproducible answer about this
specific DeepShield implementation, at its reviewed revision, relative to its
BlurGuard ancestor. It would add:

- a development-only rule for matching visual cost rather than comparing
  methods at equal epsilon;
- a held-out, reference-conditioned FLUX.2 editor absent from the implemented
  DeepShield surrogate list;
- fixed identity/no-lossy and JPEG-85 preprocessing paths;
- explicit failed, unavailable, missing, and unrun cells;
- an inspectable record that could justify retiring the approach after a
  negative result.

That is useful engineering evidence, but novelty is uncertain and probably
limited. VETO already studies modern unified editors, visual tradeoffs, and
human edit-success validation. No novelty language should be used without a
full method-level comparison, code review, and executed result.

## Official model and pipeline audit

Metadata was read from official model cards/APIs without downloading files.
Availability means that metadata was publicly reachable on the review date; it
does not mean execution is authorized or guaranteed.

| Role | Exact revision checked | License/access | Inference and implementation boundary |
|---|---|---|---|
| Development editor: [`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | `115134f363124c53c7d878647567d04daf26e41e` | Public and ungated in the HF API; CreativeML Open RAIL++-M. | Official card uses `StableDiffusionXLInpaintPipeline`, fp16, a mask, and 1024px inputs. The existing adapter has an inpainting path but has not been runtime-verified at this revision. |
| Held-out editor: [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | `e7b7dc27f91deacad38e78976d1f2b499d76a294` | Public and ungated in the HF API; Apache-2.0. The official card also states use restrictions/safety guidance that require review. | Official card reports a 4B BF16 model, ~13 GB VRAM, four-step example, image editing, and a 23.7 GB repository. Current DeepShield's generic adapter is not verified compatible with `Flux2KleinPipeline`; an adapter is a future gate, not part of this milestone. |
| FLUX.2 official inference repository | [`50fe5162777813d869182b139e83b10743caef15`](https://github.com/black-forest-labs/flux2/tree/50fe5162777813d869182b139e83b10743caef15) | Apache-2.0 repository metadata at review. | Provides official implementation context only. Nothing was cloned or executed. |

### Implemented DeepShield ensemble audit

The `nudifier-v2` preset requests older SD inpainting/VAEs plus FLUX.1-schnell
and SD 3.5. Current metadata checks expose reproducibility risks:

- `stable-diffusion-v1-5/stable-diffusion-inpainting` is public under
  CreativeML OpenRAIL-M at revision
  `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb`, but its card says it is a
  community mirror of a deprecated RunwayML repository.
- `stabilityai/stable-diffusion-2-inpainting` returned HTTP 401 without
  credentials. DeepShield remaps it to an unaffiliated community mirror; that
  substitution needs a separate license/provenance review before a frozen run.
- `stabilityai/sd-vae-ft-mse` is public MIT at revision
  `31f26fdeee1355a5c34592e401dd41e45d25a493`.
- `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` is public Open RAIL++ at
  revision `115134f...`.
- `black-forest-labs/FLUX.1-schnell` reports Apache-2.0 but automatic gating at
  revision `741f7c3ce8b383c54771c7003378a50191e9efe9`.
- `stabilityai/stable-diffusion-3.5-large` reports automatic gating and the
  Stability Community License at revision
  `ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f`.

The engine currently skips ensemble models that fail to load. That behavior is
appropriate for a demo but invalid for a frozen comparison unless the result
records the exact loaded set and aborts when it differs from the preregistered
set. No ensemble availability is claimed from this metadata-only review.

## Versioned protocol design

The machine-readable v1 protocol enforces:

- disjoint development and held-out editors;
- three development image slots and eight held-out slots, with no assets
  currently admitted;
- asset-specific rights, provenance, consent when applicable, and SHA-256;
- identical method pairing by image, prompt, seed, editor, and preprocessing;
- development-only visual-cost selection based on the BlurGuard epsilon-16
  median LPIPS reference, with PSNR and L-infinity also reported;
- one primary held-out DeepShield-minus-BlurGuard contrast;
- a blinded four-part edit-success rubric with at least three independent
  judgments, still unresolved and therefore blocking;
- image-cluster bootstrap uncertainty and an explicit exploratory multiplicity
  rule;
- categorical `unrun`, `failed`, `unavailable`, and `missing` states distinct
  from a measured numeric zero;
- stopping before held-out work if loading, pairing, missingness, reviewer,
  metric, or compute gates fail.

## Dry-run plan and compute proposal

The CPU planner produces these counts from the protocol:

| Stage | Editor calls | Protection jobs | Provisional GPU-hour arithmetic |
|---|---:|---:|---:|
| Feasibility only | 6 | 2 | 1.267 |
| Development calibration | 84 | 18 | 7.650 |
| Held-out pilot | 144 | 16 | 7.983 |
| Total | 234 | 36 | 16.900 |

The arithmetic assumes 60 seconds per editor call, 20 minutes per protection
job, and explicit setup allowances. Those are conservative planning inputs,
not measurements. No dollar estimate is emitted until a live hourly price is
supplied.

The smallest later gate proposed for review is feasibility only: no more than
2 GPU-hours, $20, or 30 GB of downloads, whichever comes first. This is not
authorization. Development and held-out execution require new approval after
measured feasibility timing. The full 16.9-hour arithmetic is specifically a
reason not to authorize the full plan as written.

Run the no-model validator:

```bash
python3 scripts/validate_transfer_protocol.py \
  configs/transfer_protocol.2026-11-09.v1.json \
  --json
```

It currently reports seven blockers: protocol not frozen, no rights-cleared
assets, editor adapter not verified, methods not verified, visual budget not
frozen, primary metric not ready, and compute not approved.

## Eventual visitor interaction

If real results ever exist, the meaningful interaction is a controlled visual
comparison. A visitor selects a measured image, editor, preprocessing path,
prompt, seed, and method, then sees the source, protection, postprocessed input,
edited output, blinded judgments, quality costs, exact revisions, and cell
state. All values come from saved result artifacts. The browser must not invent
effectiveness formulas, interpolate missing cells, or make live provider calls.

There is no frontend build in this milestone because there are no measured
paired outputs to inspect.

## Unresolved gates

1. Rights-cleared development and held-out images with provenance.
2. Review of FLUX.2 model-card terms and the exact experiment use.
3. A verified FLUX.2 adapter and exact deterministic preprocessing record.
4. A runnable, attributed BlurGuard adapter at the pinned revision.
5. A frozen, fully loaded DeepShield surrogate set; no silent skipping.
6. Human-review rubric, reviewer qualification/compensation, and adjudication.
7. Exact automated metric checkpoints, validity, weights, and digests.
8. Development visual-cost calibration followed by a frozen selection record.
9. Measured feasibility runtime, hardware selection, live price, and explicit
   cap approval.

Until all nine are resolved, every experimental cell remains `unrun`.
