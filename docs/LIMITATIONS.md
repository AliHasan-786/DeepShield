# Limits of DeepShield transfer claims

DeepShield is an adversarial intervention in an arms race, not a permanent
guarantee. A result applies only to the exact protection version, perturbation
budget, preprocessing path, attack pipelines, checkpoints, prompts, masks, and
seeds recorded in the frozen evaluation manifest.

The surrogate ensemble does not prove transfer. A model can share a VAE family
and still respond differently after fine-tuning, preprocessing, quantization,
or safety layers. Models released after the evaluated ensemble are out of
scope until tested. Proprietary services can change without notice.

JPEG robustness is measured through an explicit re-encode, but real services
may resize, crop, denoise, sharpen, recolor, or transcode differently. Human
visual acceptability also varies; LPIPS is a proxy, not a guarantee that a
perturbation is unobjectionable.

Identity and CLIP metrics have known failure modes and demographic variation.
Detector failures and ambiguous faces are reported as missing rather than
converted into favorable scores. Stock or synthetic evaluation subjects do not
establish safety for every real person.

Negative cells stay in the published matrix. If an editing pipeline succeeds
on protected images, the correct conclusion is that DeepShield failed under
that condition—not that the pipeline or metric should be removed.
