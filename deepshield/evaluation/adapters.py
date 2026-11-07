"""Lazy Diffusers adapters for black-box editing pipelines."""

from __future__ import annotations

from typing import Protocol

from PIL import Image

from .schema import AttackPipeline


class AttackAdapter(Protocol):
    def run(
        self,
        spec: AttackPipeline,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image: ...

    def close(self) -> None: ...


class DiffusersAttackAdapter:
    """Load one manifest-pinned pipeline at a time to bound GPU memory."""

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float16",
        cpu_offload: bool = False,
    ) -> None:
        self.device = device
        self.dtype_name = dtype
        self.cpu_offload = cpu_offload
        self._loaded_id: str | None = None
        self._pipe = None

    def _load(self, spec: AttackPipeline):
        if self._loaded_id == spec.id and self._pipe is not None:
            return self._pipe
        self.close()
        try:
            import torch
            from diffusers import (
                AutoPipelineForImage2Image,
                AutoPipelineForInpainting,
            )
        except ImportError as exc:
            raise RuntimeError(
                "GPU evaluation dependencies are missing; install "
                "requirements-evaluation.txt in the Python 3.10 GPU environment"
            ) from exc

        dtype = getattr(torch, self.dtype_name)
        pipeline_class = (
            AutoPipelineForInpainting
            if spec.task == "inpainting"
            else AutoPipelineForImage2Image
        )
        pipe = pipeline_class.from_pretrained(
            spec.model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        if self.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        pipe.set_progress_bar_config(desc=f"attack:{spec.id}")
        self._pipe = pipe
        self._loaded_id = spec.id
        return pipe

    def run(
        self,
        spec: AttackPipeline,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        import torch

        pipe = self._load(spec)
        generator_device = "cpu" if self.cpu_offload else self.device
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        kwargs = {
            "prompt": prompt,
            "image": image,
            "num_inference_steps": spec.steps,
            "guidance_scale": spec.guidance_scale,
            "generator": generator,
            "height": spec.height,
            "width": spec.width,
        }
        if spec.task == "inpainting":
            kwargs["mask_image"] = mask
        else:
            kwargs["strength"] = spec.strength
        return pipe(**kwargs).images[0].convert("RGB")

    def close(self) -> None:
        self._pipe = None
        self._loaded_id = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
