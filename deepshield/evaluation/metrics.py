"""Frozen metric backend for identity, edit fidelity, and visual cost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from PIL import Image


@dataclass(frozen=True)
class MetricScores:
    identity_similarity: Optional[float]
    clip_edit_fidelity: float
    protection_lpips: float


class MetricBackend(Protocol):
    def score(
        self,
        identity_reference: Image.Image,
        original: Image.Image,
        protected: Image.Image,
        attacked_protected: Image.Image,
        prompt: str,
    ) -> MetricScores: ...


class FrozenMetricBackend:
    """Pinned FaceNet, CLIP, and LPIPS metrics loaded only on GPU runs."""

    def __init__(
        self,
        device: str = "cuda",
        clip_model_id: str = "openai/clip-vit-base-patch32",
    ) -> None:
        try:
            import lpips
            import torch
            from facenet_pytorch import InceptionResnetV1, MTCNN
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Metric dependencies are missing; install "
                "requirements-evaluation.txt in the Python 3.10 GPU environment"
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.facenet = InceptionResnetV1(pretrained="vggface2").to(self.device).eval()
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()

    def _identity(self, reference: Image.Image, candidate: Image.Image) -> Optional[float]:
        torch = self.torch
        ref_faces = self.mtcnn(reference.convert("RGB"))
        out_faces = self.mtcnn(candidate.convert("RGB"))
        if ref_faces is None or out_faces is None:
            return None
        if ref_faces.ndim != 4 or out_faces.ndim != 4:
            return None
        if ref_faces.shape[0] != 1 or out_faces.shape[0] != 1:
            return None
        with torch.inference_mode():
            ref = self.facenet(ref_faces.to(self.device))
            out = self.facenet(out_faces.to(self.device))
            return float(torch.nn.functional.cosine_similarity(ref, out).item())

    def _clip_fidelity(self, image: Image.Image, prompt: str) -> float:
        torch = self.torch
        inputs = self.clip_processor(
            text=[prompt], images=[image.convert("RGB")], return_tensors="pt", padding=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.clip_model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            return float(
                torch.nn.functional.cosine_similarity(image_features, text_features).item()
            )

    def _lpips(self, original: Image.Image, protected: Image.Image) -> float:
        import numpy as np

        torch = self.torch

        def tensor(image: Image.Image):
            array = np.asarray(image.convert("RGB").resize((256, 256)), dtype="float32")
            value = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
            return value.to(self.device)

        with torch.inference_mode():
            return float(self.lpips(tensor(original), tensor(protected)).item())

    def score(
        self,
        identity_reference: Image.Image,
        original: Image.Image,
        protected: Image.Image,
        attacked_protected: Image.Image,
        prompt: str,
    ) -> MetricScores:
        return MetricScores(
            identity_similarity=self._identity(identity_reference, attacked_protected),
            clip_edit_fidelity=self._clip_fidelity(attacked_protected, prompt),
            protection_lpips=self._lpips(original, protected),
        )
