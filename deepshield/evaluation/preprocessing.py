"""Attack-side preprocessing that matches common upload pipelines."""

from __future__ import annotations

from io import BytesIO

from PIL import Image


def jpeg_reencode(
    image: Image.Image,
    quality: int,
    size: tuple[int, int],
) -> Image.Image:
    """Resize, strip metadata, and round-trip through baseline JPEG.

    This operation is intentionally non-differentiable: it models the editing
    service's upload path, not DeepShield's EOT approximation.
    """
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")
    rgb = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    rgb.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=False,
        progressive=False,
        subsampling=2,
    )
    buffer.seek(0)
    decoded = Image.open(buffer)
    decoded.load()
    return decoded.convert("RGB")
