"""DeepShield: Enhanced adversarial image protection against AI nudifiers."""

from .protect import ENSEMBLE_PRESETS, ProtectionConfig, protect_image

__all__ = ["protect_image", "ProtectionConfig", "ENSEMBLE_PRESETS"]
__version__ = "0.3.0"
