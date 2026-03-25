"""DeepShield: Enhanced adversarial image protection against AI nudifiers."""

from .protect import ProtectionConfig, protect_image

__all__ = ["protect_image", "ProtectionConfig"]
__version__ = "0.2.0"
