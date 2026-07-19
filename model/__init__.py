"""Public model factory for the Hybrid-UniSE engineering study.

This file replaces the upstream UniSE/BiCodec factory. The public snapshot
contains only the paper-inspired Hybrid-UniSE path.
"""

from .hybrid_model import HybridUniSELightning


def build_model(config):
    if config.get("model_type") != "hybrid_unise":
        raise ValueError("This repository only supports model_type: hybrid_unise")
    return HybridUniSELightning(config)


# Kept as a small compatibility alias for the existing public scripts.
Model = build_model

__all__ = ["HybridUniSELightning", "Model", "build_model"]
