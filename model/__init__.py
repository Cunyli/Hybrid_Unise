def Model(config):
    if config.get("model_type") == "hybrid_unise":
        from .hybrid_model import HybridUniSELightning

        return HybridUniSELightning(config)
    from .model import Model as UniSEModel

    return UniSEModel(config)
