"""RGB baseline variant. No model or data modifications needed."""


def update_config(cfg):
    """No config changes needed for standard RGB."""
    pass


def update_model(model, cfg):
    """No model changes needed for standard RGB."""
    return model


def get_mapper(cfg, is_train):
    """Use the default Mask2Former mapper."""
    return None
