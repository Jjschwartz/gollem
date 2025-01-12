import importlib

from gollem.models.config import ModelConfig


model_registry = {
    "gpt2": ("gpt2.config", "GPT2_CONFIG"),
    "gpt2-medium": ("gpt2.config", "GPT2_MEDIUM_CONFIG"),
    "gpt2-large": ("gpt2.config", "GPT2_LARGE_CONFIG"),
    "gpt2-xl": ("gpt2.config", "GPT2_XL_CONFIG"),
}


def get_model_config(name: str, **kwargs) -> ModelConfig:
    """Get named model config.

    Any defaults will be overwridden by values in kwargs.
    """
    assert name in model_registry

    model_cfg_file, model_cfg_var = model_registry[name]

    model_cfg_module = importlib.import_module(f"gollem.models.{model_cfg_file}")
    model_cfg = getattr(model_cfg_module, model_cfg_var)

    if not kwargs:
        return model_cfg

    return model_cfg.override(model_cfg, **kwargs)
