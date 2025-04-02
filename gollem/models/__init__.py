import importlib

from gollem.models.config import ModelConfig


model_registry = {
    "llama3": ("llama3.config", "get_llama3_model_config"),
    "gpt2": ("gpt2.config", "get_gpt2_model_config"),
}


def get_model_config(name: str, **kwargs) -> ModelConfig:
    """Get named model config.

    Any defaults will be overwridden by values in kwargs.
    """
    for registry_name in model_registry:
        if name.startswith(registry_name):
            model_cfg_file, model_registry_func_name = model_registry[registry_name]
            break
    else:
        raise ValueError(
            f"Model {name} not found in model registry."
            f" Should start with one of: {', '.join(model_registry.keys())}"
        )

    model_cfg_module = importlib.import_module(f"gollem.models.{model_cfg_file}")
    model_registry_func = getattr(model_cfg_module, model_registry_func_name)
    model_cfg = model_registry_func(name)

    if kwargs:
        model_cfg = model_cfg.override(**kwargs)

    return model_cfg
