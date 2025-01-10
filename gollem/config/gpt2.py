"""Utility model config stuff."""

from dataclasses import asdict

from gollem.config.model import ModelConfig


MODEL_CONFIGS = {
    # 124M params
    "gpt2": ModelConfig(
        n_ctx=1024,
        n_layer=12,
        n_head=12,
        d_model=768,
        d_mlp=4 * 768,
        vocab_size=50257,
        ln_bias=True,
        mlp_bias=True,
        share_embd_params=True,
    ),
    # 350M params
    "gpt2-medium": ModelConfig(
        n_ctx=1024,
        n_layer=24,
        n_head=16,
        d_model=1024,
        d_mlp=4 * 1024,
        vocab_size=50257,
        ln_bias=True,
        mlp_bias=True,
        share_embd_params=True,
    ),
    # 774M params
    "gpt2-large": ModelConfig(
        n_ctx=1024,
        n_layer=36,
        n_head=20,
        d_model=1280,
        d_mlp=4 * 1280,
        vocab_size=50257,
        ln_bias=True,
        mlp_bias=True,
        share_embd_params=True,
    ),
    # 1558M params
    "gpt2-xl": ModelConfig(
        n_ctx=1024,
        n_layer=48,
        n_head=25,
        d_model=1600,
        d_mlp=4 * 1600,
        vocab_size=50257,
        ln_bias=True,
        mlp_bias=True,
        share_embd_params=True,
    ),
}


def get_model_config(name: str, **kwargs) -> ModelConfig:
    """Get named model config.

    Any defaults will be overwridden by values in kwargs.
    """
    assert name in MODEL_CONFIGS
    if not kwargs:
        return MODEL_CONFIGS[name]

    base_cfg_kwargs = asdict(MODEL_CONFIGS[name])
    base_cfg_kwargs.update(kwargs)
    return ModelConfig(**base_cfg_kwargs)
