import importlib
from dataclasses import asdict
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Callable
from typing import Self
from typing import Tuple

import tiktoken
import torch


if TYPE_CHECKING:
    from gollem.models.model import BaseLLM


@dataclass
class ModelConfig:
    n_ctx: int = 1024
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_mlp: int = 4 * 768
    vocab_size: int = 50257
    ln_bias: bool = False
    mlp_bias: bool = False
    share_embd_params: bool = True
    flash_attention: bool = True

    def get_tokenizer(self) -> tiktoken.Encoding:
        raise NotImplementedError()

    def get_model_and_optimizer(
        self, device: str
    ) -> Tuple["BaseLLM", torch.optim.Optimizer]:
        raise NotImplementedError()

    def get_lr_scheduler(self, num_iterations: int) -> Callable[[int], float]:
        raise NotImplementedError()

    @classmethod
    def override(cls, existing: Self, **kwargs) -> Self:
        base_cfg_kwargs = asdict(existing)
        base_cfg_kwargs.update(kwargs)
        return cls(**base_cfg_kwargs)


_registry = {
    "gpt2": ("gpt2.config", "MODEL_CONFIGS"),
}


def get_model_config(name: str, **kwargs) -> ModelConfig:
    """Get named model config.

    Any defaults will be overwridden by values in kwargs.
    """
    assert name in _registry

    model_config_file, model_config_var = _registry[name]

    model_config_module = importlib.import_module(f"gollem.models.{model_config_file}")
    model_config_registry = getattr(model_config_module, model_config_var)

    base_cfg = model_config_registry[name]
    if not kwargs:
        return base_cfg

    return base_cfg.override(base_cfg, **kwargs)
