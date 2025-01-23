from dataclasses import asdict
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Callable
from typing import Self
from typing import Tuple

import torch

from gollem.tokenizer import BaseTokenizer


if TYPE_CHECKING:
    from gollem.models.model import BaseLLM


@dataclass
class ModelConfig:
    model_name: str

    def get_tokenizer(self) -> BaseTokenizer:
        raise NotImplementedError()

    def get_model_and_optimizer(
        self, device: str | torch.device
    ) -> Tuple["BaseLLM", torch.optim.Optimizer]:
        raise NotImplementedError()

    def get_lr_scheduler(self, num_iterations: int) -> Callable[[int], float]:
        raise NotImplementedError()

    @classmethod
    def override(cls, existing: Self, **kwargs) -> Self:
        base_cfg_kwargs = asdict(existing)
        base_cfg_kwargs.update(kwargs)
        return cls(**base_cfg_kwargs)
