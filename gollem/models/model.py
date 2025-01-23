from typing import Generic
from typing import Self
from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from gollem.models.config import ModelConfig


ModelConfigT = TypeVar("ModelConfigT", bound=ModelConfig)


class BaseLLM(nn.Module, Generic[ModelConfigT]):
    def __init__(self, cfg: ModelConfigT):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = True,
        inference: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass on token idxs, with optional loss computation.

        Returns logits and optional loss.
        """
        raise NotImplementedError()

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate sequence.

        Takes a conditioning sequence of token indices idx (LongTensor of shape (b,t))
        and completes the sequence max_new_tokens times, feeding the predictions back
        into the model each time.

        Most likely you'll want to make sure to be in model.eval() mode of operation
        for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            ctx = (
                tokens
                if tokens.size(1) <= self.cfg.n_ctx
                else tokens[:, -self.cfg.n_ctx :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(ctx)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            tokens = torch.cat((tokens, next_token), dim=1)

        return tokens

    def configure_optimizers(self, device_type: str) -> torch.optim.Optimizer:
        raise NotImplementedError()

    @classmethod
    def from_pretrained(cls, config: ModelConfigT) -> Self:
        raise NotImplementedError()

    def save_model(self, path: str):
        data = {
            "model_state_dict": self.state_dict(),
            "config": self.cfg,
        }
        torch.save(data, path)


def load_model(path: str, device: str | torch.device) -> BaseLLM:
    data = torch.load(path, map_location=device, weights_only=False)
    # Note we need to do it this way since the model may have been compiled
    # which results in a different state dict structure to the original model
    model = data["config"].get_model_and_optimizer(device)[0]
    model.load_state_dict(data["model_state_dict"])
    return model
