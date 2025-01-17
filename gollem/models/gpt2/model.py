"""Basic version of GPT-2 with no fancy optimizations or architecture improvements.

Based on karpathy's implementation:
- https://github.com/karpathy/llm.c/blob/master/train_gpt2.py

Note, the naming of the weights in each module is done to match huggingface's GPT-2
model so we can easily load pretained weights
"""

import inspect
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from gollem.models.config import ModelConfig
from gollem.models.model import BaseLLM


if TYPE_CHECKING:
    from gollem.models.gpt2.config import GPT2Config


class Attention(nn.Module):
    """Multi-head causal attention."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # query, key, value projections for all heads, batched together
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        # output projection
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model)
        # sets flag for init weight scaling
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1  # type: ignore

        self.d_model = cfg.d_model
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head

        # constant used for the attention masking
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx)).view(
                1, 1, cfg.n_ctx, cfg.n_ctx
            ),
        )

    def forward(self, x):
        # input is the normalized residual from the previous layer
        # x: (batch, T, d_model)
        B, T, _ = x.size()

        # compute query, key, value for all heads
        # (B, T, d_model) -> (B, T, 3 * d_model)
        qkv = self.c_attn(x)
        # (B, T, 3 * d_model) -> (B, T, d_model), (B, T, d_model), (B, T, d_model)
        q, k, v = qkv.split(self.d_model, dim=2)

        # reshape each so heads are in separate dimensions and swap axes
        # (B, T, d_model) -> (B, n_head, T, d_head)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # compute and rescale attention scores
        # attn_scores: (B, n_head, T, T)
        if self.cfg.flash_attention:
            # flash attention
            # handles default scaling of 1/sqrt(d_head)
            z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn_scores = q @ k.transpose(-2, -1)
            # rescale
            attn_scores = attn_scores / math.sqrt(self.d_head)
            # apply causal mask
            attn_scores = attn_scores.masked_fill(
                self.mask[:, :, :T, :T] == 0, float("-inf")
            )
            # softmax to generate attn patterns
            attn = attn_scores.softmax(dim=-1)
            # (B, n_head, T, T) @ (B, n_head, T, d_head) -> (B, n_head, T, d_head)
            z = attn @ v

        # re-assemble all head outputs side-by-side
        # (B, n_head, T, d_head) -> (B, T, d_model)
        z = z.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # project to output: (B, T, d_model) -> (batch, T, d_model)
        out = self.c_proj(z)
        return out


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.c_fc = nn.Linear(cfg.d_model, cfg.d_mlp)
        # Approximate version used in GPT-2
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1  # type: ignore

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(BaseLLM["GPT2Config"]):
    """GPT-2 model."""

    def __init__(self, cfg: "GPT2Config"):
        super().__init__(cfg)

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(cfg.vocab_size, cfg.d_model),
                "wpe": nn.Embedding(cfg.n_ctx, cfg.d_model),
                "h": nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)]),
                "ln_f": nn.LayerNorm(cfg.d_model),
            }
        )
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.share_embd_params:
            # don't init this one, we will tie weights
            self.lm_head.LLMC_SKIP_INIT = 1  # type: ignore
            # https://paperswithcode.com/method/weight-tying
            self.transformer.wte.weight = self.lm_head.weight

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = (
                0.02
                if not hasattr(module, "LLMC_RESIDUAL_SCALE_FLAG")
                else 0.02 / math.sqrt(2 * self.cfg.n_layer)
            )
            # if using embed-unembed weight tying we want to skip initializing unembed,
            # since embed is already initialized down below during the Embedding init
            if not hasattr(module, "LLMC_SKIP_INIT"):
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=std, generator=self.init_rng
                )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02, generator=self.init_rng
            )

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = True,
        inference: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass on token idxs, with optional loss computation."""
        device = tokens.device
        _, T = tokens.size()
        assert (
            self.cfg.n_ctx >= T
        ), f"Cannot forward sequence of length {T}, ctx size is only {self.cfg.n_ctx}"

        # generate embedding
        # token embeddings of shape (B, T, d_model)
        tok_emb = self.transformer.wte(tokens)
        # position embeddings of shape (T, d_model)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        # combine: (B, T, d_model)
        x = tok_emb + pos_emb

        # forward thru blocks: x = (B, T, d_model)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        elif inference:
            # inference-time mini-optimization: only forward unembed on final position
            # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        else:
            logits = self.lm_head(x)
            loss = None

        # there are performance reasons why not returning logits is prudent,
        # if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, device_type: str) -> torch.optim.Optimizer:
        # start with all of the candidate parameters
        param_dict = dict(self.named_parameters())
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: "
            f"{len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: "
            f"{len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # Create AdamW optimizer and use the fused version if it is available
        use_fused = False
        if self.cfg.fused_adamw:
            fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == "cuda"

        print(f"Using regular AdamW with fused={use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.cfg.learning_rate,
            betas=self.cfg.betas,
            fused=use_fused,
        )
        return optimizer

    @classmethod
    def from_pretrained(cls, config: "GPT2Config") -> "GPT":
        """Loads pretrained GPT-2 model weights from huggingface"""
        model_type = config.model_name
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.mask")]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()  # type: ignore

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        # ignore these buffer params
        sd_keys_hf = [
            k
            for k in sd_keys_hf
            if not (k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"))
        ]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to
        # use a vanilla Linear this means that we have to transpose these weights when
        # we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
