"""Basic version of GPT-2 with no fancy optimizations or architecture improvements.

Based on karpathy's implementation:
- https://github.com/karpathy/llm.c/blob/master/train_gpt2.py

Note, the naming of the weights in each module is done to match huggingface's GPT-2
model so we can easily load pretained weights
"""

import inspect
import math
import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint

from gollem.models.model import BaseLLM
from gollem.utils import print0


if TYPE_CHECKING:
    from gollem.models.llama3.config import Llama3Config


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials (cis) with given dims.

    This function calculates a frequency tensor with complex exponentials using the
    given dimension 'dim' and the end index 'end'. The 'theta' parameter scales the
    frequencies. The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies (i.e. max sequence length)
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """
    # freqs: (dim // 2,)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # t: (end,)
    t = torch.arange(end)
    # freqs: (end, dim // 2)
    freqs = torch.outer(t, freqs).float()
    # freqs_cis: (end, dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors
    using the provided frequency tensor 'freqs_cis'. The input tensors are reshaped as
    complex numbers, and the frequency tensor is reshaped for broadcasting
    compatibility. The resulting tensors contain rotary embeddings and are returned as
    real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    # xq: (B, T, n_head, d_head)
    # xk: (B, T, n_kv_head, d_head)
    # freqs_cis: (T, d_head)

    # reshape to complex numbers
    # TODO check if we need the .float() here
    # (B, T, n_head, d_head) -> (B, T, n_head, d_head // 2, 2)
    xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    # (B, T, n_kv_head, d_head) -> (B, T, n_kv_head, d_head // 2, 2)
    xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
    # convert to complex numbers
    xq_ = torch.view_as_complex(xq)
    xk_ = torch.view_as_complex(xk)

    # reshape for broadcasting
    ndim = xq.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (xq.shape[1], xq.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq.shape)]
    freqs_cis = freqs_cis.view(*shape)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Multi-head causal attention."""

    def __init__(self, cfg: "Llama3Config"):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.n_kv_head = cfg.n_kv_head
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head

        # query, key, value projections for all heads, batched together
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        # output projection
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model)
        # sets flag for init weight scaling
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1  # type: ignore

        self.wq = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        self.wk = nn.Linear(self.d_model, self.n_kv_head * self.d_head, bias=False)
        self.wv = nn.Linear(self.d_model, self.n_kv_head * self.d_head, bias=False)
        self.wo = nn.Linear(self.d_model, self.d_model, bias=False)

        # constant used for the attention masking
        self.register_buffer(
            "mask",
            torch.triu(torch.full((cfg.n_ctx, cfg.n_ctx), float("-inf"))).view(
                1, 1, cfg.n_ctx, cfg.n_ctx
            ),
        )

        # Caches for inference
        if cfg.use_kv_caching:
            # KV cache
            self.register_buffer(
                "cache_k",
                torch.zeros(
                    (cfg.max_sample_batch_size, cfg.n_kv_head, cfg.n_ctx, self.d_head)
                ),
            )
            self.register_buffer(
                "cache_v",
                torch.zeros(
                    (cfg.max_sample_batch_size, cfg.n_kv_head, cfg.n_ctx, self.d_head)
                ),
            )
            self.cache_x = None
        else:
            self.register_buffer(
                "cache_x",
                torch.zeros((cfg.max_sample_batch_size, cfg.n_ctx, self.d_model)),
            )
            self.cache_k = None
            self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # input is the normalized residual from the previous layer
        # x: (batch, T, d_model)
        B, T, _ = x.size()

        # compute query, key, value for all heads
        # (B, T, d_model) -> (B, T, d_model)
        xq = self.wq(x)
        # (B, T, d_model) -> (B, T, n_kv_head * d_head)
        xk = self.wk(x)
        xv = self.wv(x)

        # reshape each so heads are in separate dimensions and swap axes
        # (B, T, d_model) -> (B, n_head, T, d_head)
        # TODO transpose now or not? (i.e. (B T n_head d_head) or (B n_head T d_head))
        xq = xq.view(B, T, self.n_head, self.d_head)
        # (B, T, d_model) -> (B, n_kv_head, T, d_head)
        xk = xk.view(B, T, self.n_kv_head, self.d_head)
        xv = xv.view(B, T, self.n_kv_head, self.d_head)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # compute and rescale attention scores
        # attn_scores: (B, n_head, T, T)
        if self.cfg.flash:
            # flash attention
            # handles default scaling of 1/sqrt(d_head)
            z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn_scores = q @ k.transpose(-2, -1)
            # rescale
            attn_scores = attn_scores / math.sqrt(self.d_head)
            # apply causal mask
            attn_scores = attn_scores + self.mask[:, :, :T, :T]
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

    @torch.inference_mode()
    def sample(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        # input is the normalized residual from the previous layer
        # x: (batch, N, d_model)
        #   - where N is the number of new tokens since last sample, not
        #     the total sequence length T
        # start_pos: int, position 0 <= p <= n_ctx of start of x in the context
        B, N, _ = x.size()
        T = start_pos + N  # total sequence length
        assert (
            self.cfg.n_ctx >= T
        ), f"Cannot sample beyond context length: {T} > {self.cfg.n_ctx}"

        if not self.cfg.use_kv_caching:
            assert (
                self.cache_x is not None
            ), "Cache is not None but use_kv_caching is False"
            self.cache_x[:B, start_pos:T] = x
            # x: (B, T, d_model)
            x = self.cache_x[:B, :T]
            # compute query, key, value for all heads for new positions
            # (B, N, d_model) -> (B, N, 3 * d_model)
            qkv = self.c_attn(x)
            # (B, N, 3 * d_model) -> (B, N, d_model), (B, N, d_model), (B, N, d_model)
            q, k, v = qkv.split(self.d_model, dim=2)
            # reshape each so heads are in separate dimensions and swap axes
            # (B, N, d_model) -> (B, n_head, N, d_head)
            k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
            # small optimization, we only need the last N positions of Q,
            # since we are sampling the next token
            q = q[:, :, start_pos:T, :]

        if self.cfg.use_kv_caching:
            assert (
                self.cache_k is not None
            ), "Cache is not None but use_kv_caching is True"
            assert (
                self.cache_v is not None
            ), "Cache is not None but use_kv_caching is True"
            # compute query, key, value for all heads for new positions
            # (B, N, d_model) -> (B, N, 3 * d_model)
            qkv = self.c_attn(x)
            # (B, N, 3 * d_model) -> (B, N, d_model), (B, N, d_model), (B, N, d_model)
            q, k, v = qkv.split(self.d_model, dim=2)

            # reshape each so heads are in separate dimensions and swap axes
            # (B, N, d_model) -> (B, n_head, N, d_head)
            k = k.view(B, N, self.n_head, self.d_head).transpose(1, 2)
            q = q.view(B, N, self.n_head, self.d_head).transpose(1, 2)
            v = v.view(B, N, self.n_head, self.d_head).transpose(1, 2)
            # update cache with KV values for new positions
            self.cache_k[:B, :, start_pos:T] = k
            self.cache_v[:B, :, start_pos:T] = v
            # get cached K, V values for all positions up the current position
            # (B, n_head, T, d_head)
            k = self.cache_k[:B, :, :T]
            v = self.cache_v[:B, :, :T]

        # truncate mask for the current sequence positions
        # mask: (1, 1, N, T)
        mask = self.mask[:, :, start_pos:T, :T]

        # compute and rescale attention scores
        if self.cfg.flash:
            # flash attention
            # handles default scaling of 1/sqrt(d_head)
            # z: (B, n_head, N, d_head)
            z = F.scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask=mask)
        else:
            # attn_scores: (B, n_head, N, T)
            attn_scores = q @ k.transpose(-2, -1)
            # rescale
            attn_scores = attn_scores / math.sqrt(self.d_head)
            # apply causal mask
            attn_scores = attn_scores + mask
            # softmax to generate attn patterns
            attn = attn_scores.softmax(dim=-1)
            # (B, n_head, N, T) @ (B, n_head, T, d_head) -> (B, n_head, N, d_head)
            z = attn @ v

        # re-assemble all head outputs side-by-side
        # (B, n_head, N, d_head) -> (B, N, d_model)
        z = z.transpose(1, 2).contiguous().view(B, N, self.d_model)

        # project to output: (B, N, d_model) -> (batch, N, d_model)
        out = self.c_proj(z)
        return out


class MLP(nn.Module):
    """MLP layer with SwiGLU activation function.

    Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """

    def __init__(self, cfg: "Llama3Config"):
        super().__init__()
        self.cfg = cfg
        self.w1 = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.silu = nn.SiLU()
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1  # type: ignore

    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """RMSNorm normalization layer.

    Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """

    def __init__(self, cfg: "Llama3Config"):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.ones(cfg.d_model))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg.rmsnorm_eps)
        return self.weight * norm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: "Llama3Config"):
        super().__init__()
        self.ln_1 = RMSNorm(cfg)
        self.attn = Attention(cfg)
        self.ln_2 = RMSNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def sample(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        x = x + self.attn.sample(self.ln_1(x), start_pos)
        x = x + self.mlp(self.ln_2(x))
        return x


class Llama3(BaseLLM["Llama3Config"]):
    """Llama-3 model."""

    def __init__(self, cfg: "Llama3Config"):
        super().__init__(cfg)

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(cfg.vocab_size, cfg.d_model),
                "h": nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)]),
                "ln_f": RMSNorm(cfg),
            }
        )
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # TODO check if this is correct for Llama-3
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = (
                0.02
                if not hasattr(module, "LLMC_RESIDUAL_SCALE_FLAG")
                else 0.02 / math.sqrt(2 * self.cfg.n_layer)
            )
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
            if self.cfg.activation_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
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

    @torch.inference_mode()
    def sample(
        self,
        tokens: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        """Sample from the model."""
        device = tokens.device
        _, N = tokens.size()
        T = start_pos + N
        assert (
            self.cfg.n_ctx >= T
        ), f"Cannot sample sequence of length {T}, ctx size is only {self.cfg.n_ctx}"

        # generate embedding
        # token embeddings of shape (B, N, d_model)
        tok_emb = self.transformer.wte(tokens)
        # position embeddings of shape (N, d_model)
        pos = torch.arange(start_pos, T, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        # combine: (B, N, d_model)
        x = tok_emb + pos_emb

        # forward thru blocks: x = (B, N, d_model)
        for block in self.transformer.h:
            if self.cfg.activation_checkpointing:
                x = checkpoint(block.sample, x, start_pos, use_reentrant=False)
            else:
                x = block.sample(x, start_pos)
        x = self.transformer.ln_f(x)

        # inference-time mini-optimization: only forward unembed on final position
        # note: using list [-1] to preserve the time dim
        logits = self.lm_head(x[:, [-1], :])

        return logits

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
        print0(
            f"num decayed parameter tensors: "
            f"{len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print0(
            f"num non-decayed parameter tensors: "
            f"{len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # Create AdamW optimizer and use the fused version if it is available
        use_fused = False
        if self.cfg.fused_adamw:
            fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == "cuda"

        ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
        if self.cfg.zero_optimizer and ddp:
            print0(f"Using ZeroRedundancyOptimizer with fused={use_fused}")
            optimizer = ZeroRedundancyOptimizer(
                **optim_groups[0],
                optimizer_class=torch.optim.AdamW,
                lr=self.cfg.learning_rate,
                betas=self.cfg.betas,
                fused=use_fused,
            )
            optimizer.add_param_group(optim_groups[1])
        else:
            print0(f"Using regular AdamW with fused={use_fused}")
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.cfg.learning_rate,
                betas=self.cfg.betas,
                fused=use_fused,
            )
        return optimizer

    @classmethod
    def from_pretrained(cls, config: "Llama3Config") -> "Llama3":
        """Loads pretrained Llama-3 model weights from huggingface"""
        raise NotImplementedError("Not implemented")
