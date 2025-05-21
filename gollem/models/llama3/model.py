"""Llama3 implementation.

Based on: https://github.com/meta-llama/llama/blob/main/llama/model.py

Note, the naming of the weights in each module is done to match the weights in the
Llama-3 model so we can easily load pretained weights
"""

import inspect
import math
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint

from gollem.models.model import BaseLLM
from gollem.utils import get_base_dir_path
from gollem.utils import print0


if TYPE_CHECKING:
    from gollem.models.llama3.config import Llama3Config


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials (cis) with given dims.

    This function calculates a frequency tensor with complex exponentials using the
    given dimension 'dim' and the end index 'end'. The 'theta' parameter scales the
    frequencies. The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor (i.e. d_head).
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
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # freqs_cis: (end, dim // 2)
    # convert the freqs (i.e. angles) into polar coordinates represented as complex numbers
    # the distance of each point is 1, and the angle is the frequency
    # so each entry in freqs_cis is a complex number which encodes the freq (i.e. angle)
    # as a cartesian coordinate (real, imag)
    # Representing the frequency in this way makes it easy to apply the rotation matrix
    # to a vector by simply multiplying the vector by freqs_cis
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
    B, T, n_head, d_head = xq.shape
    assert freqs_cis.shape == (
        T,
        d_head // 2,
    ), f"freqs_cis.shape: {freqs_cis.shape} != (T, d_head // 2): {(T, d_head // 2)}"

    # reshape to complex numbers
    # (B, T, n_head, d_head) -> (B, T, n_head, d_head // 2, 2)
    xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    # (B, T, n_kv_head, d_head) -> (B, T, n_kv_head, d_head // 2, 2)
    xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
    # convert to complex numbers
    # xq: (B, T, n_head, d_head // 2)
    # xk: (B, T, n_kv_head, d_head // 2)
    xq_ = torch.view_as_complex(xq)
    xk_ = torch.view_as_complex(xk)

    # reshape for broadcasting
    assert freqs_cis.shape == (T, d_head // 2)
    shape = [d if i in (1, 3) else 1 for i, d in enumerate(xq_.shape)]
    # freqs_cis: (1, T, 1, d_head // 2)
    freqs_cis = freqs_cis.view(*shape).to(xq_.device)

    # apply the rotation matrix to the query and key
    # xq_encoded: (B, T, n_head, d_head // 2)
    # xk_encoded: (B, T, n_kv_head, d_head // 2)
    xq_encoded = xq_ * freqs_cis
    xk_encoded = xk_ * freqs_cis

    # Convert to real and then flatten
    # x{qk}_out: (..., d_head // 2) -> (..., d_head // 2, 2) -> (..., d_head)
    xq_out = torch.view_as_real(xq_encoded).flatten(3)
    xk_out = torch.view_as_real(xk_encoded).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    B, T, n_kv_heads, d_head = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(B, T, n_kv_heads, n_rep, d_head)
        .reshape(B, T, n_kv_heads * n_rep, d_head)
    )


class Attention(nn.Module):
    """Multi-head causal attention."""

    def __init__(self, cfg: "Llama3Config"):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_kv_repeats = self.n_head // self.n_kv_head
        self.d_head = cfg.d_model // cfg.n_head

        # query, key, value heads
        self.wq = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        self.wk = nn.Linear(self.d_model, self.n_kv_head * self.d_head, bias=False)
        self.wv = nn.Linear(self.d_model, self.n_kv_head * self.d_head, bias=False)
        # output projection
        self.wo = nn.Linear(self.d_model, self.d_model, bias=False)

        # sets flag for init weight scaling
        self.wo.LLMC_RESIDUAL_SCALE_FLAG = 1  # type: ignore

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
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
        # xq: (B, T, d_model) -> (B, T, n_head, d_head)
        xq = xq.view(B, T, self.n_head, self.d_head)
        # x{k,v}: (B, T, d_model) -> (B, T, n_kv_head, d_head)
        xk = xk.view(B, T, self.n_kv_head, self.d_head)
        xv = xv.view(B, T, self.n_kv_head, self.d_head)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Swap n_head and T axes
        # xq: (B, T, nhead, d_head) -> (B, n_head, T, d_head)
        xq = xq.transpose(1, 2)
        # x{k,v}: (B, T, n_kv_head, d_head) -> (B, n_kv_head, T, d_head)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash attention
        # Expects:
        # - xq: (B, n_head, T, d_head)
        # - xk: (B, n_kv_head, T, d_head)
        # - xv: (B, n_kv_head, T, d_head)
        # Returns:
        # - z: (B, n_head, T, d_head)
        # handles default scaling of 1/sqrt(d_head)
        enable_gqa = self.n_head != self.n_kv_head
        z = F.scaled_dot_product_attention(
            xq, xk, xv, is_causal=True, enable_gqa=enable_gqa
        )

        # re-assemble all head outputs side-by-side
        # (B, n_head, T, d_head) -> (B, T, d_model)
        z = z.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # project to output: (B, T, d_model) -> (batch, T, d_model)
        out = self.wo(z)
        return out

    def sample(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Attention.sample not implemented")


class InferenceAttention(Attention):
    """Multi-head causal attention for inference.

    Includes KV-caching for faster inference.
    Note: we keep this as a separate class so we are note wasting
    """

    def __init__(self, cfg: "Llama3Config"):
        super().__init__(cfg)
        self.register_buffer(
            "cache_k",
            torch.zeros(
                (cfg.max_sample_batch_size, cfg.n_kv_head, cfg.n_ctx, self.d_head)
            ),
            persistent=False,
        )
        self.register_buffer(
            "cache_v",
            torch.zeros(
                (cfg.max_sample_batch_size, cfg.n_kv_head, cfg.n_ctx, self.d_head)
            ),
            persistent=False,
        )

    def sample(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
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
        # mask: (1, 1, N, T)
        assert mask.shape == (1, 1, N, T)

        # compute query, key, value for all heads
        # (B, N, d_model) -> (B, N, d_model)
        xq = self.wq(x)
        # (B, N, d_model) -> (B, N, n_kv_head * d_head)
        xk = self.wk(x)
        xv = self.wv(x)

        # reshape each so heads are in separate dimensions and swap axes
        # xq: (B, N, d_model) -> (B, N, n_head, d_head)
        xq = xq.view(B, N, self.n_head, self.d_head)
        # x{k,v}: (B, N, d_model) -> (B, N, n_kv_head, d_head)
        xk = xk.view(B, N, self.n_kv_head, self.d_head)
        xv = xv.view(B, N, self.n_kv_head, self.d_head)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Swap n_head and T axes
        # xq: (B, N, nhead, d_head) -> (B, n_head, N, d_head)
        xq = xq.transpose(1, 2)
        # x{k,v}: (B, N, n_kv_head, d_head) -> (B, n_kv_head, N, d_head)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # update cache with KV values for new positions
        self.cache_k[:B, :, start_pos:T] = xk
        self.cache_v[:B, :, start_pos:T] = xv
        # get cached K, V values for all positions up the current position
        # (B, n_kv_head, T, d_head)
        xk = self.cache_k[:B, :, :T]
        xv = self.cache_v[:B, :, :T]

        # compute and rescale attention scores using flash attention
        # handles default scaling of 1/sqrt(d_head)
        # z: (B, n_head, N, d_head)
        z = F.scaled_dot_product_attention(xq, xk, xv, is_causal=False, attn_mask=mask)
        # re-assemble all head outputs side-by-side
        # (B, n_head, N, d_head) -> (B, N, d_model)
        z = z.transpose(1, 2).contiguous().view(B, N, self.d_model)

        # project to output: (B, N, d_model) -> (B, N, d_model)
        return self.wo(z)


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

        # non-linearity for SwiGLU activation function
        # Initialize it here so we get consistent state
        self.silu = nn.SiLU()

        # Flag to indicate that the weight tensor should be scaled during initialization
        # TODO check if this is correct for Llama-3
        self.w3.LLMC_RESIDUAL_SCALE_FLAG = 1  # type: ignore

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
        self.n_head = cfg.n_head
        self.dim = cfg.d_model
        self.d_head = cfg.d_model // cfg.n_head

        self.attention_norm = RMSNorm(cfg)
        if cfg.inference_mode:
            self.attention = InferenceAttention(cfg)
        else:
            self.attention = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg)
        self.feed_forward = MLP(cfg)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def sample(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention.sample(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Llama3(BaseLLM["Llama3Config"]):
    """Llama-3 model."""

    def __init__(self, cfg: "Llama3Config"):
        super().__init__(cfg)
        self.vocab_size = cfg.vocab_size
        self.n_layer = cfg.n_layer

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.layers = torch.nn.ModuleList()
        for _ in range(cfg.n_layer):
            self.layers.append(TransformerBlock(cfg))

        self.norm = RMSNorm(cfg)
        self.output = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Note that n_ctx is multiplied by 2 to allow for dynamism of token lengths
        # while training or fine-tuning.
        self.freqs_cis = precompute_freqs_cis(cfg.d_model // cfg.n_head, cfg.n_ctx * 2)

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
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass on token idxs, with optional loss computation."""
        _, T = tokens.size()
        assert (
            self.cfg.n_ctx >= T
        ), f"Cannot forward sequence of length {T}, ctx size is only {self.cfg.n_ctx}"

        # token embeddings of shape
        # x: (B, T, d_model)
        h = self.tok_embeddings(tokens)

        # Get RoPE freqs for the current sequence
        freqs_cis = self.freqs_cis[:T]

        for layer in self.layers:
            if self.cfg.activation_checkpointing:
                h = checkpoint(layer, h, freqs_cis, use_reentrant=False)
            else:
                h = layer(h, freqs_cis)
        h = self.norm(h)

        logits = self.output(h)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            loss = None

        if not return_logits:
            # there are performance reasons why not returning logits is prudent,
            # if not needed
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
        B, N = tokens.size()
        T = start_pos + N
        assert (
            self.cfg.n_ctx >= T
        ), f"Cannot sample sequence of length {T}, ctx size is only {self.cfg.n_ctx}"

        # generate embedding
        # shape (B, N, d_model)
        h = self.tok_embeddings.wte(tokens)

        # get RoPE freqs for the current sequence
        freqs_cis = self.freqs_cis[start_pos : start_pos + N]

        # To save memory we compute the mask only once and reuse it for all layers
        # We also only generate the mask for the new sequence since when performing
        # KV-caching, the matrix of attention scores we need to mask for the new
        # sequence is of size (N, T), with only masked entries (i, j) for
        # j > start_pos + i, since row i corresponds to token start_pos + i.
        # Mask shape: (N, N)
        mask = torch.triu(torch.full((N, N), float("-inf"), device=device), diagonal=1)
        # Mask shape: (N, T)
        mask = torch.hstack([torch.zeros((N, start_pos), device=device), mask]).type_as(
            h
        )
        # Make it broadcastable to (B, n_heads, N, T) for flash attention
        mask = mask.view(1, 1, N, T)

        # forward thru blocks: x = (B, N, d_model)
        for layer in self.layers:
            h = layer.sample(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        # (B, N, d_model) -> (B, N, vocab_size)
        logits = self.output(h)
        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: list[torch.Tensor],
        max_new_tokens: int,
        end_token: int | None = None,
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
        B = len(prompt_tokens)
        assert B == 1, "Only batch size 1 is supported for now"
        assert self.cfg.max_sample_batch_size >= B
        if not self.cfg.inference_mode:
            # fall back to less efficient base class implementation which uses the
            # forward pass without KV-caching
            return super().generate(
                prompt_tokens, max_new_tokens, end_token, temperature, top_k
            )

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.cfg.n_ctx
        max_total_len = min(self.cfg.n_ctx, max_prompt_len + max_new_tokens)

        # initialize token buffer with prompt tokens
        tokens = torch.full(
            (B, max_total_len),
            self.tokenizer.pad_id,
            dtype=torch.long,
        )
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)

        # initially prev_pos=0 as we need to generate the prompt tokens to populate
        # the kv-cache properly
        prev_pos = 0
        eos_reached = torch.tensor([False] * B)
        input_tokens_mask = tokens != self.tokenizer.pad_id
        if min_prompt_len == max_total_len:
            logits = self.sample(tokens, prev_pos)

        # TODO handle case where min_prompt_len == max_total_len ??

        for cur_pos in range(min_prompt_len, max_total_len):
            # forward the model to get the logits for the index in the sequence
            logits = self.sample(tokens[:, prev_pos:cur_pos], prev_pos)
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
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            # append sampled index to the running sequence and continue
            tokens = torch.cat((tokens, next_token), dim=1)
            if end_token is not None and next_token == end_token:
                break

        return tokens

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

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
            )  # type: ignore
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
    def from_pretrained(cls, cfg: "Llama3Config") -> "Llama3":
        """Loads pretrained Llama-3 model weights from huggingface"""
        print(f"Loading weights from pretrained llama3 {cfg.model_name}")
        checkpoint_dir = download_llama3_weights(cfg.model_name)
        # NOTE: we do a manual copy to avoid problems with some nn.Module parameters
        #  not existing in the checkpoint, specifically buffers like `attn.mask`
        sd_hf = torch.load(checkpoint_dir / "original" / "consolidated.00.pth")  # type: ignore
        print("Creating model and loading weights")
        model = Llama3(cfg)
        sd = model.state_dict()
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        for k in sd_hf:
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])
        return model


def download_llama3_weights(model_name: str) -> Path:
    """Download the weights for the given model name."""
    print(f"Downloading weights for {model_name}")
    checkpoint_dir = get_base_dir_path() / "checkpoints" / model_name
    if checkpoint_dir.exists():
        print(
            f"Existing checkpoint found for {model_name} at {checkpoint_dir} skipping download"
        )
        return checkpoint_dir

    from huggingface_hub import snapshot_download

    hf_model_id_map = {
        "llama-3.2-1B": "meta-llama/Llama-3.2-1B",
        "llama-3.2-3B": "meta-llama/Llama-3.2-3B",
        "llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    }

    download_dir = snapshot_download(
        repo_id=hf_model_id_map[model_name],
        repo_type="model",
        local_dir=checkpoint_dir,
    )
    if download_dir != checkpoint_dir:
        warnings.warn(
            f"Downloaded weights for {model_name} to {download_dir}, but expected {checkpoint_dir}"
        )

    return Path(download_dir)
