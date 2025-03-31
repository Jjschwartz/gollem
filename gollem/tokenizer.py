import abc
from functools import cached_property

import numpy as np
import tiktoken
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase


def get_token_dtype(n_vocab: int) -> np.uint16 | np.uint32:
    if n_vocab <= 2**16:
        return np.uint16  # type: ignore
    elif n_vocab <= 2**32:
        return np.uint32  # type: ignore
    raise ValueError(f"Vocab size {n_vocab} is too large for uint16 or uint32")


class BaseTokenizer(abc.ABC):
    """Base class for all tokenizers.

    Purpose is to provide a common interface for all tokenizers.
    """

    @abc.abstractmethod
    def encode(self, text: str, add_eot: bool = True) -> list[int]:
        """Encode a string into a list of tokens IDs."""
        pass

    @abc.abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Decode a list of tokens IDs into a string."""
        pass

    @abc.abstractmethod
    def convert_token_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        """Convert a list of token IDs to a list of tokens."""
        pass

    @property
    @abc.abstractmethod
    def eot_token(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def n_vocab(self) -> int:
        pass

    @property
    def token_dtype(self) -> np.uint16 | np.uint32:
        return get_token_dtype(self.n_vocab)


class TiktokenTokenizer(BaseTokenizer):
    def __init__(self, tokenizer: tiktoken.Encoding):
        self.tokenizer = tokenizer

    def encode(self, text: str, add_eot: bool = True) -> list[int]:
        tokens = self.tokenizer.encode_ordinary(text)
        if add_eot:
            tokens.append(self.eot_token)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def convert_token_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        token_strs = []
        for token_id in token_ids:
            token_strs.append(self.tokenizer.decode([token_id]))
        return token_strs

    @property
    def eot_token(self) -> int:
        return self.tokenizer._special_tokens["<|endoftext|>"]

    @property
    def name(self) -> str:
        return self.tokenizer.name

    @property
    def n_vocab(self) -> int:
        return self.tokenizer.max_token_value + 1


class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def encode(self, text: str, add_eot: bool = True) -> list[int]:
        # TODO need to check if this is correct/fix it
        assert add_eot, "HuggingFaceTokenizer always adds eot token"
        # by default the tokenizer adds a BOS token (128000) at the beginning
        # of each sequence
        return self.tokenizer.encode(
            text, add_special_tokens=False, verbose=False, split_special_tokens=True
        )

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def convert_token_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(token_ids)  # type: ignore

    @property
    def eot_token(self) -> int:
        # by default the tokenizer adds the BOS token (128000)
        # so we return token for empty string
        return self.tokenizer.encode("")[0]

    @property
    def name(self) -> str:
        return self.tokenizer.name_or_path.split("/")[-1]

    @cached_property
    def n_vocab(self) -> int:
        return len(self.tokenizer.get_vocab())


def get_tokenizer(model_desc: str) -> BaseTokenizer:
    if model_desc.startswith("gpt"):
        return TiktokenTokenizer(tiktoken.get_encoding("gpt2"))
    elif "llama-3" in model_desc:
        return HuggingFaceTokenizer(
            AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        )
    else:
        raise ValueError(f"Unknown model descriptor: {model_desc}")
