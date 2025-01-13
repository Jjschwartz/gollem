import abc
from functools import cached_property

import tiktoken
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase


class BaseTokenizer(abc.ABC):
    """Base class for all tokenizers.

    Purpose is to provide a common interface for all tokenizers.
    """

    @abc.abstractmethod
    def encode(self, text: str, add_eot: bool = True) -> list[int]:
        pass

    @abc.abstractmethod
    def decode(self, tokens: list[int]) -> str:
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
        assert add_eot, "HuggingFaceTokenizer always adds eot token"
        # by default the tokenizer adds the EOT token (128000)
        return self.tokenizer.encode(
            text, add_special_tokens=False, verbose=False, split_special_tokens=True
        )

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    @property
    def eot_token(self) -> int:
        # by default the tokenizer adds the EOT token (128000)
        # so we return token for empty string
        return self.tokenizer.encode("")[0]

    @property
    def name(self) -> str:
        return self.tokenizer.name_or_path

    @cached_property
    def n_vocab(self) -> int:
        return len(self.tokenizer.get_vocab())


def get_tokenizer(model_desc: str) -> BaseTokenizer:
    if model_desc.startswith("gpt-"):
        return TiktokenTokenizer(tiktoken.get_encoding("gpt2"))
    elif "llama-3" in model_desc:
        return HuggingFaceTokenizer(
            AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        )
    else:
        raise ValueError(f"Unknown model descriptor: {model_desc}")
