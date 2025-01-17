"""
Downloads and evaluates HellaSwag in Python.
Also writes the data (tokens, labels) to .bin files for parallel evaluation in C.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- llm.c 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- llm.c 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.

Ref: https://github.com/karpathy/llm.c/blob/master/dev/data/hellaswag.py
"""

import json
from typing import Iterator

import torch

from gollem.data.common import DATA_CACHE_DIR
from gollem.data.common import download_file
from gollem.tokenizer import BaseTokenizer


# TODO
# - figure out how to deal with EOT and BOS tokens
# - write tokenized data to .bin files


# -----------------------------------------------------------------------------
THIS_DATA_CACHE_DIR = DATA_CACHE_DIR / "hellaswag"

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


def download(split: str) -> None:
    """Downloads HellaSwag DATA_CACHE_DIR"""
    THIS_DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = THIS_DATA_CACHE_DIR / f"hellaswag_{split}.jsonl"
    if not data_filename.exists():
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")


def render_example(
    example: dict, tokenizer: BaseTokenizer
) -> tuple[dict, torch.Tensor, torch.Tensor, int]:
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # TODO figure out how should deal with EOT and BOS tokens

    # gather up all the tokens
    ctx_tokens = tokenizer.encode(ctx, add_eot=False)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = tokenizer.encode(
            " " + end
        )  # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # be careful during collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split: str) -> Iterator[dict]:
    # there are 10,042 examples in total in val
    download(split)
    with open(THIS_DATA_CACHE_DIR / f"hellaswag_{split}.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="HellaSwag dataset preprocessing",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument(
#         "-m",
#         "--model_desc",
#         type=str,
#         default="gpt-2",
#         choices=["gpt-2", "llama-3"],
#         help="Model type (determines the tokenizer)",
#     )
#     args = parser.parse_args()
#     tokenizer = get_model_config(args.model_desc).get_tokenizer()
#     evaluate(args.model_type, args.device)
