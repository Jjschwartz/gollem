import time
from typing import Iterator

import torch
import torch.nn.functional as F

from gollem.data.hellaswag import iterate_examples
from gollem.data.hellaswag import render_example
from gollem.models import get_model_config
from gollem.models.model import BaseLLM
from gollem.models.model import load_model
from gollem.tokenizer import BaseTokenizer


def batch_render_examples(
    examples: list[dict], tokenizer: BaseTokenizer, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rendered_examples = [render_example(example, tokenizer) for example in examples]

    max_len = max(tokens.size(1) for tokens, _, _ in rendered_examples)
    B = len(rendered_examples)
    C = len(examples[0]["endings"])
    tokens = torch.zeros((B * C, max_len), dtype=torch.long, device=device)
    mask = torch.zeros((B * C, max_len), dtype=torch.long, device=device)
    labels = torch.zeros((B,), dtype=torch.long, device=device)
    for i, (tok_row, mask_row, label) in enumerate(rendered_examples):
        j_start, j_end = i * C, (i + 1) * C
        S = tok_row.size(1)
        tokens[j_start:j_end, :S] = tok_row
        mask[j_start:j_end, :S] = mask_row
        labels[i] = label

    return tokens, mask, labels


def iterate_examples_batch(
    split: str,
    tokenizer: BaseTokenizer,
    device: str | torch.device,
    batch_size: int = 4,
) -> Iterator[tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor]]:
    examples = []
    for example in iterate_examples(split):
        examples.append(example)
        if len(examples) == batch_size:
            yield examples, *batch_render_examples(examples, tokenizer, device)
            examples = []
    if examples:
        yield examples, *batch_render_examples(examples, tokenizer, device)


@torch.no_grad()
def evaluate(
    model: BaseLLM, device: str | torch.device, batch_size: int = 4, split: str = "val"
):
    torch.set_float32_matmul_precision("high")  # use tf32

    tokenizer = model.cfg.get_tokenizer()

    C = 4  # number of choices

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    start_time = time.time()
    model_times = []
    for batch_num, batch in enumerate(
        iterate_examples_batch(split, tokenizer, device, batch_size)
    ):
        examples, tokens, mask, labels = batch
        # tokens = (B*C, S)
        # mask = (B*C, S)
        # labels = (B,)
        B = len(examples)

        # get the logits (at all positions)
        model_start_time = time.time()
        logits = model(tokens, inference=False)[0]  # (B*C, S, V)
        model_times.append(time.time() - model_start_time)

        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()  # (B*C, S-1, V)
        shift_tokens = (tokens[..., 1:]).contiguous()  # (B*C, S-1)
        flat_shift_logits = shift_logits.view(
            -1, shift_logits.size(-1)
        )  # (B*C*(S-1), V)
        flat_shift_tokens = shift_tokens.view(-1)  # (B*C*(S-1))

        shift_losses = F.cross_entropy(
            flat_shift_logits, flat_shift_tokens, reduction="none"
        )  # (B*C*(S-1),)
        shift_losses = shift_losses.view(tokens.size(0), -1)  # (B*C, S-1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        # we must shift mask, so we start at the last prompt token
        shift_mask = (mask[..., 1:]).contiguous()  # (B*C, S-1)
        masked_shift_losses = shift_losses * shift_mask  # (B*C, S-1)
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)  # (B*C,)
        avg_loss = sum_loss / shift_mask.sum(dim=1)  # (B*C,)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        # we first reshape to (B, C) and then take the argmin over the choice dim
        pred = sum_loss.view(B, C).argmin(dim=1)  # (B,)
        pred_norm = avg_loss.view(B, C).argmin(dim=1)  # (B,)

        # accumulate stats
        num_total += B
        num_correct += (pred == labels).sum().item()
        num_correct_norm += (pred_norm == labels).sum().item()
        acc = num_correct / (num_total + 1)
        acc_norm = num_correct_norm / (num_total + 1)
        print(
            f"{num_total + 1} "
            f"acc: {num_correct}/{num_total + 1}={acc:.4f} "
            f"acc_norm: {num_correct_norm}/{num_total + 1}={acc_norm:.4f}"
        )

        # debug: pretty print a few examples, and the losses in each case
        if batch_num < 10:
            print("---")
            print(f"Context:\n {examples[0]['ctx']}")
            print("Endings:")
            for i, end in enumerate(examples[0]["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm[0]}, actual: {labels[0]}")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.4f} seconds")
    print(f"Model time taken: {sum(model_times):.4f} seconds")
    print(f"Model / total: {sum(model_times) / (end_time - start_time):.4f}")


def run_eval_from_model_checkpoint(
    checkpoint_path: str, device: str | torch.device | None, batch_size: int = 4, split: str = "val"
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(checkpoint_path, device)
    evaluate(model, device, batch_size, split)


def run_eval_from_model_desc(
    model_desc: str, device: str | torch.device | None = None, batch_size: int = 4, split: str = "val"
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = get_model_config(model_desc)
    model = model_cfg.get_model_and_optimizer(device)[0]
    evaluate(model, device, batch_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_desc", type=str, default=None)
    parser.add_argument("-d", "--device", type=str, default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-c", "--checkpoint_path", type=str, default=None)
    parser.add_argument("-s", "--split", type=str, default="val", choices=["val", "test", "train"])
    args = parser.parse_args()

    if args.checkpoint_path:
        run_eval_from_model_checkpoint(
            args.checkpoint_path, args.device, args.batch_size, args.split
        )
    else:
        run_eval_from_model_desc(args.model_desc, args.device, args.batch_size, args.split)
