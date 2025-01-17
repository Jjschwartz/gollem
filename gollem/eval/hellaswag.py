import torch
import torch.nn.functional as F

from gollem.data.hellaswag import iterate_examples
from gollem.data.hellaswag import render_example
from gollem.models import get_model_config
from gollem.models.model import BaseLLM


@torch.no_grad()
def evaluate(model: BaseLLM, device: str | torch.device):
    torch.set_float32_matmul_precision("high")  # use tf32

    tokenizer = model.cfg.get_tokenizer()

    datas = []
    num_correct_norm = 0
    num_correct = 0
    for num_total, example in enumerate(iterate_examples("val")):
        data, tokens, mask, label = render_example(example, tokenizer)
        datas.append(data)
        tokens = tokens.to(device)  # (4, S)
        mask = mask.to(device)  # (4, S)

        # get the logits (at all positions)
        logits = model(tokens, inference=False)[0]  # (4, S, V)

        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()  # (4, S-1, V)
        shift_tokens = (tokens[..., 1:]).contiguous()  # (4, S-1)
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # (4*(S-1), V)
        flat_shift_tokens = shift_tokens.view(-1)  # (4*(S-1))

        shift_losses = F.cross_entropy(
            flat_shift_logits, flat_shift_tokens, reduction="none"
        )
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (
            mask[..., 1:]
        ).contiguous()  # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        acc = num_correct / (num_total + 1)
        acc_norm = num_correct_norm / (num_total + 1)
        print(
            f"{num_total + 1} "
            f"acc: {num_correct}/{num_total + 1}={acc:.4f} "
            f"acc_norm: {num_correct_norm}/{num_total + 1}={acc_norm:.4f}"
        )

        # debug: pretty print a few examples, and the losses in each case
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print("Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

    # now write the data to a .bin file
    # filename = THIS_DATA_CACHE_DIR / "hellaswag_val.bin"
    # write_evalfile(filename, datas)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_desc", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = get_model_config(args.model_desc)
    model = model_cfg.get_model_and_optimizer(device)[0]
    evaluate(model, device)
