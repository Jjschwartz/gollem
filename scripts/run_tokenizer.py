import argparse

from gollem.tokenizer import get_tokenizer


def main(model_desc: str):
    model_tokenizer = get_tokenizer(model_desc)

    try:
        while True:
            text = input("> ")
            tokens = model_tokenizer.encode(text)
            token_strs = model_tokenizer.convert_token_ids_to_tokens(tokens)

            output_lines = [[], [], []]
            tokens.insert(0, "token_id")  # type: ignore
            token_strs.insert(0, "token")
            for i, (token, token_str) in enumerate(zip(tokens, token_strs)):
                w = max(len(token_str), len(str(token)))
                output_lines[0].append(f"{str(i):{w}s}")
                output_lines[1].append(f"{str(token):{w}s}")
                output_lines[2].append(f"{str(token_str):{w}s}")

            for line in output_lines:
                print(" | ".join(line))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs interactive tokenizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model_desc",
        default="gpt2",
        choices=["gpt2", "llama3"],
        help="Model type (determines the tokenizer)",
    )
    args = parser.parse_args()
    main(args.model_desc)
