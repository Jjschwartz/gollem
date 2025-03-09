def compute_intermediate_size(
    d_model: int, ffn_dim_multiplier: float | None = None, multiple_of: int = 256
):
    d = int(2 * 4 * d_model / 3)
    if ffn_dim_multiplier is not None:
        d = int(ffn_dim_multiplier * d)
    return multiple_of * ((d + multiple_of - 1) // multiple_of)


expected_sizes = {
    4096: 14336,
    8192: 28672,
    16384: 53248,
}

for d_model, expected_size in expected_sizes.items():
    valid_multiple_of = []
    for p in range(8, 20):
        multiple_of = 2**p
        if expected_size % multiple_of == 0:
            valid_multiple_of.append(multiple_of)
    print(f"d_model: {d_model}, valid_multiple_of: {valid_multiple_of}")
    for multiple_of in valid_multiple_of:
        for ffn_dim_multiplier in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
            intermediate_size = compute_intermediate_size(
                d_model, ffn_dim_multiplier, multiple_of
            )
            if intermediate_size == expected_size:
                print(
                    f"d_model: {d_model}, "
                    f"ffn_dim_multiplier: {ffn_dim_multiplier}, "
                    f"multiple_of: {multiple_of}, "
                    f"intermediate_size: {intermediate_size}"
                )
