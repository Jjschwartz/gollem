def compute_palm_flops(
    non_embedding_params: int, n_layer: int, n_head: int, d_model: int, n_ctx: int
):
    """Estimate of the model flops following PaLM paper formula"""
    L, H, Q, T = n_layer, n_head, d_model // n_head, n_ctx
    mf_per_token = 6 * non_embedding_params + 12 * L * H * Q * T
    mf = mf_per_token * n_ctx
    return mf
