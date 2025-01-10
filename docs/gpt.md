# GPT

Playing around with training GPT style transformers.


## Optimizations to test

A number of optimizations are included, see [benchmarking.ipynb] for a comparison

- [x] different floating point precision
- [x] fused adamW optimizer
- [x] flash attention
- [x] torch compile
- [x] tensorcores


## Architecture improvements to test

- [ ] unshared embed/unembed weights
- [ ] rotary positional embeddings
- [ ] SwiGLU/GeGLU MLP activations

## Distributed Training

- [ ] distributed data parallel training (requires multiple-GPUs)