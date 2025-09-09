import torch

x = torch.tensor(1, device='cuda:0', dtype=torch.float32)
print(f"{x=} {x.dtype=}")
# output: tensor(1, device='cuda:0') float32
y = x.to('cuda:1')
torch.cuda.synchronize(1)
print(f"{y=} {y.dtype=}")
# output: tensor(0, device='cuda:1') float32

print("--------------------------------")

x = torch.tensor(1, device='cuda:0', dtype=torch.float32)
print(f"{x=} {x.dtype=}")
# output: tensor(1, device='cuda:0')
y = x.to('cpu')
print(f"{y=} {y.dtype=}")
# output: tensor(0, device='cpu')
z = y.to('cuda:1')
print(f"{z=} {z.dtype=}")
# output: tensor(0, device='cuda:1')