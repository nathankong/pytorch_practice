import torch

W = torch.rand(5, 4, requires_grad=True)
x = torch.ones(4, 1, requires_grad=True)
with torch.no_grad():
    y = W.matmul(x).sum()

assert not y.requires_grad

with torch.no_grad():
    with torch.enable_grad():
        y = W.matmul(x).sum()

    z = 2.0 * y

assert y.requires_grad
assert not z.requires_grad

y.backward()
print(W.grad.data)
print(x.grad.data)

