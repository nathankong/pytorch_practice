import torch

def reset_grad(var):
    if var.grad is not None:
        var.grad.zero_()

W1 = torch.rand(5, 4, requires_grad=True)
W2 = torch.rand(5, 4, requires_grad=True)
x = torch.ones(4, 1, requires_grad=True)

y1 = W1.matmul(x).sum()
y2 = W2.matmul(x).sum()
z = y1 + y2

## METHOD (1)
# Use y1.backward() and y2.backward() to compute gradient
# w.r.t. x.
y1.backward()
y2.backward()

assert torch.allclose(W1.grad, torch.ones_like(W1))
assert torch.allclose(W2.grad, torch.ones_like(W1))

# Gradient w.r.t x is accumulated from y1 and y2.
x_grad = W1.data.sum(axis=0) + W2.data.sum(axis=0)
assert torch.allclose(x.grad.flatten(), x_grad)

## FIRST RESET GRADIENTS
reset_grad(x)
reset_grad(W1)
reset_grad(W2)

## METHOD (2)
# Now using z.backward()
y1 = W1.matmul(x).sum()
y2 = W2.matmul(x).sum()
z = y1 + y2

z.backward()

# Gradient w.r.t. x computed from z
x_grad = W1.data.sum(axis=0) + W2.data.sum(axis=0)
assert torch.allclose(x.grad.flatten(), x_grad)


