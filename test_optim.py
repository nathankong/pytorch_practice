import torch
import torch.optim as optim

#==============================================================
# Only update W
W = torch.rand(5, 4, requires_grad=True)
x = torch.ones(4, 1, requires_grad=True)
optimizer = torch.optim.SGD([W], lr=0.1)

# Get answer for W's gradient update
W_new = W.data - (0.1 * torch.ones_like(W))

z = W.matmul(x).sum()
z.backward()

assert torch.allclose(W.grad.data, torch.ones_like(W))
assert torch.allclose(x.grad.flatten(), W.data.sum(axis=0))
optimizer.step()

# Asser that x is not updated.
assert torch.allclose(x.data, torch.ones_like(x))
assert torch.allclose(W.data, W_new)
print("Correct")

#==============================================================
# Update both W and x
W = torch.rand(5, 4, requires_grad=True)
x = torch.ones(4, 1, requires_grad=True)
optimizer = torch.optim.SGD([W, x], lr=0.1)

# Get answer for W's and x's gradient update
W_new = W.data - (0.1 * torch.ones_like(W))
x_new = x.data - (0.1 * W.data.sum(axis=0).reshape(x.shape))

z = W.matmul(x).sum()
z.backward()

assert torch.allclose(W.grad.data, torch.ones_like(W))
assert torch.allclose(x.grad.flatten(), W.data.sum(axis=0))
optimizer.step()

# Asser that x is not updated.
assert torch.allclose(x.data, x_new)
assert torch.allclose(W.data, W_new)
print("Correct")

#==============================================================
# Manually update gradient of W, but use optimizer to update W
W = torch.rand(5, 4, requires_grad=True)
x = torch.ones(4, 1, requires_grad=False)
optimizer = torch.optim.SGD([W], lr=0.1)

z = W.matmul(x).sum()
grad_W = torch.autograd.grad(z, [W])[0].detach()

# Manual update of W
W_new = W - (0.1 * grad_W)

# Manually change W's gradient and use optimizer to update W
W.grad = grad_W
optimizer.step()
optimizer.zero_grad()

assert torch.allclose(W.data, W_new)
assert torch.equal(W.grad.data, torch.zeros_like(W))
print("Correct")


