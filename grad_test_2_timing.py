import time
import torch

def reset_grad(var):
    if var.grad is not None:
        var.grad.zero_()

W1 = torch.rand(5, 4, requires_grad=True)
W2 = torch.rand(5, 4, requires_grad=True)
x = torch.ones(4, 1, requires_grad=True)

##############################################################
# METHOD (3)
# Test for when both branches have same weights.
# Use y1.backward() and y2.backward() to compute gradient
# for W1
reset_grad(x)
reset_grad(W1)
reset_grad(W2)

y1 = W1.matmul(x).sum()
y2 = W1.matmul(x).sum()
z = y1 + y2

# After backward pass from y1
start = time.time()
y1.backward()
dt_1 = time.time() - start

x_grad = W1.data.sum(axis=0)
assert torch.allclose(x.grad.flatten(), x_grad)
assert torch.allclose(W1.grad, torch.ones_like(W1))

# After backward pass from y2, gradient w.r.t. x is accumulated
# from both y1 and y2.
start = time.time()
y2.backward()
dt_2 = time.time() - start

x_grad = 2.0 * W1.data.sum(axis=0)
assert torch.allclose(x.grad.flatten(), x_grad)

# Gradient w.r.t. W1 is accumulated from y1 and y2.
assert torch.allclose(W1.grad, 2.0 * torch.ones_like(W1))
print("Method 3 correct")
print("Time for y1 and y2 backward: {} s".format(dt_1 + dt_2))

##############################################################
# METHOD (4)
# Test for when both branches have same weights.
# Use z.backward() to compute gradient for W1
reset_grad(x)
reset_grad(W1)
reset_grad(W2)

y1 = W1.matmul(x).sum()
y2 = W1.matmul(x).sum()
z = y1 + y2

start = time.time()
z.backward()
t = time.time() - start

# Gradient w.r.t. W1 computed from z
x_grad = 2.0 * W1.data.sum(axis=0)
assert torch.allclose(x.grad.flatten(), x_grad)
assert torch.allclose(W1.grad, 2.0 * torch.ones_like(W1))
print("Method 4 correct")
print("Time for z backward: {} s".format(t))


