"""
https://discuss.pytorch.org/t/clone-and-detach-in-v0-4-0/16861

This file experiments with different and equivalent ways for
performing gradient descent on a simple linear regression problem.

All methods implemented in this file for manual gradient descent 
are equivalent.
"""

import time
import argparse
import torch
torch.manual_seed(0)

# Gradient descent arguments
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=int, default=1)
parser.add_argument('--iter', type=int, default=500)
parser.add_argument('--verbose', dest="verbose", action="store_true")
parser.set_defaults(verbose=False)
args = parser.parse_args()
N_ITER = int(args.iter)
METHOD = int(args.method)

# Number of samples
N = 100

# True function: y = b + w * x
coefs = [10.1, 24.12]
x = torch.rand(N,1) * 5
y = coefs[0] + coefs[1] * x

# Get some noisy observations
y_obs = y + 0.2 * torch.randn(N,1)

# Initialize parameters to learn
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Learning rate (descent step size)
gamma = 0.01

# Perform gradient descent iterations
total_time = 0.0
for i in range(N_ITER):
    print("Iteration", i+1)

    # Calculate loss
    y_pred = w * x + b
    mse = torch.mean((y_pred - y_obs) ** 2)

    # Backward pass to compute gradients
    mse.backward()

    # Method 1
    if METHOD == 1:
        start_time = time.time()
        with torch.no_grad():
            w -= gamma * w.grad
            b -= gamma * b.grad
        assert w.requires_grad
        assert b.requires_grad
        # Have to zero grad here, since we are doing an inplace update
        # for w and b. Otherwise, the gradient will accumulate.
        w.grad.zero_()
        b.grad.zero_()
        total_time += time.time() - start_time

    # Method 2
    if METHOD == 2:
        start_time = time.time()
        with torch.no_grad():
            w = w - gamma * w.grad
            b = b - gamma * b.grad
        assert not w.requires_grad
        assert not b.requires_grad
        # Since we're not doing inplace update for w and b, need to
        # reset the requires_grad option to True.
        w.requires_grad_()
        b.requires_grad_()
        total_time += time.time() - start_time

    # Method 3; probably shouldn't use this method
    if METHOD == 3:
        start_time = time.time()
        w.data = w.data - gamma * w.grad
        b.data = b.data - gamma * b.grad
        assert w.requires_grad
        assert b.requires_grad
        w.grad.zero_()
        b.grad.zero_()
        total_time += time.time() - start_time

    # Method 4
    if METHOD == 4:
        # The detach() method makes sure that w is not a part of the
        # original computation graph. It creates a new tensor that
        # does not require grad, but shares memory with the original 
        # tensor.
        start_time = time.time()
        w = w.detach() - gamma * w.grad.detach()
        b = b.detach() - gamma * b.grad.detach()
        assert not w.requires_grad
        assert not b.requires_grad
        w.requires_grad_()
        b.requires_grad_()
        total_time += time.time() - start_time

    # Method 5
    if METHOD == 5:
        start_time = time.time()
        w = w.detach() - gamma * w.grad
        b = b.detach() - gamma * b.grad
        assert not w.requires_grad
        assert not b.requires_grad
        w.requires_grad_()
        b.requires_grad_()
        total_time += time.time() - start_time

    # Method 6
    if METHOD == 6:
        # *.detach().clone() will create a copy of the original tensor
        # with new memory, new computation graph, but without requires 
        # grad. Since this is creating new memory, this could be slow,
        # although this problem is too easy and so does not separate
        # these methods in terms of timing very well.
        start_time = time.time()
        w = w.detach().clone() - gamma * w.grad
        b = b.detach().clone() - gamma * b.grad
        assert not w.requires_grad
        assert not b.requires_grad
        w.requires_grad_()
        b.requires_grad_()
        total_time += time.time() - start_time

    # Print some diagnostics
    if args.verbose:
        print('w:', w)
        print('b:', b)
        print('w.grad:', w.grad)
        print('b.grad:', b.grad)
        print(type(w))

print("Learned:")
print(w.data)
print(b.data)
print("True:")
print(coefs[1])
print(coefs[0])

assert torch.allclose(torch.Tensor([coefs[1]]), w.data, atol=1e-1)
assert torch.allclose(torch.Tensor([coefs[0]]), b.data, atol=1e-1)
print("Approximately correct.")
print("Method", METHOD)
print("Average time per iteration: {} secs".format(total_time / N_ITER))


