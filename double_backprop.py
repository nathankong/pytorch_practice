import torch
torch.manual_seed(0)

"""
torch.autograd.grad(z,[x])[0] == W.sum(axis=0)

z = Wx
y = d^T \nabla_x z + Wx_2

Want to compute dy/dW. We need to backprop through
\nabla_x z since z depends on W.
"""

#====================================
# Wrong way to do double backprop

W = torch.rand(5, 4, requires_grad=True)
x = torch.ones(4, 1, requires_grad=True)
x2 = torch.ones(4, 1) * 2.0
z = W.matmul(x).sum()
d = torch.ones(4, 1) * 0.5

grad_x = torch.autograd.grad(z, [x])[0]
assert d.shape == grad_x.shape
y = W.matmul(x2).sum() + torch.mul(d, grad_x).sum()
x.requires_grad = False
assert W.requires_grad

y.backward()
print(W.grad)
assert torch.allclose(W.grad, torch.ones_like(W)*2.0)
assert x.grad is None

#====================================
# Right way to do double backprop

W = torch.rand(5, 4, requires_grad=True)
x = torch.ones(4, 1, requires_grad=True)
x2 = torch.ones(4, 1) * 2.0
z = W.matmul(x).sum()
d = torch.Tensor([1.0, 2.0, 3.0, 4.0]).reshape(4,-1)

grad_x = torch.autograd.grad(z, [x], create_graph=True)[0]
assert d.shape == grad_x.shape
y = W.matmul(x2).sum() + torch.mul(d, grad_x).sum()
x.requires_grad = False
assert W.requires_grad

y.backward()
print(W.grad)
assert torch.allclose(torch.mul(torch.ones(W.shape[0],1), d.t()) + \
        torch.ones_like(W) * 2.0, W.grad)
assert x.grad is None


