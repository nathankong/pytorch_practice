import torch

with torch.no_grad():
    W1 = torch.rand(5,4,requires_grad=True)
    W2 = torch.rand(5,4,requires_grad=True)

    x = torch.ones(4,1)
    delta = torch.ones(4,1) * 0.1
    delta.requires_grad_()

    with torch.enable_grad():
        # z1 = (W1 * (x + delta)).sum()
        z1 = W1.matmul(x + delta).sum()
        # z2 = (W2 * x).sum()
        x.requires_grad_()
        z2 = W2.matmul(x).sum()

        # nabla_x (W2 * x).sum()
        grad_x = torch.autograd.grad(z2, [x])[0]
        x.requires_grad = False

        # delta^T \nabla_x z2(x)
        a = torch.mul(delta, grad_x).sum()

        # Loss function
        loss = z1 + z2 + a

        grad_delta = torch.autograd.grad(loss, [delta])[0]

        print(grad_delta.flatten())
        print(W1.sum(axis=0).flatten() + grad_x.flatten())

        # Asserts to make sure behaviour is correct
        assert torch.allclose(grad_delta.flatten(), \
                        W1.sum(axis=0).flatten() + grad_x.flatten())
        assert W1.grad is None
        assert W2.grad is None
        assert x.grad is None


W = torch.rand(5,4,requires_grad=True)
x = torch.Tensor([1.,2.,3.,4.]).reshape(4,1)
x.requires_grad_()
y = W.matmul(x).sum()

x_grad = torch.autograd.grad(y, [x])[0].flatten()
print(x_grad)
assert torch.allclose(W.sum(axis=0), x_grad)
assert W.grad is None

x.requires_grad = False
y = W.matmul(x).sum()
y.backward()
print(W.grad)
assert x.grad is None

# Gradients accumulate
y = W.matmul(x).sum()
y.backward()
print(W.grad)


