import torch

q = torch.rand(4, 2, requires_grad=True)
w = q.detach()

assert not w.requires_grad

print(q)
w[1,1] = 1000.0
print(w)
print(q)
