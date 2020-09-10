import torch

torch.manual_seed(0)
class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.linear = torch.nn.Linear(2,2)

    def forward(self, x):
        y = self.linear(x)
        return y

t = Test()

for name, p in t.named_parameters():
    print name, p.requires_grad

t.eval()
for name, p in t.named_parameters():
    print name, p.requires_grad
t.train()

print(t.linear.weight)
x = torch.Tensor([2,4])
print("x requires grad", x.requires_grad)

y = t(x).sum()
print(y)

#from torchviz import make_dot
#make_dot(y, dict(t.named_parameters())).render("graph", format="png")

y.backward()
print(t.linear.weight.grad)
print("orig x grad", x.grad)

t.zero_grad()

t.eval()
#x_adv = x.detach() + torch.ones(2)
x_adv = x + torch.ones(2)
x_adv.requires_grad_(True)
print("x_adv requires_grad", x_adv.requires_grad)
print(x_adv)

y = t(x_adv).sum()
grad_x, = torch.autograd.grad(y, [x_adv])
#y.backward(); grad_x = x_adv.grad
print("x_adv grad", grad_x)
print("orig x grad", x.grad)
print("W grad", t.linear.weight.grad)

t.zero_grad()
print("orig W grad", t.linear.weight.grad)
y = t(x).sum()
y.backward()
print("W grad", t.linear.weight.grad)

#from torchviz import make_dot
#make_dot(y, dict(t.named_parameters())).render("graph", format="png")



