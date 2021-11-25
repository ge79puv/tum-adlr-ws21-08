import torch
from torch.autograd import Variable

x = Variable(torch.tensor([[3.], [2]]), requires_grad=True)
print(x)

y = Variable(torch.zeros(3, 1), requires_grad=False)
y[0] = x[0]**2
y[1] = x[1]**3
y[2] = x[1]**4
# y.backward(gradient=torch.ones(y.size()))
# print(x.grad)
y[0].backward()
print(x.grad)
y[1].backward()
print(x.grad)
y[2].backward()
print(x.grad)
