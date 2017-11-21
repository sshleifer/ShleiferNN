import torch
from torch.autograd import Variable

N, D_in, D_out = 100, 10, 1

x = Variable(torch.randn(N, D_in).type(torch.FloatTensor),
             requires_grad=False)
y = Variable(torch.randn(N, D_out).type(torch.FloatTensor),
             requires_grad=False)

w1 = Variable(torch.randn(D_in, D_out).type(torch.FloatTensor),
              requires_grad=True)

learning_rate = 1e-6

for t in range(500):
    yhat = x.mm(w1)
    loss = (yhat -y).pow(2).sum()
    loss.backward()
    w1.data -= learning_rate * w1.grad.data
    w1.grad.data.zero_()