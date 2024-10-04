import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn

# generate a data set
x = torch.linspace(-np.pi, np.pi, 100)
# example for dynamic plot
plt.ion()
for i in range(10):
    plt.cla()
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-5, 5)
    y = torch.sin(x) + torch.rand(x.size())   # random around sin(x)
    plt.scatter(x,y)
    plt.pause(0.1)   # represent 0.1s
plt.ioff()
plt.show()

x = torch.unsqueeze(x, dim=1)   # add dim >> 1 (2)
y = torch.unsqueeze(y, dim=1)
# Set a network (nn)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.predict = nn.Sequential(    # net
            nn.Linear(1, 50),  # All2All 1 input  10 output
            nn.ReLU(),         # ReLU
            nn.Linear(50, 1)   # All2All 10 input 1 output
        )

    def forward(self, x):      # def forward process
        prediction = self.predict(x)
        return prediction

# train Net
net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)  # use a optimizer
loss_func = nn.MSELoss()                                # setting a loss function

for epoch in range(2000):
    out = net(x)               # input x
    loss = loss_func(out, y)   # calculate loss
    optimizer.zero_grad()
    loss.backward()            # BP process
    optimizer.step()           # optimize
    if epoch % 25 == 0:
        plt.cla()
        plt.scatter(x, y)      # plot
        plt.plot(x, out.data.numpy(), "r", lw=2)
        plt.text(0.5, 0, f"loss = {loss}")
        plt.pause(0.1)
plt.ioff()
plt.show()



