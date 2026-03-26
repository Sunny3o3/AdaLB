"""
A scratch for PINN solving the following PDE
y_xxxx-2y_xxx+y_xx=0
Author: ST
Date: 2023/2/26
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.optimizer import Optimizer
import math


class mySin(torch.nn.Module):  # sin激活函数
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.sin()
        return x


class mySin2(torch.nn.Module):  # sin激活函数
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = pow(x.sin(),2)
        return x


loss_list = []

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ODE_Net(nn.Module):
    def __init__(self, hidden_units=10):
        super(ODE_Net, self).__init__()
        self.layer1 = nn.Linear(1, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units)
        self.layer3 = nn.Linear(hidden_units, hidden_units)
        self.layer4 = nn.Linear(hidden_units, hidden_units)
        self.layer7 = nn.Linear(hidden_units, 1)
        self.activation = mySin2()

    def forward(self, x):
        out = self.activation(self.layer1(x))
        out = self.activation(self.layer2(out))
        out = self.activation(self.layer3(out))
        out = self.activation(self.layer4(out))
        out = self.layer7(out)

        return out


def residual(model, x):
    x.requires_grad_(True)
    y = model(x)
    y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                              create_graph=True)[0]
    y_xx = torch.autograd.grad(y_x, x, grad_outputs=torch.ones_like(y_x),
                               create_graph=True)[0]

    res = -0.0001 * y_xx + 5 * (pow(y,3)-y)
    return res


def boundary_loss(model):
    x1 = torch.tensor([[1.0]], device=device, requires_grad=True)
    x2 = torch.tensor([[-1.0]], device=device, requires_grad=True)
    y1 = model(x1)
    y2 = model(x2)

    y1_x = torch.autograd.grad(y1, x1, grad_outputs=torch.ones_like(y1),
                               create_graph=True)[0]
    y2_x = torch.autograd.grad(y2, x2, grad_outputs=torch.ones_like(y2),
                                create_graph=True)[0]


    bc1 = y1 + 1.0  # y(0) = 1
    bc2 = y2 + 1.0  # y'(0) = 0
    bc3 = y1_x - 0.0  # y''(0) = -1  -> y0_xx + 1 = 0
    bc4 = y2_x - 0.0  # y'''(0) = 2

    loss_bc = bc1 ** 2 + bc2 ** 2 + bc3 ** 2 + bc4 ** 2
    return loss_bc


def main():
    model = ODE_Net(hidden_units=50).to(device)
    # optimizer = myADAM(model.parameters(), gamma=0.99)
    optimizer = optim.Adam(model.parameters())
    # optimizer = AdaBelief(model.parameters())
    # optimizer = AdamW(model.parameters())
    # optimizer = optim.Adam(model.parameters(),amsgrad=True)
    num_epochs = 5001

    N_interior = 50
    x_interior = torch.rand(N_interior, 1, device=device)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        r_interior = residual(model, x_interior)
        loss_res = torch.mean(r_interior ** 2)

        loss_bc = boundary_loss(model)
        loss = loss_res + loss_bc
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6e}")









if __name__ == "__main__":
    main()