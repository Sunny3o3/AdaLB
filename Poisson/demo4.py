"""
A scratch for PINN solving the following PDE
sin u_xx-u_yyyy=(2-x^2)*exp(-y)
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
from optimal import *

loss_list = []
loss_res_list=[]
loss_bc_list=[]

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ODE_Net(nn.Module):
    def __init__(self, hidden_units=20):
        super(ODE_Net, self).__init__()
        self.layer1 = nn.Linear(1, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units)
        self.layer3 = nn.Linear(hidden_units, hidden_units)
        self.layer4 = nn.Linear(hidden_units, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        out = self.activation(self.layer1(x))
        out = self.activation(self.layer2(out))
        out = self.activation(self.layer3(out))
        out = self.layer4(out)
        return out


def residual(model, x):
    x.requires_grad_(True)
    y = model(x)
    y_x = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                              create_graph=True)[0]
    y_xx = torch.autograd.grad(y_x, x, grad_outputs=torch.ones_like(y_x),
                               create_graph=True)[0]

    res = y_xx + 0.49 * torch.sin(0.7 * x) + 2.25 * torch.cos(1.5 * x)
    return res


def boundary_loss(model):
    x10 = torch.tensor([[10.0]], device=device, requires_grad=True)
    y10 = model(x10)  # y(10)

    x_10 = torch.tensor([[-10.0]], device=device, requires_grad=True)
    y_10 = model(x_10)  # y(-10)



    bc1 = y10 + torch.sin(torch.tensor(7)) - torch.cos(torch.tensor(15)) - 1
    bc2 = y_10 - torch.sin(torch.tensor(7)) - torch.cos(torch.tensor(15)) + 1


    loss_bc = bc1 ** 2 + bc2 ** 2
    return loss_bc


def main():
    model = ODE_Net(hidden_units=20).to(device)
    optimizer = myADAM(model.parameters(), gamma=0.99)
    #optimizer = optim.Adam(model.parameters())
    #optimizer = AdaBelief(model.parameters())
    #optimizer = AdamW(model.parameters())
    #optimizer = optim.Adam(model.parameters(),amsgrad=True)
    num_epochs = 5000

    N_interior = 100
    #x_interior = torch.arange(-10, 10 + 1/N_interior, 1/N_interior)
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





    model.eval()
    x_test = torch.linspace(-1, 1, 5000, device=device).unsqueeze(1)
    y_pred = model(x_test).detach().cpu().numpy().flatten()
    x_test_np = x_test.cpu().numpy().flatten()

    # Analytical solution: y(x) = 8 + 4x - 7e^x + 3xe^x
    y_true = np.sin(0.7 * x_test_np) + np.cos(1.5 * x_test_np) - 0.1 * x_test_np

    plt.figure(figsize=(8, 4))
    plt.plot(x_test_np, y_pred, label="Solution using PINN")
    plt.plot(x_test_np, y_true, '--', label="Analytical solution")
    #plt.plot(x_test_np, y_pred-y_true, '--', label="Analytical solution")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.show()



    with open("./loss_my.txt", 'w') as loss:
    #with open("./loss_Adam.txt", 'w') as loss:
    #with open("./loss_AdaBelief.txt", 'w') as loss:
    #with open("./loss_AdamW.txt", 'w') as loss:
    #with open("./loss_AMSgrad.txt", 'w') as loss:
        loss.write(str(loss_list))
    #return loss, loss_list

if __name__ == "__main__":
    main()