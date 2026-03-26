import math
import torch
import numpy as np
from network import Network
import matplotlib.pyplot as plt



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



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_surface(X, T, U_pred, title="Predicted Solution"):
    """
    绘制三维曲面图
    :param X: x 坐标值
    :param T: t 坐标值
    :param U_pred: 预测的 u 值
    :param title: 图像标题
    """
    X = X.detach().cpu().numpy()
    T = T.detach().cpu().numpy()
    U_pred = U_pred.detach().cpu().numpy()

    # 将数据重新调整为网格形式
    X_grid = X.reshape(int(np.sqrt(len(X))), -1)
    T_grid = T.reshape(int(np.sqrt(len(T))), -1)
    U_grid = U_pred.reshape(X_grid.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_grid, T_grid, U_grid, cmap="viridis", edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(x, t)")
    ax.set_title(title)
    plt.show()



class PINN:

    def __init__(self):

        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = Network(
            input_size=2,
            hidden_size=100,
            output_size=1,
            depth=5,
            act=torch.nn.ReLU
        ).to(device)

        self.h = 0.1
        self.k = 0.1
        x = torch.arange(0, 1 + self.h, self.h)
        t = torch.arange(0, 1 + self.k, self.k)


        self.X_inside = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T


        bc1 = torch.stack(torch.meshgrid(x[0], t)).reshape(2, -1).T
        bc2 = torch.stack(torch.meshgrid(x[1], t)).reshape(2, -1).T
        #ic = torch.stack(torch.meshgrid(x, t[0])).reshape(2, -1).T
        self.X_boundary = torch.cat([bc1, bc2])


        u_bc1 = torch.zeros(len(bc1))
        u_bc2 = torch.zeros(len(bc2))
        #u_ic = -torch.sin(math.pi * ic[:, 0])
        self.U_boundary = torch.cat([u_bc1, u_bc2])
        self.U_boundary = self.U_boundary.unsqueeze(1)


        self.X_inside = self.X_inside.to(device)
        self.X_boundary = self.X_boundary.to(device)
        self.U_boundary = self.U_boundary.to(device)
        self.X_inside.requires_grad = True


        self.criterion = torch.nn.MSELoss()


        self.iter = 1

        self.optim = torch.optim.Adam(self.model.parameters())


    def loss_func(self):

        self.optim.zero_grad()
        #self.lbfgs.zero_grad()


        U_pred_boundary = self.model(self.X_boundary)
        loss_boundary = self.criterion(
            U_pred_boundary, self.U_boundary)


        U_inside = self.model(self.X_inside)


        du_dX = torch.autograd.grad(
            inputs=self.X_inside,
            outputs=U_inside,
            grad_outputs=torch.ones_like(U_inside),
            retain_graph=True,
            create_graph=True
        )[0]
        du_dx = du_dX[:, 0]
        du_dt = du_dX[:, 1]


        du_dxx = torch.autograd.grad(
            inputs=self.X_inside,
            outputs=du_dX,
            grad_outputs=torch.ones_like(du_dX),
            retain_graph=True,
            create_graph=True
        )[0][:, 0]
        loss_equation = self.criterion(
            du_dx+900*U_inside.squeeze(), (900-901*pow(math.pi,2))*torch.sin(np.pi*U_inside.squeeze())*torch.sin(np.pi*U_pred_boundary))

        loss = loss_equation + loss_boundary


        loss.backward()


        if self.iter % 100 == 0:
            print(self.iter, loss.item())
        self.iter = self.iter + 1

        loss_list.append(loss.item())
        #with open("./loss_belief.txt", 'w') as loss:
        #   loss.write(str(loss_list))
        return loss,loss_list


    def train(self):
        self.model.train()

        print("optimization")
        for i in range(10000):
            self.optim.step(self.loss_func)





loss_list = []

pinn = PINN()


pinn.train()

# 在训练完成后调用
X = pinn.X_inside[:, 0]  # x 坐标
T = pinn.X_inside[:, 1]  # t 坐标
U_pred = pinn.model(pinn.X_inside).detach()  # 预测的 u 值

plot_surface(X, T, U_pred)





#torch.save(pinn.model, 'model.pth')





