import torch
import numpy as np
import matplotlib.pyplot as plt

# 定义 x 和 y 的范围
x = torch.linspace(0, 1, 100)
y = torch.linspace(0, 1, 100)

# 生成网格点
X, Y = torch.meshgrid(x, y)

# 计算函数值
u = torch.sin(np.pi * X) * torch.sin(30 * np.pi * Y)

# 将 PyTorch 张量转换为 NumPy 数组，以便使用 Matplotlib
X = X.numpy()
Y = Y.numpy()
u = u.numpy()

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u, cmap='viridis')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')

plt.show()