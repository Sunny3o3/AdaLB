import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 利用有限差分法求解二维burgers方程
nx = 41
ny = 41
nt = 120
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = 0.0009
nu = 0.01
dt = sigma * dx * dy / nu

# 划分网格
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

# 初始化条件
u = np.ones((nx, ny))
v = np.ones((nx, ny))
comb = np.ones((nx, ny))

u[int(0.5 / dx):int(1 / dx + 1),int(0.5 / dy):int(1 / dy + 1)] = 2
v[int(0.5 / dx):int(1 / dx + 1),int(0.5 / dy):int(1 / dy + 1)] = 2

# 计算
for n in range(nt + 1):
	un = u.copy()
	vn = v.copy()

	u[1:-1, 1:-1] = un[1:-1, 1:-1] - dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - u[0:-2, 1:-1]) - \
	                dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) + \
	                nu * dt / dx**2 * (un[2: , 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) +\
	                nu * dt / dx**2 * (un[1:-1, 2: ] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
	v[1:-1, 1:-1] = vn[1:-1, 1:-1] - dt / dx * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - u[0:-2, 1:-1]) - \
	                dt / dx * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) + \
	                nu * dt / dx**2 * (vn[2: , 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) +\
	                nu * dt / dx**2 * (vn[1:-1, 2: ] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])

	u[0, : ] = 1
	u[-1, : ] =1
	u[:, 0] =1
	u[:, -1] = 1

	v[0, : ] = 1
	v[-1, : ] =1
	v[:, 0] =1
	v[:, -1] = 1

# 绘制结果
fig = plt.figure(figsize=(11, 7),dpi=100)
ax = Axes3D(fig)
x,y = np.meshgrid(x,y)
ax.plot_surface(x,y,u[:],cmap='viridis',rstride=1,cstride=1)
ax.plot_surface(x,y,v[:],cmap='viridis',rstride=1,cstride=1)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()
