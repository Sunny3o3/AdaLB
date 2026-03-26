import torch
import seaborn as sns
import matplotlib.pyplot as plt

# 选择GPU或CPU
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# 从文件加载已经训练完成的模型
model_loaded = torch.load('model.pth', map_location=device)
model_loaded.eval()  # 设置模型为evaluation状态

# 生成时空网格
h = 0.01
k = 0.01
x = torch.arange(-1, 1, h)
t = torch.arange(0, 1, k)
X = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T
X = X.to(device)

# 计算该时空网格对应的预测值
with torch.no_grad():
    U_pred = model_loaded(X).reshape(len(x), len(t)).cpu().numpy()

# 绘制计算结果
plt.figure(figsize=(5, 3), dpi=300)
xnumpy = x.numpy()
plt.plot(xnumpy, U_pred[:, 0], 'o', markersize=1)
plt.plot(xnumpy, U_pred[:, 20], 'o', markersize=1)
plt.plot(xnumpy, U_pred[:, 40], 'o', markersize=1)
plt.figure(figsize=(5, 3), dpi=300)
sns.heatmap(U_pred, cmap='jet')
plt.show()


