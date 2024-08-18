# 高斯核
from sklearn import datasets  # 导入数据集
from sklearn.svm import SVC  # 导入svm
from sklearn.pipeline import Pipeline  # 导入python里的管道
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # 导入标准化
from matplotlib.colors import ListedColormap
import numpy as np


def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1))
    # meshgrid函数是从坐标向量中返回坐标矩阵
    x_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(x_new)  # 获取预测值
    zz = y_predict.reshape(x0.shape)
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)


data_x, data_y = datasets.make_moons(noise=0.15, random_state=777)  # 生成月亮数据集
# random_state是随机种子，nosie是方
plt.scatter(data_x[data_y == 0, 0], data_x[data_y == 0, 1])
plt.scatter(data_x[data_y == 1, 0], data_x[data_y == 1, 1])
data_x = data_x[data_y < 2, :2]  # 只取data_y小于2的类别，并且只取前两个特征
plt.show()


def RBFKernelSVC(gamma=1):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', gamma=gamma))
    ])


svc = RBFKernelSVC(gamma=100)  # gamma参数很重要，gamma参数越大，支持向量越小
svc.fit(data_x, data_y)
plot_decision_boundary(svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(data_x[data_y == 0, 0], data_x[data_y == 0, 1], color='red')  # 画点
plt.scatter(data_x[data_y == 1, 0], data_x[data_y == 1, 1], color='blue')
plt.show()
