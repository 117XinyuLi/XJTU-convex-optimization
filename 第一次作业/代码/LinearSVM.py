from sklearn import datasets  # 导入数据集
from sklearn.svm import LinearSVC  # 导入线性svm
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

data_x, data_y = datasets.make_moons(noise=0.15, random_state=777)  # 生成月亮数据集
# random_state是随机种子，nosie是方
plt.scatter(data_x[data_y == 0, 0], data_x[data_y == 0, 1])
plt.scatter(data_x[data_y == 1, 0], data_x[data_y == 1, 1])
data_x = data_x[data_y < 2, :2]  # 只取data_y小于2的类别，并且只取前两个特征
plt.show()

scaler = StandardScaler()  # 标准化
scaler.fit(data_x)  # 计算训练数据的均值和方差
data_x = scaler.transform(data_x)  # 再用scaler中的均值和方差来转换X，使X标准化
liner_svc = LinearSVC(C=1e9, max_iter=1000000)  # 线性svm分类器,iter是迭达次数，c值决定的是容错，c越大，容错越小
liner_svc.fit(data_x, data_y)


# 边界绘制函数
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


# 画图并显示参数和截距
plot_decision_boundary(liner_svc, axis=[-3, 3, -3, 3])
plt.scatter(data_x[data_y == 0, 0], data_x[data_y == 0, 1], color='red')
plt.scatter(data_x[data_y == 1, 0], data_x[data_y == 1, 1], color='blue')
plt.show()
print('参数权重')
print(liner_svc.coef_)
print('模型截距')
print(liner_svc.intercept_)
