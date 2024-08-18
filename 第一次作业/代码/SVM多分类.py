import numpy as np
from matplotlib import colors
from sklearn import svm
from sklearn.svm import SVC
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_iris


def iris_type(s):
    # 数据转为整型，数据集标签类别由string转为int
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


# 定义分类器
def classifier():
    clf = svm.SVC(C=0.5,  # 误差项惩罚系数
                  kernel='rbf',  # 线性核 kernel="rbf":高斯核
                  decision_function_shape='ovr', max_iter=10000)  # 决策函数
    return clf


def train(clf, x_train, y_train):
    # x_train：训练数据集
    # y_train：训练数据集标签
    # 训练开始
    # 同flnumpy.ravelatten将矩阵拉平
    clf.fit(x_train, y_train.ravel(), sample_weight=None)


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print('%s Accuracy:%.3f' % (tip, np.mean(acc)))


def print_accuracy(clf, x_train, y_train, x_test, y_test):
    show_accuracy(clf.predict(x_train), y_train, 'traing data')
    show_accuracy(clf.predict(x_test), y_test, 'testing data')


def draw(clf, x, label):
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围

    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    grid_hat = clf.predict(grid_test)  # 预测分类值 得到【0,0.。。。2,2,2】
    grid_hat = grid_hat.reshape(x1.shape)  # reshape grid_hat和x1形状一致

    # 指定默认的字体
    mpl.rcParams['font.sans-serif'] = ['SimHei']

    # 设置颜色
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])


    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light) # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=30, cmap=cm_dark)  # 样本点
    plt.scatter(x_test[:, 0], x_test[:, 1], s=30, facecolor='none', zorder=10)  # 测试点
    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    plt.title("SVM对鸢尾花分类")
    plt.show()


# 训练四个特征：
data = load_iris()
x, y = data.data, data.target

# x_train,x_test,y_train,y_test = 训练数据，测试数据，训练数据标签，测试数据标签
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1,
                                                                    test_size=0.3)  # 数据集划分成70%30%测试集

iris_feature = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
clf = classifier()  # 声明svm分类器对象
train(clf, x_train, y_train)  # 启动分类器进行模型训练
print_accuracy(clf, x_train, y_train, x_test, y_test)

# 训练两个特征（用于画图展示）
data = load_iris()
x, y = data.data, data.target
x = x[:, :2]  # 只要前两个特征，此时只训练前两个特征，用于画图
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, random_state=1, test_size=0.3)
clf = classifier()
train(clf, x_train, y_train)
print_accuracy(clf, x_train, y_train, x_test, y_test)
draw(clf, x, iris_feature[:2])

# 训练两个特征（用于画图展示）
data = load_iris()
x, y = data.data, data.target
x = x[:, 2:4]  # 只要后两个特征，此时只训练后两个特征，用于画图
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, random_state=1, test_size=0.3)
clf = classifier()
train(clf, x_train, y_train)
print_accuracy(clf, x_train, y_train, x_test, y_test)
draw(clf, x, iris_feature[2:])

# 训练两个特征（用于画图展示）
data = load_iris()
x, y = data.data, data.target
x = x[:, 0:3:2]  # 只要第0列和第2列特征，此时只训练第0列和第2列特征，用于画图
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, random_state=1, test_size=0.3)
clf = classifier()
train(clf, x_train, y_train)
print_accuracy(clf, x_train, y_train, x_test, y_test)
draw(clf, x, iris_feature[0:3:2])

# 训练两个特征（用于画图展示）
data = load_iris()
x, y = data.data, data.target
x = x[:, 1:4:2]  # 只要第1列和第3列特征，此时只训练第1列和第3列特征，用于画图
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, random_state=1, test_size=0.3)
clf = classifier()
train(clf, x_train, y_train)
print_accuracy(clf, x_train, y_train, x_test, y_test)
draw(clf, x, iris_feature[1:4:2])


