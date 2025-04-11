# numpy_cifar10_simple_neural_network

## 本项目使用 NumPy 从零开始实现了一个神经网络，具备以下功能：
前向传播和反向传播
随机梯度下降（SGD）优化
交叉熵损失计算
L2 正则化
超参数调优
训练和验证损失及准确率的可视化

## 使用方法
1.下载cifar10数据集
2.直接运行main函数。main函数中会首先使用data_loader函数读取数据，接着调用超参数调优函数，进行模型的训练和测试，并保存搜寻到的最优超参数。然后使用最优的超参数进行训练和测试，最终得到loss曲线和测试集上的准确率。


### 要求
确保你已经安装了以下 Python 包：
numpy
matplotlib
sklearn
keras（仅用于数据加载）
