import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10_from_dir(data_dir='cifar-10-batches-py'):
    # 加载训练集
    x_train = []
    y_train = []

    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        x_train.append(batch[b'data'])
        y_train += batch[b'labels']

    x_train = np.concatenate(x_train, axis=0).astype(np.float32) / 255.
    y_train = np.array(y_train)

    # 加载测试集
    test_batch = unpickle(os.path.join(data_dir, 'test_batch'))
    x_test = test_batch[b'data'].astype(np.float32) / 255.
    y_test = np.array(test_batch[b'labels'])

    # one-hot 编码
    y_train_onehot = np.eye(10)[y_train]
    y_test_onehot = np.eye(10)[y_test]

    # 训练集划分验证集
    x_train, x_val, y_train_onehot, y_val_onehot = train_test_split(
        x_train, y_train_onehot, test_size=0.1, random_state=42
    )

    return x_train, y_train_onehot, x_val, y_val_onehot, x_test, y_test_onehot
