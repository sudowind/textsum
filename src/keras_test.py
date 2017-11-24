import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

np.random.seed(1337)

# 生成数据  
X = np.linspace(-1, 1, 200)  # 在返回（-1, 1）范围内的等差序列
np.random.shuffle(X)  # 打乱顺序
Y = np.array([x * x for x in X]) + 2 + np.random.normal(0, 0.05, (200,))  # 生成Y并添加噪声
# plot  
# plt.scatter(X, Y)
# plt.show()

X_train, Y_train = np.array([X[:160], X[:160]]).T, Y[:160]  # 前160组数据为训练数据集
X_test, Y_test = np.array([X[160:], X[160:]]).T, Y[160:]  # 后40组数据为测试数据集
# print(X_train)

# 构建神经网络模型  
# model = Sequential()
# model.add(Dense(input_dim=2, units=10, activation='relu'))
# model.add(Dense(input_dim=10, units=10,  activation='relu'))
# model.add(Dense(input_dim=10, units=1))

a = Input(shape=(2,))
b = Dense(10, activation='relu')(a)
c = Dense(1, activation='relu')(b)
model = Model(inputs=a, outputs=c)

# 选定loss函数和优化器  
model.compile(loss='mse', optimizer='sgd')

# 训练过程  
print('Training -----------')
for step in range(501):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 50 == 0:
        print("After %d trainings, the cost: %f" % (step, cost))

        # 测试过程
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
# W, b = model.layers[0].get_weights()
# print('Weights=', W, '\nbiases=', b)

# 将训练结果绘出  
Y_pred = model.predict(X_test)
plt.scatter(X[160:], Y_test)
plt.scatter(X[160:], Y_pred)
plt.show() 