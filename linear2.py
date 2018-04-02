import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt



# 生成一些样本
x_data = [1,2,3,4,5,6,7]

y_data = [4,7,10,13,16,19,22]

#画出散点图
plt.plot(x_data, y_data, 'r*', label="Original data")
plt.show()



# 生成1维W矩阵，取值是[-1, 1]之间的随机数
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
# 生成1维b矩阵，初始值是0
b = tf.Variable(tf.zeros([1]), name='b')
# 经过计算取得预估值y
y = W * x_data + b

# 以预估值y和实际值y_data之间的均方误差作为损失
loss = tf.reduce_mean(tf.square(y - y_data), name='loss')
# 采用梯度下降法来优化参数
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练的过程就是最小化这个误差值
train = optimizer.minimize(loss, name='train')

sess = tf.Session()        #这种定义session的方法也可以，但是不推荐。
init = tf.global_variables_initializer()
sess.run(init)

# 初始化的w和b是多少
print("W=", sess.run(W), "b=", sess.run(b), "loss=", sess.run(loss))
# 执行20次训练
for step in range(20):
    sess.run(train)
# 输出训练好的W和b
    print("W=", sess.run(W), "b=", sess.run(b), "loss=", sess.run(loss))
plt.plot(x_data, y_data, 'r*', label="Original data") # 红色星形的点
plt.title("Linear Regression using Gradient Descent")
plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label="Fitted line") # 画出拟合的线
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
sess.close()
