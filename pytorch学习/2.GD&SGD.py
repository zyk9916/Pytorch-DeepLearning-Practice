# 梯度下降算法只能找到局部最优
# 但在深度学习领域，实际上很少出现局部最优点，使用梯度下降算法往往能找到全局最优点
# 深度学习领域需要克服的是鞍点（cost梯度为0的点）的影响

# 梯度下降与随机梯度下降的区别：
# 梯度下降是用整个训练集的cost（比如所有样本loss的平均值）对参数求导
# 随机梯度下降是用训练集中的一个随机样本的loss对参数求导，学习率往往比梯度下降设置的小一些
# 随机梯度下降经常可以避免鞍点的影响，从而避免陷入局部最优点，性能更高
# 但是，梯度下降法的各样本的梯度是可以并行计算的，随机梯度下降则不行。
# 如果数据集很大，梯度下降所需要的算力也会加大，有可能无法负载。
# 综合考虑GD和SGD，为了使算法性能和时间效率得到折中，经常把数据集分成若干个batch，在每个batch内部使用梯度下降


import numpy as np
import matplotlib.pyplot as plt
import random

# 梯度下降
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 4
w_list = [4]
epoch = 100
epoch_list = [0]
learning_rate = 0.01

for i in range (epoch):
    sum_gradient = 0
    for x_val, y_val in zip(x_data, y_data):
        gradient = 2 * x_val * (x_val*w - y_val)
        sum_gradient += gradient
    sum_gradient = sum_gradient / 3
    w -= learning_rate * sum_gradient
    w_list.append(w)
    epoch_list.append(i+1)

plt.plot(epoch_list,w_list)
plt.ylabel('w')
plt.xlabel('epoch')
plt.show()

# 随机梯度下降的代码类似

# 在随机梯度下降中，如何选取随机样本？
f = [1,2,3]
g = [4,5,6]
fg = list(zip(f,g))
z = random.sample(fg,1)
print(z)
x,y = z[0]
print(x)
print(y)