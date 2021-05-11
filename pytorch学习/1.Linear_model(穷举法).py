import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w_list = []                             # 备选的w列表
mse_list = []                           # 均方误差列表

def forward(x, w):                         # 前馈
    return x * w

def loss(x, y, w):                         # 单个样本的损失
    y_pred = forward(x, w)
    return (y - y_pred) * (y - y_pred)

for w in np.arange(0.0,4.1,0.1):
    print("w=", w)
    loss_sum = 0
    for x_val ,y_val in zip(x_data, y_data):        # val是validation的缩写，意为验证
        print(x_val,y_val)
        y_pred_val = forward(x_val, w)
        loss_val = loss(x_val, y_val, w)
        loss_sum += loss_val
        print('\t',x_val, y_val, y_pred_val,loss_val)
    print("MSE=", loss_sum / 3)
    w_list.append(w)
    mse_list.append(loss_sum / 3)

plt.plot(w_list,mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()


