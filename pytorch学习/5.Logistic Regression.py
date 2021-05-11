# 使用Logistic Regression处理二分类问题。其损失函数为交叉熵损失函数：Binary Cross Entropy(BCE)
# 在有了Linear_model by pytorch的框架之后，只需稍加修改就可以完成逻辑斯蒂回归的代码编写。

# torch.sigmoid:调用logistic函数，输入输出都为tensor
# class torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
# weight:a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch.
# size_average和reduce已弃用。reduction有'mean'和'sum'两种，默认为'mean'

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # 解决报错
import numpy as np
import matplotlib.pyplot as plt
import torch

x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[0],[0],[1]])

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(in_features = 1, out_features = 1, bias = True)
    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))        # 不同点
        return y_pred

model = LogisticRegressionModel()
criterion = torch.nn.BCELoss(reduction = 'sum')                     # 不同点
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred.float(),y_data.float())         # 修改数据类型，不然会报错
    print(epoch + 1, loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 可视化：
x = np.linspace(0, 10, 200)                     # 在[0,10]区间内采样200个点
x_test = torch.tensor(x).view((200,1))          # 转换为200*1的tensor
y_test = model(x_test.float())
y = y_test.data.numpy()                         # 再转换为numpy类型的n维数组
plt.plot(x, y)
plt.plot([0,10],[0.5,0.5],c='r')                # 添加一条y=0.5的直线
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()                                      # 绘制网格线
plt.show()