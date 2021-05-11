# 本例采用八维特征，即输入数据x = [batchsize,in_features] = [batchsize,8]
# 数据集采用糖尿病数据集：diabetes.csv.gz

# Dataset
import numpy as np
import torch
data = np.loadtxt("diabetes.csv.gz",delimiter = ',', dtype = np.float32)      # data的类型为numpy.ndarray
x_data = torch.from_numpy(data[:,:-1])                  # data是一个矩阵，第一个:表示取所有行，:-1表示取除最后一列外的所有列，列数为8
y_data = torch.from_numpy(data[:,[-1]])                 # 这里-1要加一个[]，使y_data从一个向量转为一个n行1列的矩阵
print(x_data)
print(y_data)

# Model:这里采用3层线性层，以增强信息获取能力
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(in_features = 8, out_features = 6, bias = True)
        self.linear2 = torch.nn.Linear(in_features = 6, out_features = 4, bias = True)
        self.linear3 = torch.nn.Linear(in_features = 4, out_features = 1, bias = True)
        self.activate = torch.nn.Sigmoid()              # 方便更改激活函数，如果要用其他激活函数，只需修改这一处即可

    def forward(self,x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x

model = Model()

# Loss and Optimizer
criterion = torch.nn.BCELoss(reduction = 'mean')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

# Training cycle
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch+1, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

