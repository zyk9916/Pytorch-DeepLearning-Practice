# pytorch中SGD优化器会使用全部传入的数据来计算梯度，在之前的例子中，我们有：y_pred = model(x_data)，将所有数据都做前馈处理，并计算loss和梯度
# 实际上是实现了Batch-GD（即所有样本的梯度下降），（虽然优化器用的是SGD，但仍然是Batch-GD）
# 在实际应用中，我们通常选择Batch-GD和SGD的折中：Mini-Batch GD

# 需要完成两个类的实例化：xxDataset 和 DataLoader
# xxDataset：构造数据集/支持索引功能/支持输出数据集的长度（继承自torch.utils.data下的Dataset类，它是一个抽象类，不能实例化，只能由其他子类继承
# DataLoader：创建Minibatch/打乱数据/并行运行。
# 原理：根据xxDataset定义的__init__进行初始化，遍历__len__长度，根据__getitem__方法定义的数据格式返回数据(tensor)并创建batch。

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 适用于数据集较小，内存可以承受的情况
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        self.data = np.loadtxt(filepath, delimiter = ',', dtype = np.float32)
        self.x_data = torch.from_numpy(self.data[:,:-1])
        self.y_data = torch.from_numpy(self.data[:,[-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        self.len = self.data.shape[0]       # 取出data矩阵的行数
        return self.len

dataset = DiabetesDataset("diabetes.csv.gz")
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(in_features=8, out_features=6)
        self.linear2 = torch.nn.Linear(in_features=6, out_features=4)
        self.linear3 = torch.nn.Linear(in_features=4, out_features=1)
        self.activate = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x

model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader,1):           # 下标从1开始
            inputs, lables = data                           # data是大小为2的list，将其分别赋值给inputs和lables

            y_pred = model(inputs)
            loss = criterion(y_pred, lables)
            print(epoch+1, i, loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()


