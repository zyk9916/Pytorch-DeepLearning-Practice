# 使用torchvision.datasets.MNIST图像数据集训练模型，并衡量准确率
# 该数据集是阿拉伯数字0-9的图像，共10个类别
# torchvision.datasets下的所有数据集都是torch.utils.data.Dataset的子类，都有__init__,__getitem__,__len__魔法方法，也可以使用多线程。

import torch
from torchvision import transforms          # 数据预处理
from torchvision import datasets            # 数据集获取
from torch.utils.data import DataLoader

# 针对MNIST图像数据集的预处理，不用深究
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
# 分别读取训练集、测试集
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=64)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=64)                # 测试集无需打乱顺序

# 在数据预处理之后，整个数据集是N*1*28*28的张量（N张图片，1个通道，28*28个像素）
# 而全连接网络只接受2维矩阵输入，所以需要使用view()方法将其转化为N*784的张量（784=1*28*28）
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(in_features=784,out_features=512)
        self.linear2 = torch.nn.Linear(in_features=512,out_features=256)
        self.linear3 = torch.nn.Linear(in_features=256,out_features=128)
        self.linear4 = torch.nn.Linear(in_features=128,out_features=64)
        self.linear5 = torch.nn.Linear(in_features=64,out_features=10)
        self.activate = torch.nn.ReLU()
    def forward(self,x):
        x = x.view(-1,784)                  # 转化为2维张量，列数为784，行数根据列数自动计算
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = self.activate(self.linear4(x))
        x = self.linear5(x)
        return x

model = Model()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)       # 这里使用了带冲量的SGD以提高性能

# 由于需要评估模型（即测试），最好把train和test过程分别集成成2个函数
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 1):
        inputs, lables = data                           # 真实标记维度:64
        outputs = model(inputs)                         # 输入维度:64*784,输出维度:64*10
        loss = criterion(outputs, lables)               # CrossEntropyLoss会自动将两个不同维度的tensor转换为分布，计算交叉熵损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:                          # 每300次迭代（即训练300个batch），输出一次Loss
            print('[%d,%5d] loss: %3f' % (epoch+1, batch_idx+1, running_loss))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():                                   # 在测试过程中无需计算梯度
        for data in test_loader:
            inputs, lables = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)   # dim=1表示取每行的最大值，dim=0表示取每列的最大值，_是最大值，predicted是最大值的下标
            total += lables.size(0)
            correct += (predicted == lables).sum().item()   # 张量之间的比较运算
    print('Accuracy on test set: %5f %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(100):
        train(epoch)
        test()
