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

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.convolution1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.convolution2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.maxpooling = torch.nn.MaxPool2d(kernel_size=2)
        self.linear1 = torch.nn.Linear(in_features=320, out_features=160)       # 这里要知道320怎么来的
        self.linear2 = torch.nn.Linear(in_features=160, out_features=80)
        self.linear3 = torch.nn.Linear(in_features=80, out_features=40)
        self.linear4 = torch.nn.Linear(in_features=40, out_features=10)
        self.activate = torch.nn.Tanh()
    def forward(self,x):
        x = self.activate(self.maxpooling(self.convolution1(x)))
        x = self.activate(self.maxpooling(self.convolution2(x)))
        x = x.view(-1, 320)
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = self.linear4(x)
        return x

model = Model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     #
model.to(device)                                                            #
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 1):
        inputs, lables = data
        inputs, lables = inputs.to(device), lables.to(device)               #
        outputs = model(inputs)
        loss = criterion(outputs, lables)
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
            inputs, lables = inputs.to(device), lables.to(device)           #
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)   # dim=1表示取每行的最大值，dim=0表示取每列的最大值，_是最大值，predicted是最大值的下标
            total += lables.size(0)
            correct += (predicted == lables).sum().item()   # 张量之间的比较运算
    print('Accuracy on test set: %5f %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(100):
        train(epoch)
        test()