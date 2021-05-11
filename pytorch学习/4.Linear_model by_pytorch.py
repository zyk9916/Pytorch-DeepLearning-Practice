# 构建模型主要分为4个步骤：
# 1.准备数据集
# 2.设计计算y_hat的模型（nn.Module)
# 3.构造loss和optimizer
# 4.设计训练过程（前馈，反馈，更新）

import torch
print("++++++++++Training on gpu++++++++++")

# 1.Dataset
# 在这个例子中，输入和输出都采用向量形式
x_data = torch.tensor([[1.0], [2.0], [3.0]])        # [batch_size * in_features] = [3 * 1]
y_data = torch.tensor([[2.0], [4.0], [6.0]])        # [batch_size * out_features] = [3 * 1]

# 2.Model
class LinearModel(torch.nn.Module):             # nn.Module是所有神经网络模型的基础类。nn下的几乎所有类都继承自nn.Module
    def __init__(self):
        super(LinearModel,self).__init__()      # 无脑写上这一行即可
        self.linear = torch.nn.Linear(in_features=1, out_features=1,bias=True)     # nn.Linear也继承自nn.Module
                                                                                    # in_features和out_features分别是input sample
                                                                                    # 和output sample的维度
    # 在torch.nn.Module类中，其__call__方法调用了forward函数。
    # 因此，我们在构建model时，必须重写forward函数
    def forward(self, x):
        y_pred = self.linear(x)                 # 由于linear是callable的，可以像调用函数一样调用它
        return y_pred

model = LinearModel()                           # model是callable的，它包括了一个linear实例

# 3.Loss and Optimizer
criterion = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.02)      # model.parameters()包含了linear的参数

# 4.Traning cycle
for epoch in range(1000):
    y_pred = model(x_data)                    # 调用model实例（自动call了forward方法）
    loss = criterion(y_pred, y_data)          # 调用criterion实例（自动call了forward方法），得到的loss是一个标量
    print(epoch+1, loss)                      # loss是一个标量，不会再创建计算图
    optimizer.zero_grad()                     # 无脑写：权重归零
    loss.backward()                           # 无脑写：反向传播，计算梯度
    optimizer.step()                          # 无脑写：根据计算出的梯度和学习率自动更新model.parameters

# 打印结果：
print("w = :", model.linear.weight.item())
print("b = :", model.linear.bias.item())

# test:
x_test = torch.tensor([4.0])
y_test = model(x_test)
print("y_pred = :",y_test)
