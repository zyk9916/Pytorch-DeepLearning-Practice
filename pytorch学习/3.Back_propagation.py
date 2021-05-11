# 由于深度学习网络按层深入，层层嵌套的特点，对深度网络目标函数计算梯度的时候，需要用反向传播的方式由深到浅倒着计算以及更新参数。
# 所以反向传播法是梯度下降法在深度网络上的具体实现方式。
# 在神经网络中，需要先前馈计算目标函数，再反馈利用梯度下降算法优化参数。
# 在pytorch中，无需显式地计算导数，而是应该将重点放在计算图的构造上。

# 张量是pytorch中最重要的概念
# 通过torch.tensor创建的张量称为叶子张量，叶子张量的requires_grad属性可以人为修改，而grad_fn属性为None
# 通过其他张量衍生的张量，其requires_grad属性不能人为更改（但也有可能为True或False，取决于其父张量的类型）。grad_fn属性记录了该张量是怎么来的
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])     # 创建张量w，注意必须是以sequence的形式传入
w.requires_grad = True      # 张量默认不需要计算梯度，如果需要计算梯度，需要将requires_grad置为True。
                            # 这样，计算图中会自动保留目标函数对w的导数）

def forward(x):
    return x * w                # 前馈，构建计算图

def loss(x, y):
    y_pred = forward(x)         # 前馈，构建计算图
    return (y_pred - y) ** 2    # 前馈，构建计算图

for epoch in range(100):
    print("epoch:",epoch+1)
    for x, y in zip (x_data, y_data):
        l = loss(x, y)                  # 前馈，构建计算图，注意l必须是一个标量，才可以反馈求梯度
        l.backward()                    # 反馈，自动地求出梯度，无需人工写出解析式。这里就是自动将w的梯度存到w的grad属性中，然后释放计算图
        # print(w.grad)                 # 输出的是一个tensor，例如:tensor([-2800.])
        print(w.grad.item())            # .item()可以对张量取值，输出的是w的梯度的值
        # w.grad == w.grad.data         # 结果为True
        w.data -= 0.01 * w.grad.data    # 注意这里，要取w.grad.data，而不要写成w.grad，否则也会构建计算图
        w.grad.data.zero_()             # grad是会被累加的，所以更新w后要对grad进行清零
    print("w:",w.data.item(),"loss:",l.item())