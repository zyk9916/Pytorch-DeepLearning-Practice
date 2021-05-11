# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

# 1.python的__call__方法：
# 对象通过提供__call__(self,*args,**kwargs)方法可以模拟函数的行为。如果一个对象x提供了该方法（称这个对象是callable，可调用的），就可以像函数一样使用它，
# 也就是说x(arg1, arg2...) 等同于调用x.__call__(self, arg1, arg2)。
class Abc():
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        print(args,kwargs)

abc = Abc()
abc(1,2,3,4,x=5,y=6)
# 结果：(1, 2, 3, 4) {'x': 5, 'y': 6}


# 2.torch.nn.Linear类：
# class torch.nn.Linear(in_features, out_features, bias=True)
# in_features:输入样本的张量大小;
# out_features:输出样本的张量大小;
# （重要！！！！！！！）in_features * out_features就是权重矩阵weight的转置的维度（重要！！！！！！！）
# bias:偏置
# 它的功能是对输入数据做一个线性变换：y = xAT + b，A为权重矩阵，b为偏置
import torch
m = torch.nn.Linear(2, 3)       # 权重矩阵为3*2，其转置为2*3
input = torch.randn(4, 2)       # 输入数据为[batch_size,in_features]=4*2
out = m(input)
print(m.weight.shape)           # 权重矩阵的维度为3*2
print(m.bias.shape)             # 偏置的维度为3，即out_features
print(out.size())               # 输出的维度为4*3,即[4*2][2*3]=[4*3]=[batch_size,out_features]

# 3.torch.nn.MSELoss类：https://pytorch.org/docs/master/generated/torch.nn.MSELoss.html
# class torch.nn.MSELoss(size_average = None, reduce = None, reduction = 'mean')
# size_average = Ture表示取各个损失的平均值，reduce = True表示将结果降维为标量
# size_average和reduce现已弃用，只需关注reduction即可
# torch.nn.MSELoss继承自nn.Module，其forward为返回2个tensor的均方误差，结果是一个标量

# 4.torch.optim.SGD类：
# class torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)
# params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
# lr (float) – 学习率
# momentum (float, 可选) – 动量因子（默认：0）
# weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认：0）
# dampening (float, 可选) – 动量的抑制因子（默认：0）
# nesterov (bool, 可选) – 使用Nesterov动量（默认：False）
# 一般只设定params和lr即可

