# 使用softmax求损失的过程中，需要经历以下几步：
# 1.应用softmax函数；    2.取对数；    3.计算损失
# 其中第1、2步可以使用torch.nn.LogSoftmax()完成，第3步可以使用torch.nn.NLLLoss完成
# 也可以使用torch.nn.CrossEntropyLoss一次性完成3步操作，构成一个完整的softmax分类器。（使用最多！！！）

# CLASS torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduction='mean')

# 在使用torch.nn.CrossEntropyLoss之前，最后一层不再需要激活函数（无需非线性变换）
# 在使用torch.nn.CrossEntropyLoss时，样本的真实类别Y应是一个LongTensor，值为各样本的类别标号。

# 使用举例：
import torch
criterion = torch.nn.CrossEntropyLoss()

Y = torch.LongTensor([2,0,1])               # 3个样本是真实标记分别为2,0,1
Y_pred1 = torch.Tensor([[0.1,0.2,0.9],
                        [1.1,0.1,0.2],
                        [0.2,2.1,0.1]])     # 注意这里没有经过激活函数（即softmax函数），而是直接扔进CrossEntropyLoss里
Y_pred2 = torch.Tensor([[0.8,0.2,0.3],
                        [0.2,0.3,0.5],
                        [0.2,0.2,0.5]])
loss1 = criterion(Y_pred1,Y)                # 注意这里要把Y_pred1放前面，否则会报错
loss2 = criterion(Y_pred2,Y)                # 注意这里要把Y_pred2放前面，否则会报错
print("loss1 = ", loss1, "loss2 = ", loss2)