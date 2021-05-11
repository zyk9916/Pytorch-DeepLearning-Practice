# https://www.bilibili.com/video/BV1Y7411d7Ys?p=10    反复观看！！！

# 对于一张图片的卷积过程：
# 输入：batch_size * n * w1 * h1(图片数*通道数*宽度*高度)
# 输出：batch_size * m * w2 * h2(图片数*通道数*宽度*高度）
# 则卷积设置：m * n * w3 * h3(filter个数*每个filter卷积核个数*卷积核宽度*卷积核高度，w3和h3由w1 h1 w2 h2以及步长决定，宽度和高度一般情况下相等)
# 输入通道数和卷积核个数相等，经过一个filter卷积后，各个卷积核对应值相加，构成一个1*w2*h2的输出。
# 所以，如果输出通道数为m，就需要m个filter进行卷积。

# CLASS torch.nn.Conv1d(in_channels, out_channels, kernel_size,
#                       stride=1, padding=0, dilation=1, groups=1,
#                       bias=True, padding_mode='zeros')
# stride：步长，默认为1
# padding：控制padding_mode的数目（在数据周围添加几圈）
# padding_mode：填充模式，默认为0填充
# dilation：控制kernel点的间距，默认值:1；
# group：控制分组卷积，默认不分组（即1组）
# bias：在输出中添加一个可学习的偏差。默认为True。

# 卷积操作使用举例：
import torch
in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
batch_size = 5

input = torch.randn(batch_size, in_channels, width, height)     # 生成一个服从正态分布的batch_size*in_channels*width*height的tensor
conv_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

output = conv_layer(input)

print(input.shape)                  # torch.Size([5, 5, 100, 100])
print(output.shape)                 # torch.Size([5, 10, 98, 98])
print(conv_layer.weight.shape)      # torch.Size([10, 5, 3, 3])

# 最大池化操作（最常用）
# CLASS torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
input = [3,4,6,5,
         2,4,6,8,
         1,6,7,8,
         9,7,4,6]
input = torch.Tensor(input).view(1,1,4,4)
maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)   # 设置kernel_size=2时，stride步长也会被设为2。有些情况下需要人为设置步长
output = maxpooling_layer(input)
print(output)                           # tensor([[[[4., 8.],
                                        #           [9., 8.]]]])
print(maxpooling_layer.stride)          # 2