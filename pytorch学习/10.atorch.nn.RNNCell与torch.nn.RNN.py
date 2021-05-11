# 弄清RNN中各个张量的维度至关重要，最好结合截图理解！！！

# RNN Cell实现
import torch

batch_size = 3
seq_len = 5
input_size = 4
hidden_size = 2

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

data = torch.randn(seq_len, batch_size, input_size)     # 注意data的第一个维度是seq_len
hidden = torch.zeros(batch_size, hidden_size)           # h0初始化为全0 tensor

# 按照时间序列依次输入x1,x2,x3,x4,x5，得到h1,h2,h3,h4,h5
# 在使用RNNCell时，是自己写循环，所以每个输入tensor xi(input)的维度为(batch_size*input_size)
# 同时，这样定义也只有一层RNN，所以输出tensor hi(hidden)的维度为(batch_size*hidden_size)
# 这与直接使用torch.nn.RNN是截然不同的
for idx, input in enumerate(data):      # 注意data的第一个维度是seq_len，也就是按照时序进行批次训练。
    print('='*20,idx+1,'='*20)
    print("input size:",input.shape)
    print("input",idx+1,":",input)
    hidden = cell(input, hidden)
    print("ouput size:",hidden.shape)
    print("hidden",idx+1,':',hidden)


# RNN网络实现
print('='*20 + "torch.nn.RNN" + '='*20)
batch_size = 3
seq_len = 5
input_size = 4
hidden_size = 2
num_layers = 3

inputs = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)
# 在torch.nn.RNN中，不需要自己写循环，因此网络的输入不再是将xi挨个输入，而是输入整个序列，再由网络自动按时序输入。
# 所以这里输入tensor(inputs)的维度为(seq_len*batch_size*input_size)
# 同时，torch.nn.RNN需要指定num_layers，即每个xi经过num_layers个RNNCell，这就需要num_layers个前一时刻的隐状态
# 所以这里每个hidden的维度为(num_layers*batch_size*hidden_size)
# out是h1,h2,...hn的集合，维度为(seq_len*batch_size*hidden_size)
RNNnet = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
out, hidden = RNNnet(inputs, hidden)            # 这里第一个hidden是RNN网络最终的输出hn
print("output size:",out.shape)
print("hidden size:",hidden.shape)

# CLASS torch.nn.RNN(*args, **kwargs)
# input_size – The number of expected features in the input x
# hidden_size – The number of features in the hidden state h
# num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
# nonlinearity – The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
# bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
# batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
# dropout – If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
# bidirectional – If True, becomes a bidirectional RNN. Default: False
# 如果将batch_first置为True，将由(seq_len*batch_size)变为(batch_size*seq_len)