# 训练一个RNN模型，使其学习："hello"--→"ohlol"  （一个字符级的自然语言处理问题）

import torch
# 首先将每个字符转换为one-hot向量，定义每个字符的类别 e:0,h:1,l:2,o:3
idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]
one_hot_lookup = [[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]
x_one_hot = [one_hot_lookup[i] for i in x_data]

input_size = 4
hidden_size = 4                                                             # 分类类别共有4种
batch_size = 1
num_layers = 2
seq_len = 5
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)           # 显然seq_len=5，(5*1*4)
labels = torch.LongTensor(y_data)                                           # 注意这里的改动

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.RNN = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
    def forward(self, input, hidden):
        outputs, _ = self.RNN(input, hidden)                                               # (5*1*4)
        outputs = outputs.view(self.seq_len*self.batch_size, self.hidden_size)             # reshape为(5*4)，注意这里的改动
        return outputs
    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.1)

for epoch in range(30):
    hidden = model.init_hidden()
    outputs = model(inputs, hidden)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)

    _, idx = torch.max(outputs, dim=1)
    idx = idx.data.numpy()
    print("Predicted string:", ''.join([idx2char[x] for x in idx]), end='')
    print(",epoch:%d/30,loss=%4f" % (epoch + 1, loss.item()))

    loss.backward()
    optimizer.step()