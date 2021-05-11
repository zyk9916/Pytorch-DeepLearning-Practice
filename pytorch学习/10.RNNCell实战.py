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
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)           # 显然seq_len=5，(5*1*4)
labels = torch.LongTensor(y_data).view(-1,batch_size)                       # 显然seq_len=5，(5*1)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)
    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden
    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)

model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(30):
    loss = 0
    optimizer.zero_grad()
    hidden = model.init_hidden()
    print("Predicted string:", end='')
    for input, label in zip(inputs, labels):
        hidden = model(input, hidden)
        loss += criterion(hidden, label)             # 注意这里loss不加item()，因为需要累加sel_len中每一项的损失，是需要构建计算图的
        _, idx = torch.max(hidden, dim=1)
        print(idx2char[idx.item()], end='')
    loss.backward()
    optimizer.step()
    print(",epoch:%d/30,loss=%4f"%(epoch+1,loss.item()))