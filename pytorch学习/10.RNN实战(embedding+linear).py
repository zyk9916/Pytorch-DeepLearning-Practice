import torch
# 定义每个字符的类别 e:0,h:1,l:2,o:3
idx2char = ['e', 'h', 'l', 'o']
x_data = torch.LongTensor([1, 0, 2, 2, 3])
y_data = torch.LongTensor([3, 1, 2, 3, 2])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.seq_len = 5
        self.batch_size = 1
        self.num_embeddings = 4             # 字符类型有4种，所以需要4种embedding
        self.embedding_dim = 3              # 将字符映射为3维向量
        self.hidden_size = 5                # 将hidden_size设为5，再经过linear层变为4维，进入softmax
        self.num_layers = 2
        self.num_classes = 4                # 4个类别的分类任务
        self.embedding = torch.nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)
        self.rnn = torch.nn.RNN(input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.linear = torch.nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)
    def forward(self, inputs, hidden):
        embeddings = self.embedding(inputs)                                                     # (5*3)
        rnn_inputs = embeddings.view(self.seq_len, self.batch_size, self.embedding_dim)         # (5*1*3)
        rnn_outputs, _ = self.rnn(rnn_inputs, hidden)                                           # (5*1*5)
        outputs = self.linear(rnn_outputs)                                                      # (5*1*4)
        outputs = outputs.view(self.seq_len*self.batch_size, self.num_classes)                  # (5*4)
        return outputs
    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(30):
    hidden = model.init_hidden()
    outputs = model(x_data, hidden)
    optimizer.zero_grad()
    loss = criterion(outputs, y_data)
    _, idx = torch.max(outputs, dim=1)
    idx = idx.data.numpy()
    print("Predicted string:", ''.join([idx2char[x] for x in idx]), end='')
    print(",epoch:%d/30,loss=%4f" % (epoch + 1, loss.item()))

    loss.backward()
    optimizer.step()