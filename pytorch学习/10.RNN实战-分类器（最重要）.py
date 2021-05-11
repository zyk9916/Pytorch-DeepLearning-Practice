import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'         # 解决报错

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gzip
import csv
import torch
import time
import math
from torch.nn.utils.rnn import pack_padded_sequence

USE_GPU = True
batch_size = 256
hidden_size = 100
num_layers = 2
num_epochs = 100
num_chars = 128                                     # ASCII码有128种，即字母表的大小

# Preparing data
# 首先将姓名按照ASCII码映射成类别标签，在经过embedding层映射为向量
# 同样，将国家也映射为类别标签。（共18个）
class NameDataset(Dataset):
    def __init__(self, is_train_set):
        filename = 'names_train.csv.gz' if is_train_set else 'names_test.csv.gz'
        with gzip.open(filename, 'rt')as f:
            reader = csv.reader(f)
            rows = list(reader)                     # [['Adsit', 'Czech'], ['Ajdrna', 'Czech'], ['Antonowitsch', 'Czech']...]
        self.names = [row[0] for row in rows]       # ['Adsit', 'Ajdrna', 'Antonowitsch', 'Antonowitz', 'Ballalatak'...]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = sorted(set(self.countries)) # ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French'...]
        self.country_dict = self.getCountryDict()   # {'Arabic': 0, 'Chinese': 1, 'Czech': 2, 'Dutch': 3, 'English': 4,...}
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]  # 返回的数据格式：tuple('Arabic',2)，即tuple(姓名，国籍对应类标）

    def __len__(self):
        return self.len

    def getCountryDict(self):
        country_dict = {}
        for idx, country_name in enumerate(self.country_list):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self, index):
        return self.country_list[index]

    def getCountriesNum(self):
        return self.country_num

trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
testset = NameDataset(is_train_set=False)
testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

num_countries = trainset.getCountriesNum()                  # 18

class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # 由于是对每个字符的ASCII值做embedding，因此有多少种ASCII值，就有多少种embedding，即使有些ASCII值可能没有出现
        # 将每个ASCII值都embedding为hidden_size维度（也可以是其他size）
        self.embedding = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                                bidirectional=bidirectional)
        self.linear = torch.nn.Linear(in_features=self.num_directions*hidden_size, out_features=output_size)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):          # seq_lengths是各个序列的长度的列表
        input = input.t()                           # input维度由(batch_size*seq_len)转为(seq_len*batch_size)，seq_len是最长长度
        batch_size = input.size(1)
        hidden = self.init_hidden(batch_size)
        embedding = self.embedding(input)           # embedding后维度：(seq_len,batch_size,hidden_size)
        gru_input = pack_padded_sequence(embedding, seq_lengths)        # 打包，提高计算效率
        output, hidden = self.gru(gru_input, hidden)                    # 使用hidden，舍弃output
        # output维度(seq_len,batch_size,hidden_size*num_directions)
        # hidden维度(num_layers*num_directions,batch_size,hidden_size)，这里是(4,batch_size,hidden_size)
        if self.num_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)     # hidden[-1]和hidden[-2]的维度是(batch_size,hidden_size)
                                                                        # 将其按照hidden_size维度进行拼接得到hidden_cat
                                                                        # hidden_cat维度为(batch_size,num_directions*hidden_size)
                                                                        # 接下来就可以放入linear层
        else:
            hidden_cat = hidden[-1]
        linear_output = self.linear(hidden_cat)                         # 维度是(batch_size,output_size)，即(256,18)
        return linear_output

# 返回一个name中各字符ASCII码值的列表及列表长度所构成的元组
def name2list(name):
    arr = [ord(c) for c in name]        # ord(c)是字符c的ASCII值
    return arr, len(arr)

def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor

# 将数据集中的数据预处理成神经网络可以接收的形式
# 将每个name转化为一个由其各个字符对应的ASCII值所构成的向量，每个country转化为其所对应的的类标（country的转化已经由DataLoader完成）
def make_tensors(names, countries):         # names,countries分别是一个batch中的姓名组成的列表，国籍对应类标组成的列表的tensor
    sqeuences_and_lengths = [name2list(name) for name in names]
    names_sequences  = [t[0] for t in sqeuences_and_lengths]                    # 得到一个batch中所有name的ASCII码值列表
    seq_lengths = torch.LongTensor([t[1] for t in sqeuences_and_lengths])       # 得到一个batch中所有name的序列长度
    countries = countries.long()                                                # 将countries转化为LongTensor

    # 构建name对应的tensor,维度为(batch_size*seq_len),seq_len是一个batch中最长name的长度
    seq_tensor = torch.zeros(len(names_sequences), int(seq_lengths.max().item())).long()    # 先构建全0tensor，再填充
    for idx, (seq, seq_len) in enumerate(zip(names_sequences, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.Tensor(seq)                  # 将第idx行的0前seq_len个0填充为ASCII值

    # 按照序列长度进行排序，为使用pack_padded_sequence做准备
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True) # tensor的sort方法返回两个值，一是排序后的tensor，二是对应的索引(也构成一个tensor)
    seq_tensor = seq_tensor[perm_idx]                         # 按照顺序将seq_tensor重新排列
    countries = countries[perm_idx]
    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)

# 返回一个字符串，显示距离since时刻，已经过去了几分几秒
def time_since(since):
    seconds = time.time() - since
    minutes = math.floor(seconds / 60)      # 向下取整
    seconds -= minutes * 60
    return '%dm %ds' % (minutes, seconds)

def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = make_tensors(names, countries)
        output = classifier(inputs, seq_lengths.cpu())
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}] ', end='')
            print(f'loss={total_loss / (i*len(inputs))}')
    return total_loss

def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths.cpu())
            pred = output.max(dim=1, keepdim=True)[1]                   # ???????
            correct += pred.eq(target.view_as(pred)).sum().item()       # ????
        percent = '%.2f' % (100 * correct / total)
        print(f'Test set:Accuracy {correct}/{total} {percent}%')
    return correct / total

if __name__ == '__main__':
    classifier = RNNClassifier(input_size=num_chars, hidden_size=hidden_size, output_size=num_countries,
                               num_layers=num_layers, bidirectional=True)
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    start = time.time()
    print("Training for %d epochs..." % num_epochs)
    acc_list = []
    for epoch in range(1, num_epochs+1):
        trainModel()
        acc = testModel()
        acc_list.append(acc)


# 绘图
import matplotlib.pyplot as plt
import numpy as np

epoch = np.arange(1, len(acc_list)+1, 1)
acc_list = np.array(acc_list)
plt.plot(epoch, acc_list)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.show()