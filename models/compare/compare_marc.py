import torch
import geohash as gh
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义嵌入层
num_embeddings = 100000  # 假设有1000个不同的单词
embedding_dim = 100  # 将每个单词表示为100维的向量
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)

# 输入数据
input_data = torch.LongTensor([99999])  # 传入几个示例的单词索引

# 前向传播
embedded_data = embedding_layer(input_data)

print(embedded_data)
print(embedded_data.shape)  # 输出形状为 (4, 100)，表示4个示例，每个示例被表示为100维的向量


import pandas as pd
from torch.utils.data import Dataset

file = r'.\data\data.csv'
# 读取数据
data = pd.read_csv(file, sep=',', header=None)
data.columns = ['id', 'nameA', 'classA', 'latA', 'longA', 'nameB', 'classB', 'latB', 'longB', 'label']

# 提取所有nameA、nameB、classA和classB
name_vocab = []
name_vocab.extend(data['nameA'].values)
name_vocab.extend(data['nameB'].values)
class_vocab = []
class_vocab.extend(data['classA'].values)
class_vocab.extend(data['classB'].values)

# 建立单词索引
name_to_index = {word: idx for idx, word in enumerate(set(name_vocab))}
class_to_index = {word: idx for idx, word in enumerate(set(class_vocab))}
# print(name_to_index)
print(class_to_index)
# print(len(name_to_index))
# print(len(class_to_index))
#------------------加载数据------------------------------------------------------
class MyTextDataset(Dataset):
    def __init__(self, filepath,name_to_index,class_to_index):
        self.data = pd.read_csv(filepath, sep=',', header=0)
        self.name_to_index = name_to_index
        self.class_to_index = class_to_index

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.data)

    def __getitem__(self, index):
        name_emb1 = torch.LongTensor([self.name_to_index.get(self.data.loc[index, 'name_A'])]).to(device)
        type_emb1 = torch.LongTensor([self.class_to_index.get(self.data.loc[index, 'class_A'])]).to(device)
        coor_emb1 = torch.Tensor(gh.bin_geohash(self.data.loc[index, 'latitude_A'],self.data.loc[index, 'longitude_A'],8)).to(device)

        name_emb2 = torch.LongTensor([self.name_to_index.get(self.data.loc[index, 'name_B'])]).to(device)
        type_emb2 = torch.LongTensor([self.class_to_index.get(self.data.loc[index, 'class_B'])]).to(device)

        coor_emb2 = torch.Tensor(gh.bin_geohash(self.data.loc[index, 'latitude_B'],self.data.loc[index, 'longitude_B'],8)).to(device)
        label = torch.Tensor(self.data.loc[index, 'label']).to(device)

        return name_emb1, type_emb1, coor_emb1, name_emb2, type_emb2, coor_emb2, label


dataset = MyTextDataset(file,name_to_index,class_to_index)
print(len(dataset))
print(dataset[1])


#---------------------------模型---------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

# 创建深度学习模型
class YourModel(nn.Module):
    def __init__(self, name_vocab_size,class_vocab_size, embedding_dim):
        super(YourModel, self).__init__()

        self.embedding_name = nn.Embedding(name_vocab_size, embedding_dim)
        self.embedding_class = nn.Embedding(class_vocab_size, embedding_dim)
        self.embedding_coor = nn.Embedding(40, embedding_dim)

        # 其他自定义网络层
        # ...

    def forward(self, nameA, classA, nameB, classB):
        embedded_nameA = self.embedding_name(nameA)
        embedded_classA = self.embedding_class(classA)
        embedded_nameB = self.embedding_name(nameB)
        embedded_classB = self.embedding_class(classB)

        # 继续编写模型的forward方法

        return output

# 定义模型超参数
name_vocab_size = len(name_to_index)
class_vocab_size = len(class_to_index)

embedding_dim = 100

# 初始化模型
model = YourModel(name_vocab_size,class_vocab_size, embedding_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)