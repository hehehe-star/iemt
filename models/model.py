import csv
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, random_split
import geohash as gh
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import dataProcess
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----------load data----------------------
class MyTextDataset(Dataset):
    def __init__(self,filepath,coorpath, num_samples=10000, random_seed=50):
        self.num_samples = num_samples
        self.random_seed = random_seed

        self.data = np.loadtxt(filepath,delimiter=",",dtype=np.float32)
        self.coordata = np.loadtxt(coorpath,delimiter=",",dtype=np.float32)

        # 在这里实现对数据集的随机抽样并保持对应
        self.data_subset, self.coordata_subset = self._sample_corresponding_data()

    #随机采样
    def _sample_corresponding_data(self):
        positive = np.arange(5000)
        negative = np.arange(5000, 10000)
        assert len(self.data) == len(self.coordata), "The two datasets must have the same length"
        np.random.seed(self.random_seed)
        #indices = np.random.choice(len(self.data), self.num_samples, replace=False)

        positive_indices = np.random.choice(positive, self.num_samples//2, replace=False)
        negative_indices = np.random.choice(negative, self.num_samples//2, replace=False)

        indices = np.concatenate((positive_indices, negative_indices))
        #print(indices,len(indices))
        return self.data[indices], self.coordata[indices]

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.data_subset)
        # return len(self.data)


    def __getitem__(self, index):
        name_emb1 = torch.from_numpy(np.array(self.data_subset[index, 1:1025])).to(device)
        type_emb1 = torch.from_numpy(np.array(self.data_subset[index, 1025:2049])).to(device)
        #coor_emb1 = torch.from_numpy(np.array(self.data_subset[index, 2049:2084])).to(device)
        coor_emb1 = torch.from_numpy(np.array(self.coordata_subset[index, 1:49])).to(device)
        name_emb2 = torch.from_numpy(np.array(self.data_subset[index, 2084:3108])).to(device)
        type_emb2 = torch.from_numpy(np.array(self.data_subset[index, 3108:4132])).to(device)
        #coor_emb2 = torch.from_numpy(np.array(self.data_subset[index, 4132:4167])).to(device)
        coor_emb2 = torch.from_numpy(np.array(self.coordata_subset[index, 49:97])).to(device)
        label = torch.from_numpy(np.array(self.data_subset[index, [-1]])).to(device)

        return name_emb1, type_emb1, coor_emb1, name_emb2, type_emb2, coor_emb2, label


data_file = r"./data/data_process.txt"
coor_file = r"./data/data_sphere.txt"

dataset = MyTextDataset(data_file,coor_file)
print(len(dataset))

#-----------划分数据-------------------
# 定义划分比例
test_split = 0.2
val_split = 0.1
# 计算划分的样本数量
dataset_size = len(dataset)
test_size = int(test_split * dataset_size)
val_size = int(val_split * dataset_size)
train_size = dataset_size - val_size - test_size
# 根据划分比例划分数据集
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
print("train",len(train_dataset))
print("val",len(val_dataset))
print("test",len(test_dataset))


batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



#---------------model------------------------
embedding_dim = 1024
hidden_dim = 256
output_dim = 1
class Model(nn.Module):
    def __init__(self, channels=embedding_dim, r=4):
        super(Model, self).__init__()
        inter_channels = int(channels // r) # channels=embedding_dim, r=4

        #inter_channels = batch_size #channels=embedding_dim, r=4

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

        self.coor_linner = nn.Linear(48,1024)

        self.linner2083 = nn.Linear(2083,1)
        self.linner96 = nn.Linear(96,1)
        self.linner70 = nn.Linear(70,1)
        self.linner2144 = nn.Linear(2144,1)
        self.linner4166 = nn.Linear(4166,1)
        self.linner4192 = nn.Linear(4192,1)
        self.linner4096 = nn.Linear(4096,1)
        self.linner2048 = nn.Linear(2048,256)
        self.linner20482 = nn.Linear(256,1)
        self.linner20481 = nn.Linear(2048,1)
        self.relu = nn.ReLU()

        # --------------------
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)



    def forward(self, name_emb1, type_emb1, coor_emb1,name_emb2, type_emb2, coor_emb2):#结果要是一个tensor 六个输入
        # -------------aff+ topo1/2 + cat + lineer + sigmoid-----------
        topn1 = self.threeTo1(name_emb1, type_emb1, coor_emb1).squeeze()
        topn2 = self.threeTo1(name_emb2, type_emb2, coor_emb2).squeeze()
        result = torch.cat((topn1,topn2),dim=1)
        output = self.linner2048(result)
        output = self.relu(output)
        output = self.linner20482(output)
        output = self.sigmoid(output)
        return output


    def threeTo1(self, name_emb, type_emb, coor_emb):
        coor_emb = self.coor_linner(coor_emb).unsqueeze(-1).unsqueeze(-1)# 35->1024

        elom_emb = name_emb.unsqueeze(-1).unsqueeze(-1) + type_emb.unsqueeze(-1).unsqueeze(-1)
        xa = coor_emb + elom_emb

        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * coor_emb * wei + 2 * elom_emb * (1 - wei)
        return xo

    def oneFusion(self,x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei




model = Model().to(device)
#------------loss-----------------------------
# 定义损失函数和优化器
criterion = nn.BCELoss(reduction='mean').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#0.001

#-------------早停-----------------------------
best_val_loss = float('inf')
patience = 10
counter = 0
#---------------train------------------------------
num_epochs = 50

for epoch in range(num_epochs):
    train_loss = 0.0
    train_accuracy = 0
    train_total = 0
    train_correct = 0
    train_y_true = []
    train_y_pred = []
    # 迭代数据加载器

    for i, (name_emb1, type_emb1, coor_emb1,name_emb2, type_emb2, coor_emb2,label) in enumerate(train_loader):
        # 在这里实现训练逻辑
        # features 和 labels 将包含一个批次的数据
        # print(f'Batch {i}: Features shape = {features1.shape}, Labels shape = {labels.shape}')
        # 将模型切换到训练模式
        optimizer.zero_grad()

        #-----forward----------
        outputs = model(name_emb1, type_emb1, coor_emb1,name_emb2, type_emb2, coor_emb2)
        #print(outputs,label,outputs.round()==label)

        #-----loss----------
        loss = criterion(outputs, label)

        #------backward----------
        # optimizer.zero_grad()
        loss.backward()
        #------update---------
        optimizer.step()


        train_loss += loss.item()


        # 训练精度计算方法貌似不对
        # 计算精度
        predicted = outputs.round()  # 将概率转换为0或1
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()
        #print(train_correct)
        train_y_true.extend(label[i].item() for i in range(label.size(0)))
        train_y_pred.extend(predicted[i].item() for i in range(predicted.size(0)))

    epoch_loss = train_loss / len(train_loader)
    epoch_acc = train_correct / train_total * 100
    accuracy = accuracy_score(train_y_true, train_y_pred)
    precision = precision_score(train_y_true, train_y_pred)
    f1 = f1_score(train_y_true, train_y_pred)
    recall = recall_score(train_y_true, train_y_pred)
    # print(correct,total)
    # print(f'train Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f},Accuracy: {epoch_acc:.4f},recall: {recall:.4f}')

    # ---------------模型验证----------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_y_true = []
    val_y_pred = []
    with torch.no_grad():
        for i, (name_emb1, type_emb1, coor_emb1,name_emb2, type_emb2, coor_emb2,label) in enumerate(val_loader):
            val_outputs = model(name_emb1, type_emb1, coor_emb1,name_emb2, type_emb2, coor_emb2)
            val_loss = criterion(val_outputs, label)
            # print(val_outputs >= 0.99)

            val_predicted = val_outputs.round()
            #val_predicted = (val_outputs >= 0.99).int()

            val_total += label.size(0)
            val_correct += (val_predicted == label).sum().item()
            val_y_true.extend(label[i].item() for i in range(label.size(0)))
            val_y_pred.extend(val_predicted[i].item() for i in range(val_predicted.size(0)))

        epoch_loss = val_loss / len(val_loader)
        epoch_acc = val_correct / val_total * 100
        accuracy = accuracy_score(val_y_true, val_y_pred)
        precision = precision_score(val_y_true, val_y_pred)
        recall = recall_score(val_y_true, val_y_pred)
        f1 = f1_score(val_y_true, val_y_pred)
        # print(val_correct,val_total)
        print(f'val Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f},Accuracy: {epoch_acc:.2f}%,recall: {recall:.4f}')

        # # 检查是否需要早停
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     counter = 0
        #     # 保存最佳模型
        #     # torch.save(model.state_dict(), 'best_model.pt')
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print("Early stopping!")
        #         break


#--------------eval-----------------------------------
# 在测试集上进行评估
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
test_y_true = []
test_y_pred = []
with torch.no_grad():
    for i, (name_emb1, type_emb1, coor_emb1,name_emb2, type_emb2, coor_emb2,label) in enumerate(test_loader):
        test_outputs = model(name_emb1, type_emb1, coor_emb1,name_emb2, type_emb2, coor_emb2)
        test_loss = criterion(test_outputs, label)

        # test_predicted = (test_outputs >= 0.99).int()
        test_predicted = test_outputs.round()
        test_total += label.size(0)
        test_correct += (test_predicted == label).sum().item()
        test_y_true.extend(label[i].item() for i in range(label.size(0)))
        test_y_pred.extend(test_predicted[i].item() for i in range(test_predicted.size(0)))

    epoch_loss = test_loss / len(test_loader)
    epoch_acc = test_correct / test_total * 100
    accuracy = accuracy_score(test_y_true, test_y_pred)
    precision = precision_score(test_y_true, test_y_pred)
    recall = recall_score(test_y_true, test_y_pred)
    f1 = f1_score(test_y_true, test_y_pred)
    print(test_correct,test_total)
    print(f'test Precision: {precision:.8f}\ntest F1 Score: {f1:.8f}\ntest Accuracy: {epoch_acc:.4f}%\ntest recall: {recall:.8f}')

    conf_matrix = confusion_matrix(test_y_true, test_y_pred)
    # 提取 TP, FP, TN, FN
    tn, fp, fn, tp = conf_matrix.ravel()
    print("True Positives:", tp)
    print("False Positives:", fp)
    print("True Negatives:", tn)
    print("False Negatives:", fn)

torch.save(model, 'model.pth')