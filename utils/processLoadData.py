
import pandas as pd
import geohash as gh
import torch
import numpy as np
from torch.utils.data import Dataset
from allennlp.modules.elmo import Elmo, batch_to_ids


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
geohash_precision = 7

# -------------------elmoEembed生成词向量------------------------
def elmoEembed(sentence_lists):
    if ' ' in sentence_lists:
        string_list = sentence_lists.split()
    else:
        string_list = [sentence_lists]

    options_file = "./resorce/options.json" # 配置文件地址
    weight_file = "./resorce/weights.hdf5" # 权重文件地址
    elmo = Elmo(options_file, weight_file, 1, dropout=0).to(device)

    character_ids = batch_to_ids([string_list]).to(device)

    embeddings = elmo(character_ids)['elmo_representations'][0].to(device)

    aver = torch.mean(embeddings, dim=1).squeeze().to(device)
    return aver
# -------------------数据编码函数------------------------
def AllEembed(point):
    # point:list[id,name1,type1,latitude1,longitude1,name2,type2,latitude2,longitude2,label]
    id = torch.tensor(point[0], dtype=torch.int32).view(1).to(device)
    name_emb1 = elmoEembed(point[1]).to(device)
    type_emb1 = elmoEembed(point[2]).to(device)
    space_geohash1 = gh.bin_geohash(point[3],point[4],7)
    space_geohash1=torch.Tensor(space_geohash1).to(device)
    name_emb2 = elmoEembed(point[5]).to(device)
    type_emb2 = elmoEembed(point[6]).to(device)
    space_geohash2 = gh.bin_geohash(point[7],point[8],7)
    space_geohash2=torch.Tensor(space_geohash2).to(device)
    label = torch.tensor(point[9], dtype=torch.float32).view(1).to(device)
    result = [id,name_emb1,type_emb1,space_geohash1,name_emb2,type_emb2,space_geohash2,label]
    return result

#------------------读取数据生成编码--------------------------------

class MyTextDataset(Dataset):
    def __init__(self,filepath):
        self.data = pd.read_csv(filepath)

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        point = data.values.tolist()
        result = AllEembed(point)
        id = result[0]
        name_emb1 = result[1]
        type_emb1 = result[2]
        coor_emb1 = result[3]
        name_emb2 = result[4]
        type_emb2 = result[5]
        coor_emb2 = result[6]
        label = result[7]

        return id, name_emb1, type_emb1, coor_emb1,name_emb2, type_emb2, coor_emb2, label

dataset = MyTextDataset("./data/data.csv")
print(dataset[0])
#----------------保存编码到文件------------------------------------------
file_path = './data/data_process.txt'
with open(file_path, 'w') as f:
    count = 0
    for index in range(0,len(dataset)+1):
        data_str = ",".join([",".join([str(num) for num in item.detach().cpu().numpy()]) for item in dataset[index]])
        np.savetxt(f, [data_str], fmt="%s")

        count += 1
        # 每10次循环保存一次数据到文件
        if count % 10 == 0:
            print('保存数据到文件:[{}/{}]'.format(count, str(len(dataset))))
            f.flush()

    if count % 10 != 0:  # 如果数据总数不能被10整除，那么最后肯定有未保存的数据
        print('保存数据到文件:[{}/{}]'.format(count, len(dataset)))
    f.flush()  # 最后一次保存操作