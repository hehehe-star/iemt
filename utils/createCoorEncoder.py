import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import torch.optim as optim
from SpatialRelationEncoder import *

batch_size = 1
num_context_pt = 1

def coorEncoder(coords):
    model = SphereSpatialRelationEncoder(spa_embed_dim=1024, frequency_num=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    result = model(coords)
    return result.squeeze()


def readfile(file_root, output):
    with open(file_root, 'r') as csvfile:
        df = pd.read_csv(csvfile)
        selected_columns = df.iloc[:, [0, 3, 4, 7, 8]]
        result = []
        for index, row in selected_columns.iterrows():
            columnID = int(row[0])
            # 合并坐标A为一个列表
            columnCoor1 = np.array([row[1], row[2]])
            coor1 = coorEncoder(columnCoor1)
            # 合并坐标B为另一个列表
            columnCoor2 = np.array([row[3], row[4]])
            coor2 = coorEncoder(columnCoor2)
            combined_tensor = torch.cat((coor1, coor2), dim=0)
            numpy_array = combined_tensor.cpu().numpy()

            # 将columnID和numpy_array合并成一行，以逗号分隔
            combined_row = str(columnID) + ',' + ','.join(map(str, numpy_array))
            print(combined_row)

            result.append(combined_row)

            # np.savetxt('combined.txt', numpy_array, delimiter=',', fmt='%d')
        np.savetxt(output, result, delimiter=',', fmt='%s')


file_root = r"./data/data.csv"
output = r"./data/data_sphere.txt"
readfile(file_root, output)
