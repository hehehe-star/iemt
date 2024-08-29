import csv

import fasttext
import numpy as np
import sklearn.ensemble
from numpy.linalg.linalg import norm
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
model = fasttext.load_model(r"./data/cc.en.300.bin")
print(model.get_dimension())

def calculate_distance(lat1, lon1, lat2, lon2):
    # 使用Haversine公式计算球面距离
    earth_radius = 6371000  # 地球半径（米）
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = earth_radius * c
    return distance

# 定义函数计算余弦相似度
def calculate_cosine_similarity(sentence_a, sentence_b):
    vector_a = model.get_sentence_vector(sentence_a)
    vector_b = model.get_sentence_vector(sentence_b)

    similarity_score = np.dot(vector_a, vector_b) / (norm(vector_a) * norm(vector_b))
    return similarity_score

data=[]
label = []
# 读取数据
dataset = './data/data.csv'
with open(dataset) as csvfile:
    reader = csv.DictReader( csvfile, fieldnames=[ "name_A" ,"class_A" ,"latitude_A",  "longitude_A", "name_B" , "class_B" , "latitude_B",  "longitude_B", "label"], delimiter='\t' )
    for row in reader:
        name_similarity = calculate_cosine_similarity( row['name_A'] , row['name_B'] )

        class_similarity =calculate_cosine_similarity(row['class_A'] , row['class_B'])

        # location_similarity = damerau_levenshtein(str(row['latitude_A']) + str(row['longitude_A']), str(row['latitude_B']) + str(row['longitude_B']))
        location_similarity = calculate_distance(float(row['latitude_A']),float(row['longitude_A']),float(row['latitude_B']),float(row['longitude_B']))
        print(name_similarity,row['label'])
        data.append([name_similarity,class_similarity,location_similarity])
        label.append(row['label'])

# print(data)
# print(label)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=100)

# 创建随机森林分类器模型并训练
#rf_classifier = svm.LinearSVC()
rf_classifier = RandomForestClassifier(n_estimators=200)
#rf_classifier = ensemble.RandomForestClassifier()
rf_classifier.fit(X_train, y_train)



predictions = rf_classifier.predict(X_test)
# print(predictions)
# print(y_test)
precision, recall, f_score, true_sum = precision_recall_fscore_support(y_test, predictions)
print(precision_recall_fscore_support(y_test, predictions))

# 输出结果
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f_score)
# 输出准确率
accuracy = rf_classifier.score(X_test, y_test)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, predictions)
# 提取 TP, FP, TN, FN
tn, fp, fn, tp = conf_matrix.ravel()
print("True Positives:", tp)
print("False Positives:", fp)
print("True Negatives:", tn)
print("False Negatives:", fn)