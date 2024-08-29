# -*- coding: utf-8 -*-
import csv
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import xgboost
from sklearn import svm, ensemble
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


class_to_index = {'canal': 0, 'rock': 1, 'reservoir': 2, 'prison': 3, 'christian catholic': 4, 'estate': 5, 'plain': 6, 'island': 7, 'vending any': 8, 'cemetery': 9, 'toilet': 10, 'agricultural school': 11, 'market': 12, 'archaeological': 13, 'church': 14, 'lake': 15, 'road': 16, 'hill': 17, 'facility': 18, 'beverages': 19, 'forest': 20, 'greengrocer': 21, 'meteorological station': 22, 'amphitheater': 23, 'cafe': 24, 'swimming pool': 25, 'volcano': 26, 'airbase': 27, 'power station': 28, 'zoo': 29, 'chemist': 30, 'clothes': 31, 'islands': 32, 'shoal': 33, 'waterfall': 34, 'police post': 35, 'christian evangelical': 36, 'viewpoint': 37, 'fort': 38, 'casino': 39, 'distributary': 40, 'kindergarten': 41, 'motel': 42, 'headland': 43, 'beauty shop': 44, 'shelter': 45, 'nightclub': 46, 'airfield': 47, 'bus stop': 48, 'wharf': 49, 'section of stream': 50, 'anchorage': 51, 'chalet': 52, 'bicycle shop': 53, 'fire station': 54, 'comms tower': 55, 'motorway junction': 56, 'police': 57, 'section of populated place': 58, 'valley': 59, 'camera surveillance': 60, 'mountains': 61, 'tourist info': 62, 'car wash': 63, 'courthouse': 64, 'bus station': 65, 'sewage treatment plant': 66, 'locality': 67, 'doityourself': 68, 'castle': 69, 'monastery': 70, 'class_A': 71, 'playground': 72, 'town': 73, 'bicycle rental': 74, 'sugar mill': 75, 'stationery': 76, 'bench': 77, 'jeweller': 78, 'optician': 79, 'ferry terminal': 80, 'cove': 81, 'land-tied island': 82, 'pass': 83, 'florist': 84, 'caravan site': 85, 'hunting stand': 86, 'point': 87, 'pub': 88, 'artwork': 89, 'biergarten': 90, 'embassy': 91, 'harbor': 92, 'bookshop': 93, 'tree': 94, 'village': 95, 'hamlet': 96, 'sulphur spring': 97, 'bridge': 98, 'post box': 99, 'dentist': 100, 'library': 101, 'parking': 102, 'car rental': 103, 'populated locality': 104, 'camp site': 105, 'guesthouse': 106, 'city': 107, 'suburb': 108, 'taxi': 109, 'railroad station': 110, 'transit terminal': 111, 'ice rink': 112, 'pitch': 113, 'swamp': 114, 'university': 115, 'sawmill': 116, 'hospital': 117, 'computer shop': 118, 'bay': 119, 'stream mouth': 120, 'community centre': 121, 'museum': 122, 'populated place': 123, 'office building': 124, 'strait': 125, 'marina': 126, 'veterinary': 127, 'hairdresser': 128, 'theatre': 129, 'bar': 130, 'garden centre': 131, 'isthmus': 132, 'peak': 133, 'pharmacy': 134, 'tidal creek': 135, 'shoe shop': 136, 'hotel': 137, 'doctors': 138, 'restaurant': 139, 'laundry': 140, 'rocks': 141, 'park': 142, 'lagoon': 143, 'stadium': 144, 'drinking water': 145, 'street lamp': 146, 'cave': 147, 'attraction': 148, 'cave entrance': 149, 'inlet': 150, 'peninsula': 151, 'railway station': 152, 'memorial': 153, 'water works': 154, 'cliff': 155, 'car dealership': 156, 'S.RDIN': 157, 'mission': 158, 'administrative facility': 159, 'sound': 160, 'public building': 161, 'athletic field': 162, 'fuel': 163, 'department store': 164, 'graveyard': 165, 'mobile phone shop': 166, 'sea': 167, 'water well': 168, 'christian': 169, 'port': 170, 'theme park': 171, 'airport': 172, 'mountain': 173, 'general': 174, 'intermittent stream': 175, 'tower': 176, 'golf course': 177, 'wayside shrine': 178, 'sports shop': 179, 'traffic signals': 180, 'dam': 181, 'factory': 182, 'post office': 183, 'region': 184, 'cultivated area': 185, 'communication center': 186, 'building': 187, 'school': 188, 'food court': 189, 'town hall': 190, 'bank': 191, 'gift shop': 192, 'nursing home': 193, 'reef': 194, 'christian methodist': 195, 'lighthouse': 196, 'intermittent lake': 197, 'muslim sunni': 198, 'street': 199, 'watercourse': 200, 'pier': 201, 'S.TOLL': 202, 'hostel': 203, 'alpine hut': 204, 'ruins': 205, 'kiosk': 206, 'recycling': 207, 'marine channel': 208, 'christian protestant': 209, 'landfill': 210, 'jetty': 211, 'resort': 212, 'clinic': 213, 'cape': 214, 'spring': 215, 'cinema': 216, 'mill': 217, 'military installation': 218, 'wayside cross': 219, 'oil refinery': 220, 'furniture shop': 221, 'monument': 222, 'class_B': 223, 'ridge': 224, 'dike': 225, 'convenience': 226, 'mall': 227, 'wildlife reserve': 228, 'bakery': 229, 'butcher': 230, 'hills': 231, 'fast food': 232, 'college': 233, 'market place': 234, 'sports centre': 235, 'camp': 236, 'aquaculture facility': 237, 'toy shop': 238, 'supermarket': 239, 'travel agent': 240, 'farm': 241, 'mine': 242, 'dog park': 243, 'outdoor shop': 244, 'atm': 245, 'weir': 246, 'crossing': 247, 'stream': 248, 'beach': 249, 'technical school': 250, 'picnic site': 251}


def calculate_distance(lat1, lon1, lat2, lon2):
    earth_radius = 6371000
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = earth_radius * c
    return distance

from datasetcreator import damerau_levenshtein, jaccard, jaro, jaro_winkler,monge_elkan, cosine, strike_a_match, soft_jaccard, sorted_winkler, permuted_winkler, skipgram, davies
data=[]
label = []
# 读取数据
dataset = './data/data.csv'
count = 0
with open(dataset) as csvfile:
    reader = csv.DictReader( csvfile, fieldnames=[ "name_A" ,"class_A" ,"latitude_A",  "longitude_A", "name_B" , "class_B" , "latitude_B",  "longitude_B", "label"], delimiter='\t' )
    for row in reader:
        name_similarity = jaccard( row['name_A'] , row['name_B'] )
        class_similarity = jaccard(row['class_A'] , row['class_B'])


        # name_similarity2 = damerau_levenshtein( row['name_A'] , row['name_B'] )
        # name_similarity3 = jaro_winkler( row['name_A'] , row['name_B'] )
        # class_similarityA = class_to_index.get(row['class_A'])
        # class_similarityB = class_to_index.get(row['class_B'])
        location_similarity = calculate_distance(float(row['latitude_A']),float(row['longitude_A']),float(row['latitude_B']),float(row['longitude_B']))
        print(location_similarity)
        if class_similarity == 0:
            count=count+1
        print(class_similarity,row['label'])


        data.append([name_similarity,class_similarity,location_similarity])
        # data.append([name_similarity,name_similarity2,name_similarity3,class_similarityA,class_similarityB,location_similarity])
        label.append(row['label'])
print(count)
# print(data)
# print(label)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=100)

# 创建随机森林分类器模型并训练
# rf_classifier = svm.LinearSVC()
num_epochs = 20

for epoch in range(num_epochs):
    # rf_classifier = ensemble.RandomForestClassifier()
    #rf_classifier = DecisionTreeClassifier(random_state=42)
    rf_classifier = MultinomialNB()
    #rf_classifier = LogisticRegression()
    rf_classifier.fit(X_train, y_train)


    predictions = rf_classifier.predict(X_test)
    # print(predictions)
    # print(y_test)
    precision, recall, f_score, true_sum = precision_recall_fscore_support(y_test, predictions)
    accuracy = rf_classifier.score(X_test, y_test)
    # print(precision_recall_fscore_support(y_test, predictions))
    print(f'train Epoch {epoch+1}/{num_epochs}, Precision: {precision[1]:.4f}, F1 Score: {f_score[1]:.4f},Accuracy: {accuracy:.4f},recall: {recall[1]:.4f}')

    # # 输出结果
    # print("Precision:", precision[1],"Recall:", recall)
    # print()
    # print("F1 Score:", f_score)
    # # 输出准确率
    # accuracy = rf_classifier.score(X_test, y_test)
    # print("Accuracy:", accuracy)
    #
    # conf_matrix = confusion_matrix(y_test, predictions)
    # # 提取 TP, FP, TN, FN
    # tn, fp, fn, tp = conf_matrix.ravel()
    # print("True Positives:", tp)
    # print("False Positives:", fp)
    # print("True Negatives:", tn)
    # print("False Negatives:", fn)