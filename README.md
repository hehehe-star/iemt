## Introduction
Multi-source toponymic matching is the process of consistently discriminating between geometric and attribute information of toponyms, so as to obtain more accurate, complete and high-quality data. Attribute information of toponyms is expressed by natural language symbols, while geometric information of toponyms is expressed by coordinate values, and there is a modal gap between the two in terms of expression system and expression mode, which leads to differences in information perspective and semantic interpretation of textual features and spatial features of toponyms. At the present stage, geographic entity matching mainly adopts similarity index calculation method to compare toponymic features. However, similarity indicators under different evaluation systems have their own judgment thresholds. The similarity thresholds often rely on experimental evaluation and expert experience, which are subjective and difficult to comprehensively access and analyze the semantic and spatial features of toponyms. A common requirement is to integrate multiple similarity metrics into a unified and objective evaluation system to realize the representation and encoding of semantic features and spatial data of toponyms in the hidden embedding space. Therefore, in our study, we propose a coding method for deep feature extraction of toponyms, which embeds textual features and spatial features of toponyms into a unified high-dimensional vector representation space, and utilizes the attention mechanism to capture the intersection features between attributes to obtain the vector representation of toponyms. We experimentally validate the fusion coding method for place name data on GeoNames and OpenStreetMap datasets. The results show that our method achieves good results in terms of accuracy, precision, recall and F1 score, and realizes a unified measure of semantic similarity and spatial similarity and a joint representation of toponym.
![image](https://github.com/user-attachments/assets/69b42a8c-cd01-464f-a43f-7cef1e830402)

## Data resource
The data used in this project comes from GeoNames [https://download.geonames.org/export/dump/] and OSM [https://download.geofabrik.de/] databases, and the preprocessed annotated data for both databases is located at: `data\data.csv`

## Quick start
1. Download the code and dataset to your local, or use the git clone command: `git clone https://github.com/hehehe-star/iemt.git`
2. Install the dependencies required for this code: `pip install -r requirements.txt`
3. Train the model and execute the code: `python models/model.py`

## Result
Table 1 shows the comparison results of hyperparameter tuning. The code is reproduced as follows: `python models/ablation_model.py`

<img width="568" alt="Snipaste_2024-09-08_20-29-34" src="https://github.com/user-attachments/assets/71ee1099-2086-4f16-8fbb-aba173d5abc4">

Table 2 shows the experimental results of the comparative model, and the code file can be found at:`models/compare`
<img width="547" alt="Snipaste_2024-09-08_20-29-48" src="https://github.com/user-attachments/assets/e3873795-e143-4f1a-a181-0fe299f7735f">
