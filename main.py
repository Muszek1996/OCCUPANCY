import pandas as pd
import sklearn.preprocessing
from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor





def prepareData(data):
    time = data['date']
    date = pd.to_datetime(time, format='%Y/%m/%d %H:%M:%S', utc=1)  # datetime
    hours = date.dt.hour
    weekend = date.dt.weekday.values
    weekend[weekend<5]= 0
    weekend[weekend>=5]= 1
    Y = data['Occupancy'].values
    data.drop({"date", "Occupancy"}, axis="columns", inplace=True)
    newAtributes = pd.DataFrame(data={'Hour': hours, 'Weekend': weekend},
                                columns=['Hour', 'Weekend'])  # 1st column as index
    X_df = data.join(newAtributes)
    X = standarizeData(X_df)
    return X,Y

def standarizeData(data):
    mapper = DataFrameMapper([
        (['Temperature','Humidity','HumidityRatio','Light','CO2','Hour'], sklearn.preprocessing.StandardScaler()),
        (['Weekend'], sklearn.preprocessing.Binarizer())
    ])

    X = mapper.fit_transform(data)
    return X



datatraining = pd.read_csv('./occupancy_data/datatraining.txt', header = 0,delimiter = ",")
datatest = pd.read_csv('./occupancy_data/datatest.txt', header = 0,delimiter = ",")
datatest2 = pd.read_csv('./occupancy_data/datatest2.txt', header = 0,delimiter = ",")

X,Y = prepareData(datatraining)
X2,Y2 = prepareData(datatest)
X3,Y3 = prepareData(datatest2)


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

classifiers = [
    KNeighborsClassifier(5,algorithm='brute'),
    SVC(kernel="linear", C=0.025),  #c=0.025
    DecisionTreeClassifier(max_depth=4,presort=True,max_features=7),
    MLPClassifier(alpha=1,solver='sgd' ,hidden_layer_sizes=(12,12),learning_rate='constant',learning_rate_init=0.0035,activation='logistic'),
]

names = ["Nearest Neighbors", "Linear SVM","Decision Tree", "Neural Net"]

from sklearn import metrics

for classifier, name in zip(classifiers, names):
    classifier.fit(X, Y)
    predicts = classifier.predict(X2)
    print(metrics.accuracy_score(Y2,predicts))
  #  print('datatraining ' + name +':\t'+ str(classifier.score(X, Y)))
    print('\ndatatest ' + name +':\t'+ str(classifier.score(X2, Y2)))
    print('datatest2 '+ name +':\t'+ str(classifier.score(X3, Y3)))

