print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
from sklearn import linear_model
regr = linear_model.LinearRegression()


import pandas as pd
import sklearn.preprocessing
from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA
# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

def prepareData(data):
    time = data['date']
    date = pd.to_datetime(time, format='%Y/%m/%d %H:%M:%S')  # datetime
    hours = date.dt.hour
    weekday = date.dt.weekday
    data.drop({"date", "Occupancy"}, axis="columns", inplace=True)
    newAtributes = pd.DataFrame(data={'hour': hours, 'weekday': weekday},
                                columns=['hour', 'weekday'])  # 1st column as index
    X = data.join(newAtributes)

    mapper = DataFrameMapper([
        (['Temperature'], sklearn.preprocessing.StandardScaler()),  # single transformation
        (['Humidity'], sklearn.preprocessing.StandardScaler()),  # single transformation
        (['Light'], sklearn.preprocessing.StandardScaler()),  # single transformation
        (['CO2'], sklearn.preprocessing.StandardScaler()),  # single transformation
        (['HumidityRatio'], sklearn.preprocessing.StandardScaler()),  # single transformation
        (['hour'],sklearn.preprocessing.Normalizer()),  # single transformation
        (['weekday'], sklearn.preprocessing.Normalizer())  # single transformation
    ])

    X = mapper.fit_transform(X)
    return X

datatraining = pd.read_csv('./occupancy_data/datatraining.txt', header = 0,delimiter = ",")
datatest = pd.read_csv('./occupancy_data/datatest.txt', header = 0,delimiter = ",")
datatest2 = pd.read_csv('./occupancy_data/datatest2.txt', header = 0,delimiter = ",")





Y = datatraining['Occupancy']
X = prepareData(datatraining)
Y2 = datatest['Occupancy']
X2 = prepareData(datatest)
Y3 = datatest2['Occupancy']
X3 = prepareData(datatest2)







knn.fit(X,Y)
regr.fit(X,Y)




print("regr")
print(regr.score(X2,Y2))
print(regr.score(X3,Y3))
print("knn")
print(knn.score(X2,Y2))
print(knn.score(X3,Y3))