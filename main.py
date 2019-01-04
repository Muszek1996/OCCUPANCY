import pandas as pd
import sklearn.preprocessing
from sklearn_pandas import DataFrameMapper
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


arr = ['Temperature','Humidity','HumidityRatio','Light','CO2','Hour','Weekend']

def prepareData(dataFrame):
    date = dataFrame['date']
    dateTime = pd.to_datetime(date, format='%Y/%m/%d %H:%M:%S')
    hours = dateTime.dt.hour
    weekend = dateTime.dt.weekday.values
    weekend[weekend<5]= 0
    weekend[weekend>=5]= 1
    Y = dataFrame['Occupancy']
    dataFrame.drop({"date", "Occupancy"}, axis="columns", inplace=True)
    newAtributes = pd.DataFrame(data={'Hour': hours, 'Weekend': weekend},
                                columns=['Hour', 'Weekend'])  # 1st column as index
    X_df = dataFrame.join(newAtributes)
    X = X_df
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



classifiers = [
    KNeighborsClassifier(5,algorithm='brute'),
    SVC(kernel="linear", C=0.025),  #c=0.025
    DecisionTreeClassifier(max_depth=4,presort=True,max_features=7),
   MLPClassifier(alpha=1,solver='sgd' ,hidden_layer_sizes=(12,12),learning_rate='constant',learning_rate_init=0.0035,activation='logistic'),
]


names = ["Nearest Neighbors", "Linear SVM","Decision Tree", "Neural Network"]

for classifier, name in zip(classifiers, names):
    classifier.fit(X, Y)

    print('datatest ' + name +' accuracy:\t'+ str(classifier.score(X2, Y2))+"%")
    print('datatest2 '+ name +' accuracy:\t'+ str(classifier.score(X3, Y3))+"%\n")



from mlxtend.feature_selection import SequentialFeatureSelector as SFS

for classifier, name in zip(classifiers, names):
    for i in range(1,7):
        features = i
        sfs = SFS(classifier,
                  k_features=features,
                  forward=True,
                  floating=False,
                  scoring='accuracy',
                  cv=4,
                  n_jobs=-1)
        sfs = sfs.fit(X, Y)



        print('\nSequential Forward Selection (k='+str(features)+', '+name+' ):')
        for i in sfs.k_feature_idx_:
            print(arr[i])
        print('CV Score:')
        print(sfs.k_score_)
