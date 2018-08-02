# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import ensemble 
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier
import os
from sklearn import ensemble
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

print('Preprocessing data...................')
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

test_id = [ ]
for doc in test_df.Id:
    test_id.append(doc)
    
y_train = train_df.Cover_Type
x_train = train_df.drop('Cover_Type', axis=1)
x_train = x_train.drop('Id',axis=1)
test_df = test_df.drop('Id',axis=1)



x_train = x_train.drop('Soil_Type7',axis=1)
x_train = x_train.drop('Soil_Type8',axis=1)
x_train = x_train.drop('Soil_Type15',axis=1)
x_train = x_train.drop('Soil_Type25',axis=1)
# x_train = x_train.drop('Hillshade_3pm',axis=1)
# x_train = x_train.drop('Hillshade_Noon',axis=1)
# x_train = x_train.drop('Hillshade_9am',axis=1)
# x_train = x_train.drop('Slope',axis=1)
# x_train = x_train.drop('Aspect',axis=1)

test_df = test_df.drop('Soil_Type7',axis=1)
test_df = test_df.drop('Soil_Type8',axis=1)
test_df = test_df.drop('Soil_Type15',axis=1)
test_df = test_df.drop('Soil_Type25',axis=1)
# test_df = test_df.drop('Hillshade_9am',axis=1)
# test_df = test_df.drop('Hillshade_Noon',axis=1)
# test_df = test_df.drop('Hillshade_3pm',axis=1)
# test_df = test_df.drop('Slope',axis=1)
# test_df = test_df.drop('Aspect',axis=1)


x_train['HF1'] = x_train['Horizontal_Distance_To_Hydrology']+x_train['Horizontal_Distance_To_Fire_Points']
x_train['HF2'] = abs(x_train['Horizontal_Distance_To_Hydrology']-x_train['Horizontal_Distance_To_Fire_Points'])
x_train['HR1'] = abs(x_train['Horizontal_Distance_To_Hydrology']+x_train['Horizontal_Distance_To_Roadways'])
x_train['HR2'] = abs(x_train['Horizontal_Distance_To_Hydrology']-x_train['Horizontal_Distance_To_Roadways'])
x_train['FR1'] = abs(x_train['Horizontal_Distance_To_Fire_Points']+x_train['Horizontal_Distance_To_Roadways'])
x_train['FR2'] = abs(x_train['Horizontal_Distance_To_Fire_Points']-x_train['Horizontal_Distance_To_Roadways'])
x_train['distance'] = np.sqrt(np.array(x_train['Horizontal_Distance_To_Hydrology']**2 + x_train["Vertical_Distance_To_Hydrology"]**2))
x_train['Mean_Amenities']=(x_train.Horizontal_Distance_To_Fire_Points + x_train.Horizontal_Distance_To_Hydrology + x_train.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
x_train['Mean_Fire_Hyd']=(x_train.Horizontal_Distance_To_Fire_Points + x_train.Horizontal_Distance_To_Hydrology) / 2 
x_train['mean_hillshade'] = (x_train['Hillshade_9am']  + x_train['Hillshade_Noon']  + x_train['Hillshade_3pm'] ) / 3
x_train["Vertical_Distance_To_Hydrology"] = abs(x_train['Vertical_Distance_To_Hydrology'])
x_train['Mean_HorizontalHydrology_HorizontalFire'] = (x_train['Horizontal_Distance_To_Hydrology']+x_train['Horizontal_Distance_To_Fire_Points'])/2
x_train['Mean_HorizontalHydrology_HorizontalRoadways'] = (x_train['Horizontal_Distance_To_Hydrology']+x_train['Horizontal_Distance_To_Roadways'])/2
x_train['Mean_HorizontalFire_Points_HorizontalRoadways'] = (x_train['Horizontal_Distance_To_Fire_Points']+x_train['Horizontal_Distance_To_Roadways'])/2

x_train['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (x_train['Horizontal_Distance_To_Hydrology']-x_train['Horizontal_Distance_To_Fire_Points'])/2
x_train['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (x_train['Horizontal_Distance_To_Hydrology']-x_train['Horizontal_Distance_To_Roadways'])/2
x_train['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (x_train['Horizontal_Distance_To_Fire_Points']-x_train['Horizontal_Distance_To_Roadways'])/2


# x_train['angle'] = x_train['Slope'] * x_train['Elevation']

test_df['HF1'] = test_df['Horizontal_Distance_To_Hydrology']+test_df['Horizontal_Distance_To_Fire_Points']
test_df['HF2'] = abs(test_df['Horizontal_Distance_To_Hydrology']-test_df['Horizontal_Distance_To_Fire_Points'])
test_df['HR1'] = abs(test_df['Horizontal_Distance_To_Hydrology']+test_df['Horizontal_Distance_To_Roadways'])
test_df['HR2'] = abs(test_df['Horizontal_Distance_To_Hydrology']-test_df['Horizontal_Distance_To_Roadways'])
test_df['FR1'] = abs(test_df['Horizontal_Distance_To_Fire_Points']+test_df['Horizontal_Distance_To_Roadways'])
test_df['FR2'] = abs(test_df['Horizontal_Distance_To_Fire_Points']-test_df['Horizontal_Distance_To_Roadways'])
test_df['distance'] = np.sqrt(np.array(test_df['Horizontal_Distance_To_Hydrology']**2 + test_df["Vertical_Distance_To_Hydrology"]**2))
test_df['Mean_Amenities']=(test_df.Horizontal_Distance_To_Fire_Points + test_df.Horizontal_Distance_To_Hydrology + test_df.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
test_df['Mean_Fire_Hyd']=(test_df.Horizontal_Distance_To_Fire_Points + test_df.Horizontal_Distance_To_Hydrology) / 2
test_df['mean_hillshade'] = (test_df['Hillshade_9am']  + test_df['Hillshade_Noon']  + test_df['Hillshade_3pm'] ) / 3
test_df["Vertical_Distance_To_Hydrology"] = abs(test_df['Vertical_Distance_To_Hydrology'])
test_df['Mean_HorizontalHydrology_HorizontalFire'] = (test_df['Horizontal_Distance_To_Hydrology']+test_df['Horizontal_Distance_To_Fire_Points'])/2
test_df['Mean_HorizontalHydrology_HorizontalRoadways'] = (test_df['Horizontal_Distance_To_Hydrology']+test_df['Horizontal_Distance_To_Roadways'])/2
test_df['Mean_HorizontalFire_Points_HorizontalRoadways'] = (test_df['Horizontal_Distance_To_Fire_Points']+test_df['Horizontal_Distance_To_Roadways'])/2

test_df['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (test_df['Horizontal_Distance_To_Hydrology']-test_df['Horizontal_Distance_To_Fire_Points'])/2
test_df['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (test_df['Horizontal_Distance_To_Hydrology']-test_df['Horizontal_Distance_To_Roadways'])/2
test_df['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (test_df['Horizontal_Distance_To_Fire_Points']-test_df['Horizontal_Distance_To_Roadways'])/2
# test_df['angle'] = test_df['Slope'] * test_df['Elevation']


scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(test_df)

preds = pd.DataFrame()

print('Training on data...................')
m1 = ensemble.AdaBoostClassifier(ensemble.ExtraTreesClassifier(n_estimators=500), n_estimators=250, learning_rate=0.01, algorithm='SAMME')  
m1.fit(X_train, y_train) 
preds["Model1"] = m1.predict(X_test)

m2 = ensemble.ExtraTreesClassifier(n_estimators=950)  
m2.fit(X_train, y_train)
preds["Model2"] = m2.predict(X_test)

m3 = XGBClassifier(max_depth=20, n_estimators=500)  
m3.fit(X_train, y_train)
preds["Model3"] = m3.predict(X_test)

m4 = LGBMClassifier(n_estimators=500, max_depth=15)
m4.fit(X_train, y_train)
preds["Model4"] = m4.predict(X_test)

m5 = ensemble.AdaBoostClassifier(ensemble.GradientBoostingClassifier(n_estimators=1000, max_depth=10), n_estimators=1000, learning_rate=0.01, algorithm="SAMME")
m5.fit(X_train, y_train)
preds["Model5"] = m5.predict(X_test)

m6 = SGDClassifier(loss='hinge')
m6.fit(X_train, y_train)
preds["Model6"] = m6.predict(X_test)

preds['Model7'] = m2.predict(X_test)

pred = preds.mode(axis=1)

print ("Generate Submission File ... ")
sub = pd.DataFrame({"Id": test_id, "Cover_Type": pred[0].astype('int').values})
sub.to_csv("emsemble_sub.csv", index=False)
