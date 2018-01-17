import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('HR_comma_sep.csv')

labelencoder=LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

x_data = data.drop('left', axis=1) # all rows, all the features and no labels
y_data = data['left']  # all rows, label only




# print y_data
print np.unique(y_data)

# x_data = x_data.drop('class_type', axis=1)
# X=preprocessing.scale(X)


X_train, X_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=4)
print y_test
print np.bincount(y_train)
print np.bincount(y_test)
model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, y_train)

y_pred = model_tree.predict(X_test)

print model_tree.score(X_test, y_test)

# Random Forest Model
clf = RandomForestClassifier(
    n_estimators=1000, 
    max_depth=None, 
    min_samples_split=10, 
    class_weight="balanced"
    #min_weight_fraction_leaf=0.02 
    )
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
y_acc = clf.score(X_test, y_test)
print y_acc