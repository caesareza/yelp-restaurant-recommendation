import time
import json
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
nltk.download('stopwords')

# 1050 chinese cuisine / Tangra Villa Hakka Chinese Cuisine
# 1200 pizza / Zeppe's Tavern & Pizzeria
# 4000 meat / Smoked Up Meats
# 5700 pizza / Ital Vera Pizza
# 6000 baked / The Baked Bear
# 7500 mexican / 1900 Mexican Grill
businesses = pd.read_csv('yelp_academic_dataset_business.csv', nrows=7500)
review = pd.read_csv('yelp_academic_dataset_review.csv', nrows=7500)

restoran = businesses[['business_id','name','address', 'categories', 'attributes','stars']]

rest = restoran[restoran['categories'].str.contains('Restaurant.*')==True].reset_index()


df_categories_dummies = pd.Series(rest['categories']).str.get_dummies(',')
# print(df_categories_dummies.head(10))

# pull out names and stars from rest table
result = rest[['name','stars']]


# Concat all tables and drop Restaurant column
df_final = pd.concat([df_categories_dummies, result], axis=1)

# map floating point stars to an integer
mapper = {1.0:1,1.5:2, 2.0:2, 2.5:3, 3.0:3, 3.5:4, 4.0:4, 4.5:5, 5.0:5}
df_final['stars'] = df_final['stars'].map(mapper)

# print(df_final.head(10))

# Create X (all the features) and y (target)
X = df_final.iloc[:,:-2]
y = df_final['stars']

# Split the data into train and test sets
from sklearn.model_selection import train_test_split
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train_knn, y_train_knn)

print('X_test_knn shape',X_train_knn.shape)
print('y_train_knn shape',y_train_knn.shape)

accuracy_train = knn.score(X_train_knn, y_train_knn)
accuracy_test = knn.score(X_test_knn, y_test_knn)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = LogisticRegression()
cvs = cross_val_score(model, X_train_knn, y_train_knn, scoring='accuracy', cv=cv, n_jobs=-1)
y_pred = cross_val_predict(model, X_train_knn, y_train_knn, cv=cv)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
mae = mean_absolute_error(y_train_knn, y_pred)
mse = mean_squared_error(y_train_knn, y_pred)
rmse = mean_squared_error(y_train_knn, y_pred, squared=False)

print('y_pred.shape',y_pred.shape)
print(f"Score on training set: {accuracy_train}")
print(f"Score on test set: {accuracy_test}")

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(cvs), std(cvs)))
print('MAE', mae)
print('MSE', mse)
print('RMSE', rmse)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_knn, y_pred)
print('confusion matrix', cm)

# Recall
from sklearn.metrics import recall_score
recall = recall_score(y_train_knn, y_pred, average='weighted')
print('recall:', recall)

# Precision
from sklearn.metrics import precision_score
precision = precision_score(y_train_knn, y_pred, average='weighted')
print('precision:', precision)

# Method 1: sklearn
from sklearn.metrics import f1_score
f1_score(y_train_knn, y_pred, average=None)
# Method 2: Manual Calculation
F1 = 2 * (precision * recall) / (precision + recall)
print('F-Measure:', F1)


# look at the last row for the test
print(df_final.iloc[-1:])

# look at the restaurant name from the last row.
print("Validation set (Restaurant name): ", df_final['name'].values[-1])


# test set from the df_final table (only last row): Restaurant name: "Steak & Cheese & Quick Pita Restaurant"
test_set = df_final.iloc[-1:,:-2]

# validation set from the df_final table (exclude the last row)
X_val =  df_final.iloc[:-1,:-2]
y_val = df_final['stars'].iloc[:-1]

# fit model with validation set
n_knn = knn.fit(X_val, y_val)

# distances and indeces from validation set (Steak & Cheese & Quick Pita Restaurant)
distances, indeces =  n_knn.kneighbors(test_set)
#n_knn.kneighbors(test_set)[1][0]

# create table distances and indeces from "Steak & Cheese & Quick Pita Restaurant"
final_table = pd.DataFrame(n_knn.kneighbors(test_set)[0][0], columns = ['distance'])
final_table['index'] = n_knn.kneighbors(test_set)[1][0]
final_table.set_index('index')

print(final_table.set_index('index'))

# get names of the restaurant that similar to the "Steak & Cheese & Quick Pita Restaurant"
result = final_table.join(df_final,on='index')
hasil = result.head(5)
print(hasil)