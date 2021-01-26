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
businesses = pd.read_csv('yelp_academic_dataset_business.csv', nrows=10000)
reviews = pd.read_csv('yelp_academic_dataset_review.csv', nrows=10000)

restoran = businesses[['business_id','name','address', 'categories', 'attributes','stars']]
review = reviews[['user_id','business_id','stars', 'date']]

rest = restoran[restoran['categories'].str.contains('Restaurant.*')==True].reset_index()
# pull out names and addresses of the restaurants from rest table
restaurant = rest[['business_id', 'name', 'categories', 'address']]
restaurant

# combine df_review and restaurant table
combined_business_data = pd.merge(review, restaurant, on='business_id')

# number uniqe user and restaurant
print ('num of users:',combined_business_data['user_id'].nunique())
print ('num of business:',combined_business_data['business_id'].nunique())

# map floating point stars to an integer
mapper = {1.0:1,1.5:2, 2.0:2, 2.5:3, 3.0:3, 3.5:4, 4.0:4, 4.5:5, 5.0:5}
combined_business_data['stars'] = combined_business_data['stars'].map(mapper)

# print(combined_business_data.head(10))
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

# Create X (all the features) and y (target)
X = combined_business_data.iloc[:,:-2]
y = combined_business_data['stars']

# Split the data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(X_train.shape) # --> pakai yg ini
# print(X_test.shape)
print(y_train.shape) # --> dan pakai yg ini
# print(y_test.shape)

# define the pipeline
steps = [('svd', TruncatedSVD(n_components=5)), ('m', LogisticRegression())]
model = Pipeline(steps=steps)
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

