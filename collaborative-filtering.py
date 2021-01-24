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

