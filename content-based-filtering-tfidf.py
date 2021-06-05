import time
import json
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
nltk.download('stopwords')

def json_to_csv(directory, fileNames, createSample=False):
    print("json to csv")
    start = time.time()
    jsonData = []

    for fileName in fileNames:
        with open(directory + fileName, encoding="utf8") as file:
            print('{0} opened'.format(fileName))
            for line in file:
                # I use an rstrip here because some of the files have trailing blank spaces
                try:
                    jsonData.append(json.loads(line.rstrip(), strict=False))
                except ValueError:
                    print(line)

        df = pd.DataFrame.from_dict(jsonData)

        csvFileName = fileName[:len(fileName) - 5] + '.csv'

        df.to_csv(directory + csvFileName)
        print('{0} created'.format(csvFileName))

        if createSample:
            np.random.seed(9001)
            msk = np.random.rand(len(df)) <= 0.1
            sample = df[msk]

            csvSampleFileName = fileName[:len(fileName) - 5] + '_sample.csv'

            sample.to_csv(directory + csvSampleFileName)
            print('{0} created'.format(csvSampleFileName))

    print('This function took {} minutes to run'.format((time.time() - start) / 60))

# json_to_csv("/Users/dreas/www/yelp/", ["yelp_academic_dataset_review.json"])

business = pd.read_csv('yelp_academic_dataset_business.csv')
mask_restaurants = business['categories'].str.contains('Restaurants')
# create a mask for food
mask_food = business['categories'].str.contains('Food')
# apply both masks
restaurants_and_food = business[mask_restaurants & mask_food]
# number of businesses that have food and restaurant in their category
restaurants_and_food.drop_duplicates(subset='name', keep=False, inplace=True)
# print(restaurants_and_food[['name', 'categories']].head(20))


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# restaurants_and_food['categories'] = restaurants_and_food['categories'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(restaurants_and_food['categories'])
0
tfidf_matrix.shape
# print (tfidf_matrix.shape)

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel


# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(restaurants_and_food.index, index=restaurants_and_food['name']).drop_duplicates()

# print (indices)

def rekomendasi(name, cosine_sim = cosine_sim):
    idx = indices[name]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    restaurant_indices = [i[0] for i in sim_scores]

    # return restaurants_and_food['name'].iloc[restaurant_indices]
    print(restaurants_and_food[['name', 'categories']].iloc[restaurant_indices])

rekomendasi('Allegria')
# print(restaurants_and_food['name'].iloc[147])
# print(restaurants_and_food['name'].iloc[1000])