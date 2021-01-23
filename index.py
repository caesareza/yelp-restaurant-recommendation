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
                    js onData.append(json.loads(line.rstrip(), strict=False))
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
# print(restaurants_and_food.head(10))

# restaurants_and_food.describe()
# restaurants_and_food.info()

clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
stopworda = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower() # lowercase text
    text = clean_spcl.sub(' ', text)
    text = clean_symbol.sub('', text)
    text = ' '.join(word for word in text.split() if word not in stopworda) # hapus stopword dari kolom deskripsi
    return text

# Buat kolom tambahan untuk data description yang telah dibersihkan
restaurants_and_food['cat'] = restaurants_and_food['categories'].apply(clean_text)


restaurants_and_food.drop_duplicates(subset='name', keep=False, inplace=True)
print(restaurants_and_food.head(30)[['name', 'cat']])

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

restaurants_and_food.set_index('name', inplace=True)
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(restaurants_and_food['cat'])
con_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# print (con_sim)

tampil = pd.Series(restaurants_and_food.index)
# print (tampil[:20])

def rekomendasi(name, con_sim = con_sim):
    recommeded_hotel = []

    idx = tampil[tampil == name].index[0]

    score_series = pd.Series(con_sim[idx]).sort_values(ascending=False)

    top_10_restaurant = list(score_series.iloc[1:11].index)

    for i in top_10_restaurant:
        recommeded_hotel.append(list(restaurants_and_food.index)[i])

    print (recommeded_hotel)
    print (restaurants_and_food.iloc[top_10_restaurant, [2,12,14]])


rekomendasi('Our Daily Bread')

# print (restaurants_and_food.iloc[[12,10], [0,1]])