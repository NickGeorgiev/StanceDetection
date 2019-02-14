import json
import operator
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessor import initial_text_clean_up

labeled_data = []
with open('../jsons/labeled_data.json', 'r') as file:
    labeled_data = [ initial_text_clean_up(tweet) for tweet, target, stance in list(json.load(file))]

vectorizer = TfidfVectorizer(stop_words='english')
new_data = vectorizer.fit_transform(labeled_data)
idf = vectorizer.idf_
res = dict(zip(vectorizer.get_feature_names(), idf))
res = sorted(res.items(), key=operator.itemgetter(1), reverse=False)

with open('../jsons/tfidf.json','w') as file:
    file.write(json.dumps(list(res)))
