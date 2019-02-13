import json
import operator
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessor import initial_text_clean_up

labeled_data = []
with open('../jsons/labeled_data.json', 'r') as file:
    labeled_data = [ initial_text_clean_up(tweet) for tweet, target, stance in list(json.load(file))]

def calculate_tfidf_unigrams(n_grams_count):
    vectorizer = TfidfVectorizer(stop_words='english')
    new_data = vectorizer.fit_transform(labeled_data)
    # print(vectorizer.vocabulary_)
    # test_data = vectorizer.inverse_transform(["Overheard at the Davenport open house: \"Oh, I know you from Facebook!\" #DigitalOrganizing #SemST"])
    # print(test_data)
    idf = vectorizer.idf_
    # print(idf)
    res = dict(zip(vectorizer.get_feature_names(), idf))
    print(sorted(res.items(), key=operator.itemgetter(1), reverse=False)[:50])
    with open('../jsons/tfidf-{0}.json'.format(n_grams_count),'w') as file:
        file.write(json.dumps(res))



calculate_tfidf_unigrams(1)