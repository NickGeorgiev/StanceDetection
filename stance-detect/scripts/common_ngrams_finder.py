import nltk

import re
import csv
import json
import string

from collections import Counter
from preprocessor import remove_punctuation, remove_escaped_characters, get_hashtags

words = set(nltk.corpus.words.words())

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

data = []
with open('../jsons/labeled_data.json', 'r') as file:
    data = json.load(file)

data = [tweet for tweet, target, stance in data]

all_data_as_text = ' '.join(data)
data = [item for row in data for item in nltk.wordpunct_tokenize(row.lower()) if item in words]


tokens = nltk.word_tokenize(remove_punctuation(all_data_as_text))
for t in tokens:
    if t in string.punctuation or t.isnumeric():
        tokens.remove(t)
pos_tags = nltk.pos_tag(tokens, tagset="universal")
pos_tags = [tag[1] for tag in pos_tags]

pos_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(pos_tags)
pos_trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(pos_tags)
bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(data)
trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(data)

# # get only the bigrams and trigrams that appear 3+ times
bigram_finder.apply_freq_filter(3)
trigram_finder.apply_freq_filter(3)
pos_bigram_finder.apply_freq_filter(3)
pos_trigram_finder.apply_freq_filter(3)

# write the n-grams to json files so that they could be easily imported
with open('../jsons/bigrams.json', 'w') as file:
    file.write(json.dumps(['{0} {1}'.format(bi[0], bi[1]) for bi in bigram_finder.nbest(bigram_measures.pmi, 10)]))

with open('../jsons/trigrams.json', 'w') as file:
    file.write(json.dumps(['{0} {1} {2}'.format(tri[0], tri[1], tri[2]) for tri in trigram_finder.nbest(trigram_measures.pmi, 10)]))

with open('../jsons/pos_bigrams.json', 'w') as file:
    file.write(json.dumps(['{0} {1}'.format(bi[0], bi[1]) for bi in pos_bigram_finder.nbest(bigram_measures.pmi, 10)]))

with open('../jsons/pos_trigrams.json', 'w') as file:
    file.write(json.dumps(['{0} {1} {2}'.format(tri[0], tri[1], tri[2]) for tri in pos_trigram_finder.nbest(trigram_measures.pmi, 10)]))

stopwords = set(nltk.corpus.stopwords.words('english'))
word_count = Counter([word for word in data if word.lower() not in stopwords])

with open('../jsons/unigrams.json', 'w') as file:
     file.write(json.dumps([word for word, freq in word_count.most_common(10)]))

