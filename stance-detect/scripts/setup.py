import nltk

nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('stopwords')
nltk.download('wordnet')

import spacy
nlp = spacy.load('en')