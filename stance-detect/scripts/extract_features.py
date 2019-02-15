import json
from textblob import TextBlob

from preprocessor import *
from utilities import *

from word_embeddings import extract_glove_features
from nmf_clustering import extract_topic_features

COMMON_UNIGRAMS = set()
COMMON_BIGRAMS = set()
COMMON_TRIGRAMS = set()
COMMON_POS_BIGRAMS = set()
COMMON_POS_TRIGRAMS = set()
COMMON_HASHTAGS = set()
TFIDF_DATA = set()

with open('../jsons/unigrams.json', 'r') as file:
    COMMON_UNIGRAMS = set(json.load(file))
with open('../jsons/bigrams.json', 'r') as file:
    COMMON_BIGRAMS = set(json.load(file))
with open('../jsons/trigrams.json', 'r') as file:
    COMMON_TRIGRAMS = set(json.load(file))
with open('../jsons/pos_bigrams.json', 'r') as file:
    COMMON_POS_BIGRAMS = set(json.load(file))
with open('../jsons/pos_trigrams.json', 'r') as file:
    COMMON_POS_TRIGRAMS = set(json.load(file))
with open('../jsons/common_hashtags.json','r') as file:
    COMMON_HASHTAGS = dict(json.load(file))
with open('../jsons/tfidf.json', 'r') as file:
    TFIDF_DATA = dict(json.load(file))


def extrtact_tfidf_vector_for_tweet(features, tweet):
    tweet = get_tweet_text_only(tweet)
    tweet_tokens = nltk.word_tokenize(tweet)
    for key in TFIDF_DATA:
        features['token-{0}'.format(key)] = TFIDF_DATA[key] if key in tweet_tokens else 0
    # features['avg-tfidf'] = 0
    # for token in tweet_tokens:
    #     features['token-{0}'.format(token)] = TFIDF_DATA[token] if token in TFIDF_DATA else 0
    #     features['avg-tfidf'] += TFIDF_DATA[token] if token in TFIDF_DATA else 0
    # features['avg-tfidf'] /= len(tweet_tokens)

 
def extract_sentiment_features_of_tweet(features, tweet):
    tweet = get_tweet_text_only(tweet)
    tokens = nltk.word_tokenize(tweet)

    # Extract features of full sentence.
    tweet_blob = TextBlob(' '.join(tokens))
    features['tweet_polarity'] = tweet_blob.sentiment.polarity
    features['tweet_subjectivity'] = tweet_blob.sentiment.subjectivity


def extract_capitalization_features(features, text):
    capitalized_phrases = [t for t in get_capitalized_text(text) if len(t) > 1]
    features['capitalization_presence'] = int(len(capitalized_phrases) > 0)
    polarity = 0.0
    for phrase in capitalized_phrases:
        polarity += TextBlob(phrase).polarity
    if len(capitalized_phrases):
        polarity /= len(capitalized_phrases)
    features['capitalization_polarity'] = polarity


def extract_interjections_features(features, text):
    interjection_words_descriptions = get_interjection_words_descriptions(text)
    polarity = 0.0
    subjectivity = 0.0
    for interj in interjection_words_descriptions:
        blob = TextBlob(interj)
        polarity += blob.polarity
        subjectivity += blob.subjectivity
    if len(interjection_words_descriptions):
        polarity /= len(interjection_words_descriptions)
        subjectivity /= len(interjection_words_descriptions)
    features['interjections_polarity'] = polarity
    features['interjections_subjectivity'] = subjectivity


def extract_hashtag_features(features, text):
    hashtags = get_hashtags(text)
    polarity = 0.0
    subjectivity = 0.0

    features['hillary_hashtags_count'] = 0
    features['climate_hashtags_count'] = 0
    features['abortion_hashtags_count'] = 0
    features['feminism_hashtags_count'] = 0
    features['atheism_hashtags_count'] = 0

    for tag in hashtags:
        blob = TextBlob(tag)
        polarity += blob.polarity
        subjectivity += blob.subjectivity

        features['hillary_hashtags_count'] += 1 if tag in COMMON_HASHTAGS['Hillary'] else 0
        features['climate_hashtags_count'] += 1 if tag in COMMON_HASHTAGS['Climate'] else 0
        features['abortion_hashtags_count'] += 1 if tag in COMMON_HASHTAGS['Abortion'] else 0
        features['feminism_hashtags_count'] += 1 if tag in COMMON_HASHTAGS['Feminism'] else 0
        features['atheism_hashtags_count'] += 1 if tag in COMMON_HASHTAGS['Atheism'] else 0

    if len(hashtags):
        polarity /= len(hashtags)
        subjectivity /= len(hashtags)
    features['hashtags_polarity'] = polarity
    features['hashtags_subjectivity'] = subjectivity


def extract_punctuation_features(features, text):
    features["punctuation_feature"] = len(get_punctuation(text))


def extract_quoted_text_features(features, text):
    features["quoted_text"] = len(get_quoted_text(text))


def extract_quoted_text_polarity(features, text):
    quotes = get_quoted_text(text)
    polarity = 0.0
    for _, quote in quotes:
        quote_blob = TextBlob(quote)
        polarity += quote_blob.sentiment.polarity
    features["quoted_text_polarity"] = -polarity


def extract_ngrams_features(features, text):
    text = get_tweet_text_only(text)

    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t.lower()) for t in tokens]

    bigrams = nltk.bigrams(tokens)
    bigrams = ['{0} {1}'.format(bi[0], bi[1]) for bi in bigrams]
    trigrams = nltk.trigrams(tokens)
    trigrams = ['{0} {1} {2}'.format(tri[0], tri[1], tri[2]) for tri in trigrams]

    features['common_words_count'] = 0

    for token in tokens:
        if token in COMMON_UNIGRAMS:
            features['common_words_count'] += 1

    for bigram in COMMON_BIGRAMS:
        features['bigram: ' + bigram] = 1 if bigram in bigrams else 0

    for trigram in COMMON_TRIGRAMS:
        features['trigram: ' + trigram] = 1 if trigram in trigrams else 0


def extract_pos_ngrams_features(features, text):
    pos_bigrams, pos_trigrams = get_pos_ngrams(text)

    for pos_bigram in COMMON_POS_BIGRAMS:
        features['pos_bigram: ' + pos_bigram] = 1 if pos_bigram in pos_bigrams else 0

    for pos_trigram in COMMON_POS_TRIGRAMS:
        features['pos_trigram: ' + pos_trigram] = 1 if pos_trigram in pos_trigrams else 0


# looks up whether there are pronouns that should be considered as othering language in the whole sentence
def extract_othering_language_features(features, text):
    text = get_pos_sentence(text)
    adj_prp_text = [[word,tag] for word, tag in text if tag in ['PRP', 'PRP$', 'JJ']]

    # filter founded pronouns and adjectives to exclude ones that should be considered as outgroup language
    filtered_text = [word for word, tag in adj_prp_text if word in outgroup_pronouns or word == outgroup_adjective]
    features['outgroup_language_coef'] =  len(filtered_text) / (len(adj_prp_text) + 0.00001)


def extract_othering_language_collocations(features, text):
    text = get_pos_sentence(text)
    all_text = ''.join([tag for word,tag in text])
    vrb_prn_tuples = re.findall(verb_pronoun_regex, all_text)
    features['othering_tuples_polarity'] = 0
    polarity = 0
    for verb, pronoun in vrb_prn_tuples:
        if pronoun == outgroup_adjective or pronoun in outgroup_pronouns:
            polarity += TextBlob(verb).sentiment.polarity
    if len(vrb_prn_tuples):
        polarity /= len(vrb_prn_tuples)
    features['othering_tuples_polarity'] = polarity


def extract_features_of_tweet(tweet, raw=False):
    features = {}
    if raw is False:
        tweet = initial_text_clean_up(tweet)
    tweet = remove_unicode_characters(tweet)
    tweet = remove_escaped_characters(tweet)
    extract_glove_features(features, tweet)
    extrtact_tfidf_vector_for_tweet(features, tweet)
    extract_punctuation_features(features, tweet)
    extract_quoted_text_features(features, tweet)
    extract_capitalization_features(features, tweet)
    extract_quoted_text_polarity(features, tweet)
    extract_hashtag_features(features, tweet)
    extract_ngrams_features(features, tweet)
    extract_pos_ngrams_features(features, tweet)
    extract_sentiment_features_of_tweet(features, tweet)
    extract_topic_features(features, tweet)

    return features
