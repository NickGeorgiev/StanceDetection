import numpy

DIMENSIONS = 50

WORD_EMBEDDINGS = {}


file = open('../data/glove/glove.6B.{0}d.txt'.format(DIMENSIONS), encoding="utf8")
for line in file:
    values = line.split()
    WORD_EMBEDDINGS[values[0]] = numpy.asarray(values[1:], dtype='float32')


print("EMBEDDINGS READ SUCCESSFULLY!")


def get_tweet_vector(tweet):
    return sum([
      WORD_EMBEDDINGS.get(word, numpy.zeros((DIMENSIONS,)))
      for word
      in tweet.split()]) / (len(tweet.split())+0.001) if len(tweet) != 0 else numpy.zeros((DIMENSIONS,))


def extract_glove_features(features, tweet):
    tweet_vector = get_tweet_vector(tweet)
    for idx, value in enumerate(tweet_vector):
        features['glove: {0}'.format(idx)] = float(value)
