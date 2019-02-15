import json
import nltk
import numpy as np
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from nltk.metrics import precision, recall, f_measure
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

TEST_SET_SIZE = 0.2

featuresets = []
with open('../jsons/features_sets.json', 'r') as file:
    featuresets = json.load(file)

data_size = len(featuresets)
train_set_end = round(data_size * (1 - TEST_SET_SIZE))
train_set, test_set = featuresets[:train_set_end], featuresets[train_set_end:]

refsets = defaultdict(set)
testsets = defaultdict(set)

X = np.array([np.fromiter(feature_dict.values(), dtype=float) for [feature_dict, label] in train_set])
y = np.array([len(label) for [feature_dict, label] in train_set])

X_test = np.array([np.fromiter(feature_dict.values(), dtype=float) for [feature_dict, label] in test_set])
y_test = np.array([len(label) for [feature_dict, label] in test_set])

scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)
normalized_X = normalize(scaled_X, norm='l1', axis=1, copy=True)
scaled_X_test = scaler.fit_transform(X_test)
normalized_X_test = normalize(scaled_X_test, norm='l1', axis=1, copy=True)
multinomial_naive_bayes = MultinomialNB()
multinomial_naive_bayes.fit(normalized_X, y)
print("Accuracy - Multinomial Naive Bayes Classifier: ")
print(multinomial_naive_bayes.score(normalized_X_test, y_test))

refsets = defaultdict(set)
testsets = defaultdict(set)
for (i, (x, label)) in enumerate(zip(normalized_X_test, y_test)):
    refsets[label].add(i)
    multinomial_result = multinomial_naive_bayes.predict([x])[0]
    testsets[multinomial_result].add(i)


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
print("Accuracy - 3 Nearest Neighbours Classifier: ")
print(neigh.score(X_test, y_test))

refsets = defaultdict(set)
testsets = defaultdict(set)
for (i, (x, label)) in enumerate(zip(X_test, y_test)):
    refsets[label].add(i)
    knn_result = neigh.predict([x])[0]
    testsets[knn_result].add(i)

print("3 Nearest Neighbours Classifier F-measure:")
print('against precision:', precision(refsets[7], testsets[7]))
print('against recall:', recall(refsets[7], testsets[7]))
print('against F-measure:', f_measure(refsets[7], testsets[7]))
print('favor precision:', precision(refsets[5], testsets[5]))
print('favor recall:', recall(refsets[5], testsets[5]))
print('favor F-measure:', f_measure(refsets[5], testsets[5]))
print("F1 score average: ",
      (f_measure(refsets[7], testsets[7]) + f_measure(refsets[5], testsets[5])) / 2)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X, y)
print("Accuracy - 5 Nearest Neighbours Classifier: ")
print(neigh.score(X_test, y_test))

refsets = defaultdict(set)
testsets = defaultdict(set)
for (i, (x, label)) in enumerate(zip(X_test, y_test)):
    refsets[label].add(i)
    knn_result = neigh.predict([x])[0]
    testsets[knn_result].add(i)

print("5 Nearest Neighbours Classifier F-measure:")
print('against precision:', precision(refsets[7], testsets[7]))
print('against recall:', recall(refsets[7], testsets[7]))
print('against F-measure:', f_measure(refsets[7], testsets[7]))
print('favor precision:', precision(refsets[5], testsets[5]))
print('favor recall:', recall(refsets[5], testsets[5]))
print('favor F-measure:', f_measure(refsets[5], testsets[5]))
print("F1 score average: ",
      (f_measure(refsets[7], testsets[7]) + f_measure(refsets[5], testsets[5])) / 2)


linear_svm_classifier = nltk.SklearnClassifier(LinearSVC(C=3.0, dual=True, fit_intercept=True,
                                                         intercept_scaling=0.1, loss='squared_hinge',
                                                         max_iter=2500, penalty='l2', random_state=0,
                                                         tol=0.0001), sparse=True)
linear_svm_classifier.train(train_set)
print("Accuracy - Linear SVM Classifier: ")
print(nltk.classify.accuracy(linear_svm_classifier, test_set))


for i, (features, label) in enumerate(test_set):
    refsets[label].add(i)
    lsvm_result = linear_svm_classifier.classify(features)
    testsets[lsvm_result].add(i)

print("Linear SVM Classifier F-measure:")
print('against precision:', precision(refsets['AGAINST'], testsets['AGAINST']))
print('against recall:', recall(refsets['AGAINST'], testsets['AGAINST']))
print('against F-measure:', f_measure(refsets['AGAINST'], testsets['AGAINST']))
print('favor precision:', precision(refsets['FAVOR'], testsets['FAVOR']))
print('favor recall:', recall(refsets['FAVOR'], testsets['FAVOR']))
print('favor F-measure:', f_measure(refsets['FAVOR'], testsets['FAVOR']))
print("F1 score average: ",
      (f_measure(refsets['AGAINST'], testsets['AGAINST']) + f_measure(refsets['FAVOR'], testsets['FAVOR'])) / 2)


random_forest = SklearnClassifier(RandomForestClassifier(n_estimators=100,
                                                         criterion='gini',
                                                         max_depth=5,
                                                         min_samples_split=2,
                                                         min_samples_leaf=1,
                                                         min_weight_fraction_leaf=0.0,
                                                         max_features=25,
                                                         max_leaf_nodes=20,
                                                         min_impurity_decrease=0.0,
                                                         bootstrap=True,
                                                         oob_score=False,
                                                         random_state=None),
                                  sparse=False)
random_forest.train(train_set)
print("Accuracy - Random Forest Classifier: ")
print(nltk.classify.accuracy(random_forest, test_set))

refsets = defaultdict(set)
testsets = defaultdict(set)
for i, (features, label) in enumerate(test_set):
    refsets[label].add(i)
    random_forest_result = random_forest.classify(features)
    testsets[random_forest_result].add(i)

print("Random Forest Classifier F-measure:")
print('against precision:', precision(refsets['AGAINST'], testsets['AGAINST']))
print('against recall:', recall(refsets['AGAINST'], testsets['AGAINST']))
print('against F-measure:', f_measure(refsets['AGAINST'], testsets['AGAINST']))
print('favor precision:', precision(refsets['FAVOR'], testsets['FAVOR']))
print('favor recall:', recall(refsets['FAVOR'], testsets['FAVOR']))
print('favor F-measure:', f_measure(refsets['FAVOR'], testsets['FAVOR']))
print("F1 score average: ",
      (f_measure(refsets['AGAINST'], testsets['AGAINST']) + f_measure(refsets['FAVOR'], testsets['FAVOR'])) / 2)
