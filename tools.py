from sklearn.ensemble import *
from sklearn.naive_bayes import *
from sklearn import svm
from sklearn.neighbors import *
from sklearn import tree
from sklearn import linear_model

from sklearn import model_selection
from sklearn import metrics

from bs4 import BeautifulSoup

from nltk.corpus import stopwords
#from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

import itertools
import numpy as np
import matplotlib.pyplot as plt

import re
import csv
#import logging

class tools:
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    def __init__(self):
        return None

    def save_data(self, labels, file_name):
        with open(file_name, 'w') as my_file:
            my_writer = csv.writer(my_file)
            for label in labels:
                tmp = []
                tmp.append(label)
                my_writer.writerow(tmp)

    def to_words(self, raw_text, remove_stopwords = False, lemmatize = True, stem = False):
        text = BeautifulSoup(raw_text, 'html.parser').get_text()
        text = re.sub('[^a-zA-Z]', ' ', text)
        # tokenizer = RegexpTokenizer(r'\w+')
        # words = tokenizer.tokenize(words.strip().lower())
        words = text.lower().split()

        if remove_stopwords:
            stops = set(stopwords.words('english'))
            words = [w for w in words if not w in stops]

        if lemmatize:
            words = [self.lemmatizer.lemmatize(w) for w in words]

        if stem:
            words = [self.stemmer.stem(w) for w in words]

        return(' '.join(words))

    def to_wordlist(self, raw_text, remove_stopwords=False):
        text = BeautifulSoup(raw_text, 'html.parser').get_text()
        text = re.sub('[^a-zA-Z]', ' ', text)
        words = text.lower().split()

        if remove_stopwords:
            stops = set(stopwords.words('english'))
            words = [w for w in words if not w in stops]
        return (words)

    def to_sentences(self, raw_text, tokenizer, remove_stopwords=False):
        raw_sentences = tokenizer.tokenize(raw_text.strip())
        sentences = []

        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(self.to_wordlist(raw_sentence, remove_stopwords))
        return sentences

    def ml_classify(self, method, my_train_data, my_train_label, my_test_data, file_name, debug = False):
        if method == 'knn':
            clf = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
            if debug:
                scores = model_selection.cross_val_score(clf, my_train_data, my_train_label, cv=5)
                print('knn(C=%.1f) accuracy: %0.3f (+/- %0.3f)' % (3, scores.mean(), scores.std() * 2))

        elif method == 'decision_tree':
            clf = tree.DecisionTreeClassifier()
            if debug:
                scores = model_selection.cross_val_score(clf, my_train_data, my_train_label, cv=5)
                print('decision tree accuracy: %0.3f (+/- %0.3f)' % (scores.mean(), scores.std() * 2))

        elif method == 'random_forest':
            clf = RandomForestClassifier(n_estimators=100)
            if debug:
                scores = model_selection.cross_val_score(clf, my_train_data, my_train_label, cv=5)
                print('random forest(%d) accuracy: %0.3f (+/- %0.3f)' % (100, scores.mean(), scores.std() * 2))

        elif method == 'logistic_regression':
            clf = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                                                  C=1, fit_intercept=True, intercept_scaling=1.0,
                                                  class_weight=None, random_state=None)
            if debug:
                scores = model_selection.cross_val_score(clf, my_train_data, my_train_label, cv=5)
                print('logistic regression accuracy: %0.3f (+/- %0.3f)' % (scores.mean(), scores.std() * 2))

        elif method == 'naive_bayes':
            # only non-negative values, not usable for word2vec
            #clf = MultinomialNB(alpha=0.01)
            clf = GaussianNB()
            if debug:
                scores = model_selection.cross_val_score(clf, my_train_data, my_train_label, cv=5)
                print('naive bayes(alpha=%0.3f) accuracy: %0.3f (+/- %0.3f)' % (0.01, scores.mean(), scores.std() * 2))

        elif method == 'svm':
            clf = svm.SVC(C=10.0)
            if debug:
                scores = model_selection.cross_val_score(clf, my_train_data, my_train_label, cv=5)
                print('svm(C=%.1f) accuracy: %0.3f (+/- %0.3f)' % (10.0, scores.mean(), scores.std() * 2))

        elif method == 'gradient_boosting':
            clf = GradientBoostingClassifier(n_estimators=100)
            if debug:
                scores = model_selection.cross_val_score(clf, my_train_data, my_train_label, cv=5)
                print('gradient boosting(%d) accuracy: %0.3f (+/- %0.3f)' % (100, scores.mean(), scores.std() * 2))

        predicted = model_selection.cross_val_predict(clf, my_train_data, my_train_label, cv=5)
        if debug: print(metrics.accuracy_score(my_train_label, predicted))
        clf.fit(my_train_data, my_train_label)
        my_test_label = clf.predict(my_test_data)
        self.save_data(my_test_label, file_name)
        return my_test_label

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def feature_avg(self, words, model, num_features):
        feature_vec = np.zeros((num_features,), dtype="float32")
        nwords = 0.
        index2word_set = set(model.index2word)

        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                feature_vec = np.add(feature_vec, model[word])

        feature_vec = np.divide(feature_vec, nwords)
        return feature_vec

    def text_feature_avg(self, texts, model, num_features):
        counter = 0.
        text_feature_vecs = np.zeros((len(texts), num_features), dtype="float32")

        for text in texts:
            text_feature_vecs[counter] = self.feature_avg(text, model, num_features)
            counter = counter + 1.

        return text_feature_vecs

    def create_bag_of_centroids(self, wordlist, word_centroid_map):
        num_centroids = max(word_centroid_map.values()) + 1
        bag_of_centroids = np.zeros(num_centroids, dtype="float32")

        for word in wordlist:
            if word in word_centroid_map:
                index = word_centroid_map[word]
                bag_of_centroids[index] += 1
        print(bag_of_centroids)
        return bag_of_centroids