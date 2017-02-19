import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from tools import *

def main():
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    t = tools()

    logger.info('Read train data...')
    train_o = pd.read_csv('data/in/en_sentiment.tsv', header=0, delimiter='\t', quoting=3)
    train = train_o.loc[(train_o['sentiment'] != 'neutral') & (train_o['sentiment'] != 'na'), ['sentiment','tweet']]
    train['sent_num'] = train.apply(lambda train: 1 if train['sentiment'] == 'positive' else -1, axis=1)

    clean_train_tweets = []
    logger.info('Cleaning and parsing the training set...')
    for index, row in train.iterrows():
        clean_train_tweets.append(t.to_words(row['tweet'], True, True, True))

    logger.info('Creating the tf-idf of words from training set...')
    vectorizer_tfidf = TfidfVectorizer(min_df=1)
    train_data_features = vectorizer_tfidf.fit_transform(clean_train_tweets)
    train_data_features = train_data_features.toarray()
    logger.debug(train_data_features.shape)

    logger.info('Read the test data...')
    test_1 = pd.read_csv('data/in/experiment-1_twitter.csv', header=0, delimiter=',', quoting=0)
    test_2 = pd.read_csv('data/in/experiment-2_twitter.csv', header=0, delimiter=',', quoting=0)
    frames = [test_1, test_2]
    test = pd.concat(frames)

    clean_test_tweets = []
    logger.info('Cleaning and parsing the test set ...')
    for index, row in test.iterrows():
        clean_test_tweets.append(t.to_words(row['text'], True, True, True))

    logger.info('Creating the tf-idf of words from test set...')
    test_data_features = vectorizer_tfidf.transform(clean_test_tweets)
    test_data_features = test_data_features.toarray()
    logger.debug(test_data_features.shape)

    ml_methods = ['knn', 'decision_tree', 'random_forest', 'logistic_regression', 'naive_bayes', 'svm', 'gradient_boosting']
    for ml_method in ml_methods:
        logger.info('Prediction with %s...' % (ml_method))
        file_name = 'data/out/tfidf_%s_score.csv' % (ml_method)
        t.ml_classify(ml_method, train_data_features, train['sent_num'], test_data_features, file_name,True)

if __name__ == '__main__':
    main()