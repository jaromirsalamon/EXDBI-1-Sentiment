import pandas as pd
import logging
from tools import *
import nltk.data
from gensim.models import word2vec

def main(action):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    t = tools()
    num_features = 100

    # common part
    logger.info('Read train data...')
    train_o = pd.read_csv('data/in/en_sentiment.tsv', header=0, delimiter='\t', quoting=3)
    train = train_o.loc[(train_o['sentiment'] != 'neutral') & (train_o['sentiment'] != 'na'), ['sentiment', 'tweet']]
    train['sent_num'] = train.apply(lambda train: 1 if train['sentiment'] == 'positive' else -1, axis=1)

    logger.info('Read the test data...')
    test_1 = pd.read_csv('data/in/experiment-1_twitter.csv', header=0, delimiter=',', quoting=0)
    test_2 = pd.read_csv('data/in/experiment-2_twitter.csv', header=0, delimiter=',', quoting=0)
    frames = [test_1, test_2]
    test = pd.concat(frames)

    if action == 'model':
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        sentences = []
        logger.info('Parsing sentences from training set...')
        for index, row in train.iterrows():
            sentences += t.to_sentences(row['tweet'], tokenizer)

        logger.info('Training Word2Vec model...')
        model = word2vec.Word2Vec(sentences, workers=4, size=num_features, min_count=1, window=5)
        model.init_sims(replace=True)

        model_name = "data/out/100features_1minwords_5context.model"
        model.save(model_name)
    elif action == 'avg_vectors':
        model = word2vec.Word2Vec.load("data/out/100features_1minwords_5context.model")
        logger.info(model.syn0.shape)

        clean_train_tweets = []
        logger.info('Cleaning and parsing the training set...')
        for index, row in train.iterrows():
            clean_train_tweets.append(t.to_wordlist(row['tweet'], remove_stopwords=True))

        logger.info("Creating average feature vectors for train set...")
        train_vec_features = t.text_feature_avg(clean_train_tweets, model, num_features)

        clean_test_tweets = []
        logger.info('Cleaning and parsing the test set ...')
        for index, row in test.iterrows():
            clean_test_tweets.append(t.to_wordlist(row['text'], remove_stopwords=True))

        logger.info("Creating average feature vectors for test set...")
        test_vec_features = t.text_feature_avg(clean_test_tweets, model, num_features)

        ml_methods = ['knn', 'decision_tree', 'random_forest', 'logistic_regression', 'naive_bayes', 'svm', 'gradient_boosting']

        for ml_method in ml_methods:
            logger.info('Prediction with %s...' % (ml_method))
            file_name = 'data/out/w2v_%s_score.csv' % (ml_method)
            t.ml_classify(ml_method, train_vec_features, train['sent_num'], test_vec_features, file_name, True)

    elif action == 'bag_of_centroids':
        None
    else:
        None

if __name__ == '__main__':
    main('avg_vectors')
