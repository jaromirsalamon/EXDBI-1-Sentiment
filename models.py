from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim import models, similarities, matutils
from pandas import *
import nltk.data
import logging
from tools import *
import pprint
import time
from sklearn.cluster import KMeans

def main(m):
    t = tools()
    pp = pprint.PrettyPrinter(indent=1)
    corpus = [
        'this is the first document',
        'this is the second document',
        'and the third one',
        'is this the first document',
    ]

    if m == 'bow':
        print("Vectorizer Bag of Words...")
        vectorizer_bag = CountVectorizer(min_df=1)
        features_bag = vectorizer_bag.fit_transform(corpus)
        features_bag = features_bag.toarray()
        print(features_bag.shape)

        df = pandas.DataFrame(features_bag, columns=vectorizer_bag.get_feature_names(), index=range(1,len(corpus)+1))
        print(df)

        print(dict(zip(vectorizer_bag.get_feature_names(), np.sum(features_bag, axis=0))))

    elif m == 'tdifd':
        print("Vectorizer Tfidf...")
        vectorizer_tfidf = TfidfVectorizer(min_df=1)
        features_tfidf = vectorizer_tfidf.fit_transform(corpus)
        features_tfidf = features_tfidf.toarray()
        print(features_tfidf.shape)

        df = pandas.DataFrame(np.round(features_tfidf,3), columns=vectorizer_tfidf.get_feature_names(), index=range(1,len(corpus)+1))
        print(df)

        print(dict(zip(vectorizer_tfidf.get_feature_names(), vectorizer_tfidf.idf_)))

    elif m == 'w2v_avg':
        print("Vectorizer word2vec AVG...")
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        sentences = []
        for c in corpus:
            sentences += t.to_sentences(c, tokenizer)
        #pp.pprint(sentences)

        model = models.word2vec.Word2Vec(sentences, min_count=1)
        print(model.syn0.shape)

        words = model.index2word


        for word in words:
            if word == 'and':
                None
                #pp.pprint(model[word])

        print()
        for word1 in words:
            for word2 in words:
                None
                #pp.pprint("%s to %s: %f" % (word1, word2, model.similarity(word1, word2)))

        clean_corpus = []
        for c in corpus:
            clean_corpus.append(t.to_wordlist(c, remove_stopwords=True))

        print(t.text_feature_avg(clean_corpus, model, 100))
    elif m == 'w2v_boc':
        print("Vectorizer word2vec BOC...")
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        sentences = []
        for c in corpus:
            sentences += t.to_sentences(c, tokenizer)
        # pp.pprint(sentences)

        model = models.word2vec.Word2Vec(sentences, min_count=1)

        start = time.time()  # Start time
        word_vectors = model.syn0
        print("Number of words: %d" % (word_vectors.shape[0]))
        num_clusters = int(word_vectors.shape[0] / 3)
        print("Number of clusters: %d" % (num_clusters))

        print("Running K means")
        kmeans_clustering = KMeans(n_clusters=num_clusters)
        idx = kmeans_clustering.fit_predict(word_vectors)
        print("KMeans clustering indexes %s" % (idx))

        end = time.time()
        elapsed = end - start
        print("Time taken for K Means clustering: ", elapsed, "seconds.")

        word_centroid_map = dict(zip(model.index2word, idx))
        print("word centroid map %s" % (word_centroid_map))

        clean_corpus = []
        for c in corpus:
            clean_corpus.append(t.to_wordlist(c, remove_stopwords=False))
        pp.pprint(clean_corpus)

        centroids = np.zeros((len(corpus), num_clusters), dtype="float32")

        counter = 0
        for c in clean_corpus:
            centroids[counter] = t.create_bag_of_centroids(c, word_centroid_map)
            counter += 1

        pp.pprint(centroids)

if __name__ == '__main__':
    main('w2v_boc')