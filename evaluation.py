import logging
import pandas as pd
from ggplot import *
from sklearn import metrics
import numpy as np
from tools import *
import matplotlib.pyplot as plt

def main(model):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    t = tools()

    logger.info('Read the test data...')
    test_1 = pd.read_csv('data/in/experiment-1_twitter.csv', header=0, delimiter=',', quoting=0)
    test_2 = pd.read_csv('data/in/experiment-2_twitter.csv', header=0, delimiter=',', quoting=0)
    frames = [test_1, test_2]
    test = pd.concat(frames)

    ml_methods = ['knn', 'decision_tree', 'random_forest', 'logistic_regression', 'naive_bayes', 'svm']
    #ml_methods = ['knn', 'logistic_regression']

    logger.info('Read ML methods scores...')
    scores = {}
    scores['human'] = test['sent_num'].tolist()
    for ml_method in ml_methods:
        file_name = 'data/out/%s_%s_score.csv' % (model, ml_method)
        csv = pd.read_csv(file_name, header=None, delimiter=',', quoting=3)
        scores[ml_method] = csv[0].tolist()

    logger.info('Calculation ROC and AUC...')
    lfpr = []
    ltpr = []
    lml_methods = []
    for ml_method in ml_methods:
        logger.debug(ml_method)
        fpr, tpr, _ = metrics.roc_curve(scores['human'], scores[ml_method])
        auc = metrics.auc(fpr, tpr)
        lfpr.extend(fpr)
        ltpr.extend(tpr)
        lml_methods.extend(['%s, auc=%s %%' % (ml_method, round(auc * 100,2))] * len(fpr))

    df = pd.DataFrame({'fpr': lfpr, 'tpr': ltpr, 'ml_method': lml_methods})

    g = ggplot(df,aes(x='fpr', y='tpr')) + geom_line() + \
        facet_wrap('ml_method') + geom_abline(linetype='dashed') + \
        xlab("False positive rate") + ylab("True positive rate")
    #g.show()
    #file_name = 'data/out/%s_roc.png' % (model)
    #g.save(filename = file_name, width = 8, height = 9)


    #logger.debug(sum(1 for score in scores['human'] if score == -1))
    #logger.debug(sum(1 for score in scores['human'] if score == 1))

    logger.info('Calculaion accuracy and confusion matrices...')
    cnf_df = pd.DataFrame()
    for ml_method in ml_methods:
        accuracy = metrics.accuracy_score(scores['human'],scores[ml_method])
        cnf = metrics.confusion_matrix(scores['human'],scores[ml_method], labels=[-1,1])
        np.set_printoptions(precision=2)
        logger.info('%s accuracy:%s %%' % (ml_method, round(accuracy * 100,2)))

        #logger.debug(sum(1 for score in scores[ml_method] if score == -1))
        #logger.debug(sum(1 for score in scores[ml_method] if score == 1))

        #logger.debug(cnf)
        cnf_l = []
        for i in range(0,2):
            for j in range(0,2):
                cnf_l.append(cnf.tolist()[i][j])

        df = pd.DataFrame({'class.act':['negative', 'negative', 'positive', 'positive'],
                           'class.pred':['negative', 'positive', 'negative', 'positive'],
                           'cnf':cnf_l, 'accuracy': ['%s accuracy: %s %%' % (ml_method, round(accuracy * 100,2))] * 4,
                           'ml_method': [ml_method] * 4})

        #logger.debug(df)
        cnf_df = cnf_df.append(df)

        # plt.figure()
        # t.plot_confusion_matrix(cnf,['negative','positive'], normalize=False,
        #                        title= '%s, accuracy=%s %%' % (ml_method, round(accuracy * 100,2)))
        # plt.show()

    logger.debug(cnf_df)
    file_name = 'data/out/%s_cnf.csv' % (model)
    cnf_df.to_csv(path_or_buf=file_name, index=False)
    logger.debug("%s file saved" % (file_name))





if __name__ == '__main__':
    main('w2v')