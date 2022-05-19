#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
import pandas as pd
from loglizer.models.ChangePoints import ChangePoints
from loglizer import dataloader, preprocessing

run_models = ['ChangePoints']
struct_log = '../data/HDFS/HDFS.npz' # The benchmark dataset

if __name__ == '__main__':

    (x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS_int(struct_log,
                                                           window='session', 
                                                           train_ratio=0.5,
                                                           split_type='uniform')
    benchmark_results = []
    for _model in run_models:
        print('Evaluating {} on HDFS:'.format(_model))
        if _model == 'PCA':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf', 
                                                      normalization='zero-mean')
            model = PCA()
            model.fit(x_train)

        elif _model == 'ChangePoints':
            feature_extractor = preprocessing.Doc2VecVectorizer()
            y_train, x_train = feature_extractor.fit_transform(x_tr, y_train)
            model = ChangePoints()
            model.fit(x_train,y_train) 

        
        y_test, x_test = feature_extractor.transform(x_te, y_test)
        print('Train accuracy:')
        precision, recall, f1 = model.evaluate(x_train, y_train)
        benchmark_results.append([_model + '-train', precision, recall, f1])
        print('Test accuracy:')
        precision, recall, f1 = model.evaluate(x_test, y_test)
        benchmark_results.append([_model + '-test', precision, recall, f1])

    pd.DataFrame(benchmark_results, columns=['Model', 'Precision', 'Recall', 'F1']) \
      .to_csv('benchmark_result.csv', index=False)
