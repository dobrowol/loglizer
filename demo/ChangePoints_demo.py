#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models.ChangePoints import ChangePoints
from loglizer import dataloader, preprocessing

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS_int(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='uniform')

    feature_extractor = preprocessing.Doc2VecVectorizer()
    y_train,x_train = feature_extractor.fit_transform(x_train, y_train)
    y_test,x_test = feature_extractor.transform(x_test, y_test)

    model = ChangePoints()
    model.fit(x_train, y_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)
