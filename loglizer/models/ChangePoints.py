import numpy as np
from sklearn import tree
import sys
sys.path.append('..')
from ..utils import metrics




"""
The implementation of the decision tree model for anomaly detection.

Authors: 
    LogPAI Team

Reference: 
    [1] Mike Chen, Alice X. Zheng, Jim Lloyd, Michael I. Jordan, Eric Brewer. 
        Failure Diagnosis Using Decision Trees. IEEE International Conference 
        on Autonomic Computing (ICAC), 2004.

"""

import numpy as np
from sklearn import tree
from ..utils import metrics
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
from sklearn import utils

class ChangePoints(object):

    def __init__(self, criterion='gini', max_depth=None, max_features=None, class_weight=None):
        """ The Invariants Mining model for anomaly detection
        Arguments
        ---------
        See DecisionTreeClassifier API: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

        Attributes
        ----------
            classifier: object, the classifier for anomaly detection

        """
        self.classifier = RandomForestClassifier(max_depth=None, n_estimators=100, criterion='gini', max_features='log2')

    def fit(self, X, y):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """
        print('====== Model summary ======')
        self.classifier.fit(X, y)

    def predict(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """

        y_pred = self.classifier.predict(X)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1
