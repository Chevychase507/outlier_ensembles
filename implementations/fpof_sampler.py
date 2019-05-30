import sys
import numpy as np
import pandas as pd
from fim import fpgrowth
from sklearn.metrics import confusion_matrix, roc_auc_score
import time

class FPOFSampler:
    """
    The class FPOFSampler implements the the F-P Outlier with a sampling
    extension that makes it scale better. It requires a data set of unique
    strings. Original F-P Outlier paper:
    https://pdfs.semanticscholar.org/5f79/d24667a5fc9584c3687569b3b2a75c4f22a7.pdf

    """

    def __init__(self, t, n, m=None):
        """
        Takes t, the number of samples, n, the sample size and possibly m,
        the subspace size as hyperparameters
        """
        self.t = t

        self.min_sup = 10
        self.n = n #pattern sample size
        self.m = m #subspace sample size
        self.epochs = 5
        self.pattern_sets = None


    def fit(self, X):
        """
        Takes the data set X,  and fits the model by finding the frequent patterns
        for t samples of size n. Returns the sets of frequent patterns.
        """
        if len(np.unique(X)) <= 3:
            raise ValueError("Input data is not in distinct format")
        if len(np.unique(X)) > 80 and self.m == None:
            print("There are more than 80 unique values in X. Consider subspace sampling")

        pattern_sets = None

        if self.m is not None:
            pattern_sets = []
            pattern_sum = 0
            for i in range(self.t):
                subspace = np.random.choice(len(X[0]), self.m, replace=False)
                FPS_sub = self.find_patterns(X[:,subspace], self.min_sup)
                pattern_sum += len(FPS_sub)
                for j in range(self.epochs):
                    np.random.shuffle(FPS_sub)
                    pattern_sets.append(FPS_sub[:self.n])

        else:
            pattern_sets = [None] * self.t
            FPS = self.find_patterns(X, self.min_sup)
            for i in range(self.t):
                np.random.shuffle(FPS)
                pattern_sets[i] = FPS[:self.n]

        return pattern_sets


    def find_patterns(self, X_i, min_sup):
        """
        Returns the frequent patterns for a sample X_i.
        """
        return fpgrowth(X_i, target='s', supp=min_sup, zmin=1, zmax=5, report='S')


    def score_sample(self, x, patterns_i):
        """
        Calculates and returns the score of an instance x given patterns i.
        """
        if len(patterns_i) == 0:
            return 0
        score_sum = 0

        for i in range(len(patterns_i)):
            if set(patterns_i[i][0]) <= set(x):
                score_sum += patterns_i[i][1]
        return score_sum/len(patterns_i)


    def predict(self, X, pattern_sets):
        """
        Gives each instance in X an anomaly score based on the pattern sets.
        Returns an array of scores for X.
        """

        scores = [None] * len(X)
        for i in range(len(X)):
            score_sum = 0
            for j in range(len(pattern_sets)):
                score = self.score_sample(X[i], pattern_sets[j])
                score_sum += score

            scores[i] = score_sum /len(pattern_sets)

        return np.array(scores)
