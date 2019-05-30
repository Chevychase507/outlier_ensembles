import sys
import numpy as np
import pandas as pd
from fim import fpgrowth
from sklearn.metrics import confusion_matrix, roc_auc_score
import time

class FPOFSampler:

    def __init__(self, t, n, m=None, epochs=5, min_sup=10, subspace_sampling=False):
        #print("t:", t, ", sample size:", n, ", epochs:", epochs, ", m:", m)
        self.t = t

        self.min_sup = min_sup
        self.n = n #pattern sample size
        self.m = m #subspace sample size
        self.epochs = epochs
        self.subspace_sampling = subspace_sampling
        self.pattern_sets = None

    def fit(self, X):
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
                for j in range(self.epochs): #how many subsamples are appropriate?
                    np.random.shuffle(FPS_sub)
                    pattern_sets.append(FPS_sub[:self.n])
            #print("subspace sampling, avg # patterns:", pattern_sum / self.t)

        else:
            pattern_sets = [None] * self.t
            FPS = self.find_patterns(X, self.min_sup)
            #print("total number of patterns:", len(FPS))
            for i in range(self.t):
                np.random.shuffle(FPS)
                pattern_sets[i] = FPS[:self.n]


        #self.pattern_sets = pattern_sets
        return pattern_sets


    def find_patterns(self, X_i, min_sup):
        return fpgrowth(X_i, target='s', supp=min_sup, zmin=1, zmax=5, report='S')


    def score_sample(self, x, patterns_i):
        if len(patterns_i) == 0:
            return 0
        score_sum = 0

        for i in range(len(patterns_i)):
            if set(patterns_i[i][0]) <= set(x):
                score_sum += patterns_i[i][1]
        return score_sum/len(patterns_i)


    def predict(self, X, pattern_sets):

        scores = [None] * len(X)
        for i in range(len(X)):
            score_sum = 0
            for j in range(len(pattern_sets)):
                score = self.score_sample(X[i], pattern_sets[j])
                score_sum += score

            scores[i] = score_sum /len(pattern_sets)

        return np.array(scores)
        #return np.negative(scores)


"""
if __name__ == '__main__':
    path = sys.argv[1]
    df = pd.read_csv(path)
    X = np.array(df.drop("label", axis=1))
    print(X.shape)
    print("unique values in X:", len(np.unique(X)))
    labels = np.array(df['label'])


    #fpof = FPOFSampler(100, 10, 10, 2, 25, True) # for high-dimensional data
    fpof = FPOFSampler(10, 100)
    pattern_sets = fpof.fit(X)
    scores = fpof.predict(X, pattern_sets)
    print(np.mean(scores[np.where(labels > 0)]))
    print(np.mean(scores[np.where(labels < 0)]))
    ind_auc = roc_auc_score(labels, scores)
    print(ind_auc, np.mean(scores))
"""
