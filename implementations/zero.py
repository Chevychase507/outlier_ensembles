import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


"""
Designed for one hot encoded data sets
"""

class Zero:

    def __init__(self, t=None, n=None):
        self.t = t #number of probability tables
        self.n = n #subsample size
        self.m = 2 #m, standard subspace size
        self.q = None #number of dimensions
        self.H = [None] * self.t # the list of probability tables


    def fit(self, X):
        """
        Takes a 2d np array containing the training data set.
        Fits the model and builds t probability tables.
        """
        if isinstance(X, (list,tuple,np.ndarray)):
            X = np.asarray(X)
        else:
            raise TypeError('X must be an array or a tuple')

        X = np.unique(X, axis=0)

        if len(X) < self.n:
            self.n = len(X)

        X_cp = X.copy()
        np.random.shuffle(X_cp)

        self.q = len(X_cp[0])
        for i in range(self.t):
            N_i = X_cp[:self.n].copy()
            indices = self.gen_indices()
            R_mark_2 = self.gen_subspaces()
            self.H[i] = self.gen_prob_table(indices, N_i, R_mark_2)
            np.random.shuffle(X_cp)



    def gen_prob_table(self, indices, N_i, R_mark_2):
        """
        Takes the indices, the sample N_i and R_mark_2. For each indices pair,
        its feature space is iterated over and checked whether it is contained
        in N_i.
        """
        h_i = {}
        for i_pair in indices:
            N_i_subspace = np.vstack((N_i[:,i_pair[0]], N_i[:,i_pair[1]])).T #the transpose of the vertical stack
            S_i = R_mark_2 #nonsense
            #S_i_tup = [None] * len(S_i)
            S_i_tup = {}

            for j in range(len(S_i)):
                for k in range(len(N_i_subspace)):
                    if tuple(N_i_subspace[k]) == tuple(S_i[j]):
                        #S_i_tup[j] = (S_i[j],1)
                        S_i_tup[tuple(S_i[j])] = 0 #not zero appearances of N_i in S_i
                        break

                    else:
                        #S_i_tup[j] = (S_i[j], 0)
                        S_i_tup[tuple(S_i[j])] = 1 #zero appearances of N_i in S_i
            h_i[i_pair] = S_i_tup


        return h_i


    def predict(self, X):
        """
        Takes a 2d numpy array with the test set. Scores the rows in the test set
        according to the probability tables in H.
        """
        if isinstance(X, (list,tuple,np.ndarray)):
            X = np.asarray(X)
        else:
            raise TypeError('X must be an array or a tuple')

        if self.H[0] == None:
            raise ValueError('Models must be trained with fit() before score() is called')

        scores = [None] * len(X)

        for i in range(len(X)):
            scores[i] = self.score_instance(X[i])

        #the score is negated to cater for the roc auc function
        #return np.array(scores)
        return np.negative(scores)


    def score_instance(self, X_i):
        """
        Scores an instance X_i according to how many zero appearances is has
        """
        outlier_score = 0
        for i in range(self.t):
            indices = list(self.H[i].keys())
            for j in range(len(indices)):
                X_i_sub = tuple([X_i[indices[j][0]], X_i[indices[j][1]]])
                outlier_score += self.H[i][indices[j]][X_i_sub]

        return outlier_score



    def gen_indices(self):
        """
        Generates the indices which are used to generate subspaces in R'2
        and h_i. Returns a list of tuples with random indices, where each
        index appear exactly twice. Hardcoded for m = 2.
        """
        q_indices = [x for x in range(self.q)]
        #shuffle indices and zip them with themselves displaced by 1
        np.random.shuffle(q_indices)
        return list(zip(q_indices, q_indices[1:] + [q_indices[0]]))


    def gen_subspaces(self):
        """
        Returns the possible feature space for all S with m=2 and domain of
        each feature being (0,1)
        """

        return np.array([[0,0],
                      [0,1],
                      [1,0],
                      [1,1]])


"""
if __name__ == '__main__':
    path1 = sys.argv[1]

    df_bin = pd.read_csv(path1).sample(frac=1, random_state=700)
    X = np.array(df_bin.drop("label", axis=1))
    labels = np.array(df_bin['label'])

    labels[labels == -1] = 0
    print(labels)
    t = 50
    psi = 2 ** 3
    zero = Zero(t, psi)
    zero.fit(X)
    scores = zero.score(X)


    auc = roc_auc_score(labels, scores)
    neg_auc = roc_auc_score(labels, np.negative(scores))

    print("psi:", psi, ", auc:", auc, "neg auc:", neg_auc)
"""
