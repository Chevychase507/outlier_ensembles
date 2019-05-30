import numpy as np
import pandas as pd
from scipy import stats
import math
import sys
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold

#some sklearn methods print annoying warnings 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def sort_by_variance(df, feats, t, desc):
    """
    Takes a df, athreshold t, a list of features and a boolean deciding on the
    direction of the sort. Sorts the list of features by their variance in
    descending/ascending order. It returns a sorted list of feats whose variance
    is above t.
    """
    var_list, cand_feats = [],[]

    for i in range(len(feats)):
        col = np.array(df[feats[i]])
        var = np.var(col)
        if var > t:
            var_list.append(var)
            cand_feats.append(feats[i])

    sorted_feats = [x for _,x in sorted(zip(var_list,cand_feats), reverse=desc)]

    return sorted_feats



def top_k(pred_i, the_labels, k):
    """
    Takes a 1d list of anomaly scores, the label and the number of true
    outliers in the data set k. Returns the ratio of outliers in the top k
    anomaly scores.
    """
    tmp_lab = the_labels
    sort_lab = np.array([x for _,x in sorted(zip(pred_i,tmp_lab), reverse=True)])
    sort_pred = sorted(pred_i, reverse=True)
    cnts = (sort_lab[:k] == -1).sum()

    return cnts / k


def sign_change(labels):
    """
    Takes a list of labels and changes their signs. Returns the sing changed
    labels.
    """
    changed = np.zeros(len(labels))
    changed[np.where(labels == 1)] = -1
    changed[np.where(labels == -1)] = 1
    return changed


def comb_by_avg(preds):
    """
    Takes a 2d array of anomaly scores and returns the average values
    in a 1d array.
    """
    return np.mean(preds, axis=0)

def comb_by_min(preds):
    """
    Takes a 2d array of anomaly scores and returns the min (works as max
    for negative values) values in a 1d array.
    """
    return np.min(preds, axis=0)


def comb_by_thresh(z_preds):
    """
    Takes a 2d array of z-scored anomaly scores and returns the average values
    of all values above 0 in a 1d array.
    """
    thresh = z_preds
    thresh[thresh < 0] = 0
    return np.mean(thresh, axis=0)


def extreme_vals(preds, mu):
    """
    Takes a 2d array of z-scored anomaly scores and returns the average values
    of all values above above or below mu in a 1d array.
    """
    ext_preds = [None] * len(preds)

    for i in range(len(preds)):
        std = np.std(preds[i]) * mu
        tmp = preds[i]
        tmp[(tmp < std) & (tmp > -std)] = 0
        ext_preds[i] = tmp

    return ext_preds



def z_score(preds):
    """
    Takes a 2d np array of anomaly scores. Returns the z-score normalized
    anomaly scores
    """

    z_preds = [None] * len(preds)
    for i in range(len(preds)):
        if len(set(preds[i])) <= 1: # check if all elements are equal
            z_preds[i] = np.zeros(len(preds[i]))
        else:
            z_preds[i] = stats.zscore(preds[i])

    return np.array(z_preds)


def rank(preds):
    """
    Takes a 2d np array of anomaly scores. Returns the ranks of the anomaly scores
    """

    ranks = [None] * len(preds)
    for i in range(len(preds)):
        v = preds[i]
        temp = v.argsort()#[::-1]
        ranks[i] = temp.argsort()
        ranks[i] +=1

    return ranks


def min_max(preds):
    """
    Takes a 2d np array of anomaly scores. Returns the min-max normalized
    anomaly scores.
    """
    norm_preds = [None] * len(preds)
    for i in range(len(preds)):
        v = preds[i]
        if np.max(v) == np.min(v):
            norm_preds[i] = v
        else:
            norm_preds[i] = (v - np.min(v)) / (np.max(v) - np.min(v))

    return np.array(norm_preds)


def loc_thresh(preds):
    """
    Takes a 2d np array of anomaly scores. Returns the predicted class
    of the instance according to the decision function based on
    Cantellis inequality.
    """

    loc_ = [None] * len(preds)
    for i in range(len(preds)):
        can_pred[i] = decision_func(preds[i], 2)
    return can_pred

def decision_func(preds_i, k):
    """
    Takes a 1d array of anomaly scores. Each score that is below k std. dev from
    the mean is set to -1, and the rest is set to 1. Returns the binary
    predicted scores.
    """

    preds_i = np.array(preds_i)
    mean = np.median(preds_i)
    std_dev = np.std(preds_i)

    idx = np.where(preds_i < (mean - k * std_dev))
    result = np.full(len(preds_i), 1)
    result[idx] = -1

    return result


def majority_vote(can_preds):
    """
    Takes a 2d array of binary anomaly scores. Combines the scores by majority
    voting. Returns a 1d array of the majority votes
    """

    if len(can_preds) % 2 == 0:
        raise ValueError("can_preds should contain an unequal number of predictions")

    majority = [None] * len(can_preds[0])
    for i in range(len(can_preds[0])):
        anorm = 0
        norm = 0
        for j in range(len(can_preds)):
            if can_preds[j][i] == 1:
                norm += 1
            else:
                anorm += 1

        if norm > anorm:
            majority[i] = 1
        else:
            majority[i] = -1
    return majority





def param_search(data, labels, k, Estimator, model_name):
    """
    Hyperparameter exhaustive grid search method for ocsvm, zero++, iForest and
    LOF. Takes a data set, the labels, the estimator and the model name.
    Performs an exhaustive search through all combinations of hyperparameters
    and returns the best combination along with the PR AUC score.
    """

    #choose the searched hyperparameters
    params = {'t':[10,20,30,40,50,60], 'n':[2**x for x in range(1,9)]}
    if model_name == 'iForest':
        params = {'t':[10,20,30,40,50,60,70,80,90], 'n':[2**x for x in range(1,11)]}
    elif model_name == 'LOF':
        params = {'t':[10,20,30,40,50,60,70,80,90], 'n':['minkowski', 'jaccard', 'sokalsneath']} #not pretty
    elif model_name == 'ocsvm':
        params = {'t':[2**-x for x in range(0,10)] , 'n': ['linear', 'poly', 'rbf', 'sigmoid'], "nu":[x/10 for x in range(1,10)]} #even uglier


    kf = StratifiedKFold(n_splits=k)
    labels = sign_change(labels)

    best_auc = 0
    best_params = (np.inf, np.inf)
    curr_params = None
    predictions = None

    #iterate through all combinations
    for i in range(len(params['t'])):
        print(model_name, "best_params:", best_params, round(best_auc, 3), "curr_params:", curr_params)
        for j in range(len(params['n'])):
            curr_params = (params['t'][i], params['n'][j])

            #if the candidate value is larger than the data sample --> break
            if model_name != 'LOF' and model_name !='ocsvm':
                if params['n'][j] >= (len(data) - len(data) / k):
                    break
            sum_auc = 0
            cnt = 0

            #special third loo+ for ocsvm since it has three parameters
            if model_name == 'ocsvm':
                for l in range(len(params['nu'])):
                    sum_auc = 0
                    for train_idx, test_idx in kf.split(data, labels):
                        estimator = Estimator(kernel=params['n'][j], gamma=params['t'][i], nu=params['nu'][l])
                        estimator.fit(data[train_idx])
                        predictions = estimator.score_samples(data[test_idx])
                        p, r, _ = precision_recall_curve(labels[test_idx], np.negative(predictions))
                        pr_auc = auc(r, p)
                        sum_auc += pr_auc
                    avg_auc = sum_auc / k
                    if avg_auc > best_auc:
                        best_auc = avg_auc
                        best_params = (params['t'][i], params['n'][j], params['nu'][l])

            else:

                for train_idx, test_idx in kf.split(data, labels):
                    if model_name == 'iForest':
                        estimator = Estimator(n_estimators=params['t'][i], max_samples=params['n'][j], behaviour='new')
                        estimator.fit(data[train_idx])
                        predictions = estimator.score_samples(data[test_idx])
                    elif model_name == 'LOF':
                        estimator = Estimator(n_neighbors=params['t'][i], metric=params['n'][j], novelty=True)
                        estimator.fit(data[train_idx])
                        predictions = estimator.predict(data[test_idx])
                    elif model_name == 'ocsvm':
                        estimator = Estimator(kernel=params['n'][j], gamma=params['t'][i], nu=0.01)
                        estimator.fit(data[train_idx])
                        predictions = estimator.score_samples(data[test_idx])
                    else:
                        estimator = Estimator(params['t'][i], params['n'][j])
                        estimator.fit(data[train_idx])
                        predictions = estimator.predict(data[test_idx])
                    p, r, _ = precision_recall_curve(labels[test_idx], np.negative(predictions))
                    pr_auc = auc(r, p)

                    cnt += 1
                    sum_auc += pr_auc

            avg_auc = sum_auc / k

            if avg_auc > best_auc:
                best_auc = avg_auc
                best_params = (params['t'][i], params['n'][j])
            elif avg_auc == best_auc:
                if best_params[0] > params['t'][i] and best_params[1] > params['n'][j]:
                    best_params = (params['t'][i], params['n'][j])

        if best_auc == 1:
            return best_params, best_auc

    return best_params, best_auc


def param_search_fpof(data, labels, k, Estimator):
    """
    Hyperparameter exhaustive grid search method for FPOF. Takes a data set,
    the labels and the estimator. Performs an exhaustive search through all
    combinations of hyperparameters and returns the best combination along
    with the PR AUC score.
    """

    params = {'t':[10,20,30,40,50], 'n':[2**x for x in range(1,11)],
    'm':[5,10,15,20,25,30]}

    labels = sign_change(labels)

    kf = StratifiedKFold(n_splits=k)
    best_auc = 0
    best_params = (np.inf, np.inf)
    curr_params = None
    for i in range(len(params['t'])):
        for j in range(len(params['n'])):
            print("fpof, best_params:", best_params, round(best_auc, 3), "curr params:", curr_params)
            sum_auc = 0
            #if the data set contains more than 80 unique literals, it is searched for m.
            if len(np.unique(data)) > 80:
                for l in range(len(params['m'])):
                    sum_auc = 0
                    curr_params = (params['t'][i], params['n'][j], params['m'][l])
                    for train_idx, test_idx in kf.split(data, labels):

                        estimator = Estimator(params['t'][i], params['n'][j], params['m'][l])
                        pattern_sets = estimator.fit(data[train_idx])
                        predictions = estimator.predict(data[test_idx], pattern_sets)
                        p, r, _ = precision_recall_curve(labels[test_idx], np.negative(predictions))
                        pr_auc = auc(r, p)
                        sum_auc += pr_auc

                    avg_auc = sum_auc / k
                    if avg_auc > best_auc:
                        best_auc = avg_auc
                        best_params = curr_params
                if best_auc == 1:
                    return best_params, best_auc

            else:
                curr_params = (params['t'][i], params['n'][j])
                for train_idx, test_idx in kf.split(data, labels):
                    estimator = Estimator(params['t'][i], params['n'][j])
                    pattern_sets = estimator.fit(data[train_idx])
                    predictions = estimator.predict(data[test_idx], pattern_sets)

                    sum_auc += roc_auc_score(labels[test_idx], np.negative(predictions)) # if bad results, flip predictions
            avg_auc = sum_auc / k
            if avg_auc > best_auc:
                best_auc = avg_auc
                best_params = curr_params
            elif avg_auc == best_auc:
                if best_params[0] > params['t'][i] and best_params[1] > params['n'][j]:
                    best_params = curr_params

            #return the first combination that reaches a score of 1.
            if best_auc == 1:
                return best_params, best_auc

    return best_params, best_auc
