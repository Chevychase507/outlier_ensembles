import numpy as np
import pandas as pd
from scipy import stats
import math
import sys
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def sort_by_variance(df, feats, t, desc):
    """
    Takes a df and a list of feats. It sorts the list of feats by their variance
    in descending order. It returns a sorted list of feats whose variance is above t.

    """

    var_list = []
    cand_feats = []

    for i in range(len(feats)):
        col = np.array(df[feats[i]])
        var = np.var(col)
        if var > t:
            var_list.append(var)
            cand_feats.append(feats[i])


    sorted_feats = [x for _,x in sorted(zip(var_list,cand_feats), reverse=True)]
    #sorted_feats = [x for _,x in sorted(zip(var_list,cand_feats))]

    return sorted_feats



def top_k(pred_i, the_labels, k):
    """
    Finds the ratio of outliers in the top k anomaly scores
    """
    tmp_lab = the_labels

    sort_lab = np.array([x for _,x in sorted(zip(pred_i,tmp_lab), reverse=True)])
    sort_pred = sorted(pred_i, reverse=True)

    sort_pred = sorted(pred_i, reverse=True)
    cnts = (sort_lab[:k] == -1).sum()


    return cnts / k


def sign_change(lab):
    tmp = np.zeros(len(lab))
    tmp[np.where(lab == 1)] = -1
    tmp[np.where(lab == -1)] = 1
    return tmp


def comb_by_avg(preds):
    comb = np.mean(preds, axis=0)
    return comb

def comb_by_min(preds):
    #comb = np.max(preds, axis=0)
    comb = np.min(preds, axis=0)
    return comb

def comb_by_logmean(preds):
    log_preds = np.log2(np.array(preds) + 0.000001)
    comb = np.mean(log_preds, axis=0)
    return comb

def comb_by_thresh(z_preds):
    thresh = z_preds
    thresh[thresh < 0] = 0
    comb = np.sum(thresh, axis=0)
    return comb


def extreme_vals(preds, mu):
    """
    takes z-scored only
    """
    ext_preds = [None] * len(preds)

    for i in range(len(preds)):
        std = np.std(preds[i]) * mu
        tmp = preds[i]
        tmp[(tmp < std) & (tmp > -std)] = 0
        ext_preds[i] = tmp

    return ext_preds



def z_score(preds):
    z_preds = [None] * len(preds)
    for i in range(len(preds)):
        if len(set(preds[i])) <= 1: # check if all elements are equal
            z_preds[i] = np.zeros(len(preds[i]))
        else:
            z_preds[i] = stats.zscore(preds[i])

    return np.array(z_preds)




def rank(preds):

    ranks = [None] * len(preds)
    for i in range(len(preds)):
        v = preds[i]
        temp = v.argsort()#[::-1]
        ranks[i] = temp.argsort()
        ranks[i] +=1

    return ranks


def min_max(preds):
    """
    min max normalization of multiple lists of predictions
    """
    norm_preds = [None] * len(preds)
    for i in range(len(preds)):
        v = preds[i]
        if np.max(v) == np.min(v):
            norm_preds[i] = v
        else:
            norm_preds[i] = (v - np.min(v)) / (np.max(v) - np.min(v))

    return np.array(norm_preds)


def cantelli_pred(preds):
    can_pred = [None] * len(preds)
    for i in range(len(preds)):
        can_pred[i] = decision_func(preds[i], 2)
    return can_pred

def decision_func(preds, k):
    """
    following Cantelli's inequality
    """
    preds = np.array(preds)
    mean = np.median(preds)
    std_dev = np.std(preds)
    #print(mean, std_dev, mean + k * std_dev)
    idx = np.where(preds < (mean - k * std_dev))
    result = np.full(len(preds), 1)
    result[idx] = -1

    return result


def majority_vote(cant_pred):

    if len(cant_pred) % 2 == 0:
        raise ValueError("cant_pred should contain an unequal number of predictions")

    majority = [None] * len(cant_pred[0])
    for i in range(len(cant_pred[0])):
        anorm = 0
        norm = 0
        for j in range(len(cant_pred)):
            if cant_pred[j][i] == 1:
                norm += 1
            else:
                anorm += 1

        if norm > anorm:
            majority[i] = 1
        else:
            majority[i] = -1
    return majority



def param_search_fpof(data, labels, k, Estimator):
    """
    for fpof_sampler
    """
    params = {'t':[10,20,30,40,50], 'n':[2**x for x in range(1,11)], 'm':[5,10,15,20,25,30]}

    labels = sign_change(labels)

    kf = StratifiedKFold(n_splits=k)
    best_auc = 0
    best_params = (np.inf, np.inf)
    curr_params = None
    for i in range(len(params['t'])):
        for j in range(len(params['n'])):
            print("fpof, best_params:", best_params, round(best_auc, 3), "curr params:", curr_params)
            sum_auc = 0
            if len(np.unique(data)) > 80: # why 80
                for l in range(len(params['m'])):
                    sum_auc = 0
                    curr_params = (params['t'][i], params['n'][j], params['m'][l])
                    for train_idx, test_idx in kf.split(data, labels):

                        estimator = Estimator(params['t'][i], params['n'][j], params['m'][l])
                        pattern_sets = estimator.fit(data[train_idx])
                        predictions = estimator.predict(data[test_idx], pattern_sets)
                        p, r, _ = precision_recall_curve(labels[test_idx], np.negative(predictions))
                        pr_auc = auc(r, p)


                        #auc = roc_auc_score(labels[test_idx], np.negative(predictions))
                        #sum_auc += auc
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
                    #if len(data) < 300:
                    #    predictions = np.negative(predictions)

                    #for b in range(len(predictions)):
                    #    print(label_test[b], predictions[b])
                    sum_auc += roc_auc_score(labels[test_idx], np.negative(predictions)) # if bad results, flip predictions
            avg_auc = sum_auc / k
            if avg_auc > best_auc:
                best_auc = avg_auc
                best_params = curr_params
            elif avg_auc == best_auc:
                if best_params[0] > params['t'][i] and best_params[1] > params['n'][j]:
                    best_params = curr_params

            if best_auc == 1:
                return best_params, best_auc

    return best_params, best_auc

def param_search(data, labels, k, Estimator, model_name):
    """
    for ocsvm, zero, iForest and LOF
    """
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
    for i in range(len(params['t'])):
        print(model_name, "best_params:", best_params, round(best_auc, 3), "curr_params:", curr_params)
        for j in range(len(params['n'])):
            curr_params = (params['t'][i], params['n'][j])
            if model_name != 'LOF' and model_name !='ocsvm':
                if params['n'][j] >= (len(data) - len(data) / k):
                    break
            sum_auc = 0
            cnt = 0
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

                #roc_auc = roc_auc_score(labels[test_idx], np.negative(predictions))
                    cnt += 1
                #sum_auc += roc_auc
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



def csv_results_kfold(member_list, ensemble, data, distinct_data, parameters, data_name, k):

    file_name = data_name + "_results.csv"

    with open(file_name, 'w') as out:
        out.write("algoritm,params,data_set,raw_auroc,norm_auroc,rank_auroc,raw_auprc,norm_auprc,rank_auprc\n")

        kf = StratifiedKFold(n_splits=k)
        cnt = 0
        for train_idx, test_idx in kf.split(data, labels):
            print(str(cnt) + "th epoch")
            ensemble, fpof_patterns = fit_ensemble(member_list, ensemble, data[train_idx], distinct_data[train_idx])
            preds = predict_ensemble(member_list, ensemble, data[test_idx], distinct_data[test_idx], fpof_patterns)
            norm_preds = min_max(preds)
            rank_preds = rank(preds)
            #precision, recall, thresholds = precision_recall_curve(labels[test_idx], predictions)
            #pr_auc = auc(recall, precision)

            for j in range(len(member_list)):
                raw_auroc = roc_auc_score(labels[test_idx], preds[j])
                norm_auroc = roc_auc_score(labels[test_idx], norm_preds[j])
                rank_auroc = roc_auc_score(labels[test_idx], rank_preds[j])
                p, r, t = precision_recall_curve(labels[test_idx], preds[j])
                raw_auprc = auc(r, p)
                p, r, t = precision_recall_curve(labels[test_idx], norm_preds[j])
                norm_auprc = auc(r, p)
                p, r, t = precision_recall_curve(labels[test_idx], rank_preds[j])
                rank_auprc = auc(r, p)

                out.write(member_list[j] + "," +  str(parameters[member_list[j]]).replace(",", ";") + "," +
                 data_name + "," + str(raw_auroc) + "," + str(norm_auroc) + "," + str(rank_auroc) +
                 "," + str(raw_auprc) + "," + str(norm_auprc) + "," + str(rank_auprc) + "\n")

            #various avg. strategies
            raw_auroc = roc_auc_score(labels[test_idx], comb_by_avg(preds))
            norm_auroc = roc_auc_score(labels[test_idx], comb_by_avg(norm_preds))
            rank_auroc = roc_auc_score(labels[test_idx], comb_by_avg(rank_preds))
            p, r, t = precision_recall_curve(labels[test_idx], comb_by_avg(preds))
            raw_auprc = auc(r, p)
            p, r, t = precision_recall_curve(labels[test_idx], comb_by_avg(norm_preds))
            norm_auprc = auc(r, p)
            p, r, t = precision_recall_curve(labels[test_idx], comb_by_avg(rank_preds))
            rank_auprc = auc(r, p)
            out.write("ensemble_avg" + "," +  "no ocsvm" + "," + data_name + "," + str(raw_auroc) + "," + str(norm_auroc) + "," + str(rank_auroc) +
             "," + str(raw_auprc) + "," + str(norm_auprc) + "," + str(rank_auprc) +"\n")

            #various min strategies
            raw_auroc = roc_auc_score(labels[test_idx], comb_by_min(preds))
            norm_auroc = roc_auc_score(labels[test_idx], comb_by_min(norm_preds))
            rank_auroc = roc_auc_score(labels[test_idx], comb_by_min(rank_preds))
            p, r, t = precision_recall_curve(labels[test_idx], comb_by_min(preds))
            raw_auprc = auc(r, p)
            p, r, t = precision_recall_curve(labels[test_idx], comb_by_min(norm_preds))
            norm_auprc = auc(r, p)
            p, r, t = precision_recall_curve(labels[test_idx], comb_by_min(rank_preds))
            rank_auprc = auc(r, p)
            out.write("ensemble_min" + "," +  "no ocsvm" + "," + data_name + "," + str(raw_auroc) + "," + str(norm_auroc) + "," + str(rank_auroc) +
             "," + str(raw_auprc) + "," + str(norm_auprc) + "," + str(rank_auprc) + "\n")
            cnt += 1
