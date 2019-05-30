import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from implementations.utils import *
from implementations.zero import Zero
from implementations.fpof_sampler import FPOFSampler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.svm import OneClassSVM

def fit_ensemble(member_list, ensemble, data, distinct_data):
    """
    Takes the list of ensemble members and fit them with the training data.
    Returns the fitted ensemble members and the patterns from FPOF. The
    latter is necessary due to a glitch in the FP-Growth implementation.
    """
    fpof_patterns = None
    for i in range(len(ensemble)):
        if member_list[i] == 'fpof':
            fpof_patterns = ensemble[i].fit(distinct_data)
        else:
            ensemble[i].fit(data)

    return ensemble, fpof_patterns


def predict_ensemble(member_list, ensemble, data, distinct_data, fpof_patterns):
    """
    Takes the list of fitted ensemble members, the test data and the FPOF patterns.
    It predicts the anomaly scores of the test data and return an np array
    of score vectors.
    """

    predictions = [None] * len(ensemble)

    for i in range(len(ensemble)):
        print("predicting", member_list[i])
        if member_list[i] == 'LOF':
            predictions[i] = ensemble[i].score_samples(data)
        elif 'iForest' in member_list[i] or 'ocsvm' in  member_list[i]:
            predictions[i] = ensemble[i].score_samples(data)
        elif member_list[i] == 'fpof':
            predictions[i] = ensemble[i].predict(distinct_data, fpof_patterns)
        else:
            predictions[i] = ensemble[i].predict(data)
    print("**** Prediction finished *****")
    print()

    return np.array(predictions)




def find_best_params(data, distinct_data, labels, estimators, member_list, k):
    """
    Takes a data set in binary and distinct format together with the list of
    estimators. It then searches for the best hyperparameters for each method
    and returns a list of tuples containing the best parameters.

    """

    for i in range(len(member_list)):
        print("finding best params for:", member_list[i])
        if member_list[i] == 'fpof':
            #fpof needs a special hyperparameter search method
            params, auc = param_search_fpof(distinct_data, labels, k, estimators[member_list[i]])
            best_params.append((member_list[i], params, auc))

        else:
            params, auc = param_search(data, labels, k, estimators[member_list[i]], member_list[i])
            best_params.append((member_list[i], params, auc))

    return best_params


def csv_auc(member_list, ensemble, the_labels, data, distinct_data, parameters, data_name, k):

    file_name = data_name + "_results.csv"

    with open(file_name, 'w') as out:
        out.write("algoritm,params,ensemble,data_set,auroc,auprc\n")

        kf = StratifiedKFold(n_splits=k)
        cnt = 0
        for train_idx, test_idx in kf.split(data, the_labels):
            print(str(cnt) + "th epoch")
            cnt += 1
            ensemble, fpof_patterns = fit_ensemble(member_list, ensemble, data[train_idx], distinct_data[train_idx])
            preds = predict_ensemble(member_list, ensemble, data[test_idx], distinct_data[test_idx], fpof_patterns)

            preds = np.negative(preds) #flip to accommodate auc
            norm_preds = min_max(preds)
            rank_preds = rank(preds)
            z_preds = z_score(preds)
            if len(member_list) % 2 != 0:
                can_preds = cantelli_pred(preds)
            #precision, recall, thresholds = precision_recall_curve(labels[test_idx], predictions)
            #pr_auc = auc(recall, precision)

            labels_ = sign_change(the_labels[test_idx])

            for j in range(len(member_list)):
                if len(set(preds[j])) == 1:
                    raw_auroc = 0
                    raw_auprc = 0
                else:
                    raw_auroc = roc_auc_score(labels_, preds[j])
                    p, r, t = precision_recall_curve(labels_, preds[j])
                    raw_auprc = auc(r, p)

                out.write(member_list[j] + "," +  str(parameters[member_list[j]]).replace(",", ";") + "," + "0" + "," +
                data_name + "," + str(raw_auroc) + "," + str(raw_auprc) + "\n")


            #min_max --> avg
            min_max_avg_auroc = roc_auc_score(labels_, comb_by_avg(norm_preds))
            p, r, t = precision_recall_curve(labels_, comb_by_avg(norm_preds))
            min_max_avg_auprc = auc(r, p)
            out.write("E_minmax_avg" + "," +  "all_members,1" + "," + data_name + "," + str(min_max_avg_auroc) + "," + str(min_max_avg_auprc) +  "\n")

            #min_max --> max
            if len(set(comb_by_min(norm_preds))) == 1:
                min_max_max_auroc = 0
                min_max_max_auprc = 0
            else:
                min_max_max_auroc = roc_auc_score(labels_, comb_by_min(norm_preds))
                p, r, t = precision_recall_curve(labels_, comb_by_min(norm_preds))
                min_max_max_auprc = auc(r, p)
            out.write("E_minmax_max" + "," +  "all_members,1" + "," + data_name + "," + str(min_max_max_auroc) + "," + str(min_max_max_auprc) + "\n")

            #rank --> avg
            rank_avg_auroc = roc_auc_score(labels_, comb_by_avg(rank_preds))
            p, r, t = precision_recall_curve(labels_, comb_by_avg(rank_preds))
            rank_avg_auprc = auc(r, p)
            out.write("E_rank_avg" + "," +  "all_members,1" + "," + data_name + "," + str(rank_avg_auroc) + "," + str(rank_avg_auprc) + "\n")

            #rank --> max
            rank_max_auroc = roc_auc_score(labels_, comb_by_min(rank_preds))
            p, r, t = precision_recall_curve(labels_, comb_by_min(rank_preds))
            rank_max_auprc = auc(r, p)
            out.write("E_rank_max" + "," +  "all_members,1" + "," + data_name + "," + str(rank_max_auroc) + "," + str(rank_max_auprc) + "\n")

            #z_score --> avg

            z_avg_auroc = roc_auc_score(labels_, comb_by_avg(z_preds))
            p, r, t = precision_recall_curve(labels_, comb_by_avg(z_preds))
            z_avg_auprc = auc(r, p)
            out.write("E_z_avg" + "," +  "all_members,1" + "," + data_name + "," + str(z_avg_auroc) + "," + str(z_avg_auprc) + "\n")

            #z_score --> max
            z_max_auroc = roc_auc_score(labels_, comb_by_min(z_preds))
            p, r, t = precision_recall_curve(labels_, comb_by_min(z_preds))
            z_max_auprc = auc(r, p)
            out.write("E_z_max" + "," +  "all_members,1" + "," + data_name + "," + str(z_max_auroc) + "," + str(z_max_auprc) + "\n")

            #z_score --> extr vals
            z_extr_auroc = roc_auc_score(labels_, comb_by_avg(extreme_vals(z_preds, 0.3)))
            p, r, t = precision_recall_curve(labels_, comb_by_avg(extreme_vals(z_preds, 0.3)))
            z_extr_auprc = auc(r, p)
            out.write("E_z_extr" + "," +  "all_members,1" + "," + data_name + "," + str(z_extr_auroc) + "," + str(z_extr_auprc) + "\n")

            #z_score --> thresh
            z_thresh_auroc = roc_auc_score(labels_, comb_by_thresh(z_preds))
            p, r, t = precision_recall_curve(labels_, comb_by_thresh(z_preds))
            z_thresh_auprc = auc(r, p)
            out.write("E_z_thresh" + "," +  "all_members,1" + "," + data_name + "," + str(z_thresh_auroc) + "," + str(z_thresh_auprc) + "\n")

            if len(member_list) % 2 != 0:
                #loc_thres --> majority_vote
                if len(set(majority_vote(can_preds))) == 1:
                    maj_auroc = 0
                    maj_auprc = 0
                else:
                    maj_auroc = roc_auc_score(labels_, majority_vote(can_preds))
                    p, r, t = precision_recall_curve(labels_, majority_vote(can_preds))
                    maj_auprc = auc(r, p)
                out.write("E_majority" + "," +  "all_members,1" + "," + data_name + "," + str(maj_auroc) + "," + str(maj_auprc) + "\n")



def csv_topk(member_list, ensemble, the_labels, data, distinct_data, parameters, data_name, out_cnt, k):
    file_name = data_name + "_topk_results.csv"

    with open(file_name, 'w') as out:
        out.write("algoritm,params,ensemble,data_set,top_k\n")

        kf = StratifiedKFold(n_splits=k)
        cnt = 0
        for train_idx, test_idx in kf.split(data, the_labels):
            print(cnt, "th epoch at", data_name)
            cnt += 1
            out_cnt = len(labels[test_idx][labels[test_idx] == -1])

            ensemble, fpof_patterns = fit_ensemble(member_list, ensemble, data[train_idx], distinct_data[train_idx])
            preds = predict_ensemble(member_list, ensemble, data[test_idx], distinct_data[test_idx], fpof_patterns)
            preds = np.negative(preds)
            norm_preds = min_max(preds)
            rank_preds = rank(preds)
            z_preds = z_score(preds)
            if len(member_list) % 2 != 0:
                can_preds = cantelli_pred(preds)





            for i in range(len(member_list)):
                tp_k = top_k(preds[i], labels[test_idx], out_cnt)
                out.write(member_list[i] + "," +  str(parameters[member_list[i]]).replace(",", ";") + "," + "0" + "," +
                data_name + "," + str(tp_k) + "\n")

            #print(preds[4])
            #print(labels[test_idx])

            #min_max --> avg
            tp_k = top_k(comb_by_avg(norm_preds), labels[test_idx], out_cnt)
            out.write("E_minmax_avg" + "," +  "all_members,1" + "," + data_name + "," +  str(tp_k) +  "\n")

            #min_max --> max
            tp_k = top_k(comb_by_min(norm_preds), labels[test_idx], out_cnt)
            out.write("E_minmax_max" + "," +  "all_members,1" + "," + data_name + ","  + str(tp_k) + "\n")


            #rank --> avg
            tp_k = top_k(comb_by_avg(rank_preds), labels[test_idx], out_cnt)
            out.write("E_rank_avg" + "," +  "all_members,1" + "," + data_name + ","  + str(tp_k) + "\n")

            #rank --> max

            tp_k = top_k(comb_by_min(rank_preds), labels[test_idx], out_cnt)
            out.write("E_rank_max" + "," +  "all_members,1" + "," + data_name + "," + str(tp_k) + "\n")

            #z_score --> avg
            tp_k = top_k(comb_by_avg(z_preds), labels[test_idx], out_cnt)
            out.write("E_z_avg" + "," +  "all_members,1" + "," + data_name + ","  + str(tp_k) + "\n")

            #z_score --> max
            tp_k = top_k(comb_by_min(z_preds), labels[test_idx], out_cnt)
            out.write("E_z_max" + "," +  "all_members,1" + "," + data_name + "," + str(tp_k) +  "\n")

            #z_score --> extr vals
            tp_k = top_k(comb_by_avg(extreme_vals(z_preds, 0.3)), labels[test_idx], out_cnt)
            out.write("E_z_extr" + "," +  "all_members,1" + "," + data_name + "," + str(tp_k) + "\n")

            #z_score --> thresh
            tp_k = top_k(comb_by_thresh(z_preds), labels[test_idx], out_cnt)
            out.write("E_z_thresh" + "," +  "all_members,1" + "," + data_name + "," + str(tp_k) + "\n")

            if len(member_list) % 2 != 0:
                #loc_thres --> majority_vote
                tp_k = top_k(majority_vote(can_preds), labels[test_idx], out_cnt)
                pre = majority_vote(can_preds)
                out.write("E_majority" + "," +  "all_members,1" + "," + data_name + "," + str(tp_k) + "\n")




def choose_params(data_name):
    """
    BEST PARAMS for each method at each data set
    """
    if "spect" in data_name:
        return {"ocsvm":('sigmoid', 0.0078125, 0.1), "zero": (10, 128), "iForest":(10, 2, "new"), "LOF":(30, 0.1,'sokalsneath'), 'fpof':(10, 2)}
    elif "nurse" in data_name:
        return {"ocsvm":('rbf', 0.5, 0.7), "zero": (50, 8), "iForest":(80, 256, "new"), "LOF":(90, 0.1,'minkowski'), 'fpof':(10, 1024)}
    elif "chess" in data_name:
        return {"ocsvm":('rbf', 0.015625, 0.8), "zero": (60, 32), "iForest":(70, 1024, "new"), "LOF":(30, 0.1,'jaccard'), 'fpof':(40, 16)}
    elif "mushroom" in data_name:
        return {"ocsvm":('rbf', 0.5, 0.9), "zero": (40, 2), "iForest":(70, 32, "new"), "LOF":(90, 0.1,'minkowski'), 'fpof':(50, 32, 20)}
    elif "solar" in data_name:
        return {"ocsvm":('rbf', 0.5, 0.9), "zero": (40, 2), "iForest":(10, 2, "new"), "LOF":(90, 0.1,'minkowski'), 'fpof':(10, 16)}

    else:
        return {"ocsvm":('rbf', 0.0078125, 0.9), "zero": (50, 16), "iForest":(10, 2, "new"), "LOF":(60, 0.1,'sokalsneath'), 'fpof':(10, 8, 15)}


def init_members(member_list, parameters):
    members = []
    for i in range(len(member_list)):
        if 'ocsvm' == member_list[i]:
            ocsvm = OneClassSVM(kernel=parameters['ocsvm'][0], gamma=parameters['ocsvm'][1], nu=parameters['ocsvm'][2])
            members.append(ocsvm)
        elif 'zero' == member_list[i]:
            zero = Zero(parameters['zero'][0], parameters['zero'][1])
            members.append(zero)
        elif 'iForest' == member_list[i]:
            iForest = IsolationForest(n_estimators=parameters['iForest'][0],max_samples=parameters['iForest'][1], behaviour=parameters['iForest'][2])
            members.append(iForest)
        elif 'LOF' == member_list[i]:
            LOF = LocalOutlierFactor(n_neighbors=parameters['LOF'][0], contamination=parameters['LOF'][1], metric=parameters['LOF'][2], novelty=True)
            members.append(LOF)
        elif 'fpof' == member_list[i]:
            if len(parameters['fpof']) > 2:
                fpof = FPOFSampler(parameters['fpof'][0], parameters['fpof'][1], parameters['fpof'][2])
            else:
                fpof = FPOFSampler(parameters['fpof'][0], parameters['fpof'][1]) #adapt to three parameter case
            members.append(fpof)
    return members

def write_preds(preds, member_list, labels, data_name):

    file_name = data_name + " preds.csv"
    memb_str = ",".join(member_list)

    with open(file_name, 'w') as out:
        out.write("data_name," + memb_str + ",label\n")
        for i in range(len(preds[0])):
            out.write(data_name + "," + str(preds[0][i]) + "," + str(preds[1][i]) +  "," +
             str(preds[2][i]) + "," + str(preds[3][i]) + "," + str(preds[4][i]) + "," + str(labels[i]) + "\n" )



def plot_preds(preds, member_list, data_name):

    for i in range(len(member_list)):
        mean_ = np.mean(preds[i])
        #std_pos = mean_ + np.std(preds[i]) * 2
        std_neg = mean_ - np.std(preds[i]) * 2
        n, bins, patches = plt.hist(preds[i], bins='auto', facecolor='g')
        title = "Anomaly scores of " + member_list[i] + " on " + data_name
        plt.title(title)
        plt.axvline(mean_, color='k', linestyle='solid', linewidth=1)
        plt.axvline(std_neg, color='k', linestyle='dashed', linewidth=1)
        #plt.axvline(std_pos, color='k', linestyle='dashed', linewidth=1)
        plt.xlabel('Anomaly score')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


def csv_preds(preds, labels, member_list, data_name):
    """
    writes file for OCSVM results and mm-avg ensemble
    """
    file_name = data_name + "_preds.csv"

    norm_preds = min_max(np.negative(preds))


    with open(file_name, 'w') as out:
        out.write("data_name,Algorithm,Score,Label\n")
        for i in range(len(member_list)):
            for j in range(len(preds[0])):
                out.write(data_name + "," + str(member_list[i]) + "," + str(norm_preds[i][j]) + "," + str(labels[j]) + "\n")





def mush_print(preds, labels):
    preds = min_max(preds)
    ens = comb_by_avg(preds)
    for i in range(100):
        print(round(1 - preds[0][i],3), round(1 - preds[1][i],3),round( 1 - preds[2][i],3), round( 1 - preds[3][i],3), round(1 -  preds[4][i],3), round( 1 - ens[i], 3), labels[i])



def sort_forward_feed(member_list, ensemble, df_bin, df_dist, labels, out_cnt, data_name):
    sorted_feats = sort_by_variance(df_bin, list(df_bin.columns), .0)
    print(df_bin.shape)

    file_name = data_name + "_" + member_list[0] + "sort_forward_feed_ascending.csv"
    #labels = sign_change(labels)

    print(labels)

    with open(file_name, 'w') as out:
        out.write("n_feats,algorithm,auprc_mean,auprc_std,top_k_mean,top_k_std\n")

        for i in range(4,len(sorted_feats)):
            topk_results = [None] * 10
            prc_results = [None] * 10
            for m in range(10):

                data = np.array(df_bin[sorted_feats[:i]])
                distinct_data = np.array(df_dist[sorted_feats[:i]])
                print(data.shape)

                ensemble, fpof_patterns = fit_ensemble(member_list, ensemble, data, distinct_data)
                preds = predict_ensemble(member_list, ensemble, data, distinct_data, fpof_patterns)
                preds = np.negative(min_max(preds))

                p, r, _ = precision_recall_curve(labels, preds[0], pos_label=-1)
                prc_results[m] = auc(r,p)
                topk_results[m] = top_k(preds[0], labels, out_cnt)

                print(i, prc_results[m], topk_results[m])

            topk_results = np.array(topk_results)
            prc_results = np.array(prc_results)


            out.write(str(i) + "," + member_list[0] + "," + str(np.mean(prc_results)) + "," +
                 str(np.std(prc_results)) + "," + str(np.mean(topk_results)) + "," + str(np.std(topk_results)) + "\n")



def rfe(df_bin, labels, out_cnt):
    """
    Takes the binary data and does recursive feature ranking using
    the coef_ attribute from OCSVM with a linear kernel. Prints results
    before and after the features are selected. Designed for optimizing
    fraud webshops results
    """
    X = np.array(df_bin.drop("label", axis=1))

    #initialize w. fraud webshops hyperparameters
    ocsvm = OneClassSVM(kernel="rbf", gamma=0.0078125, nu=0.9)
    zero = Zero(50, 8)

    #fit and predict
    ocsvm.fit(X)
    zero.fit(X)
    o_preds = ocsvm.score_samples(X)
    z_preds = zero.predict(X)

    #normalize
    o_preds = min_max([o_preds])[0]
    z_preds = min_max([z_preds])[0]

    print("**** BEFORE **** ")
    #calc measures
    p, r, _ = precision_recall_curve(sign_change(labels), np.negative(o_preds))
    auc_ = auc(r,p)
    tpk = top_k(np.negative(o_preds), labels, out_cnt)

    print("OCSVM PR AUC and TOP K:", auc_, tpk)

    p, r, _ = precision_recall_curve(sign_change(labels), np.negative(z_preds))
    auc_ = auc(r,p)
    tpk = top_k(np.negative(z_preds), labels, out_cnt)
    print("ZERO PR AUC and TOP K:", auc_, tpk)



    estimator = OneClassSVM(kernel="linear", gamma=0.0078125, nu=0.9)
    selector = RFE(estimator, step=1)
    selector = selector.fit(X, labels)
    idx = selector.support_


    X = X[:,idx]

    print("**** AFTER **** ")

    ocsvm.fit(X)

    o_preds = ocsvm.score_samples(X)
    o_preds = min_max([o_preds])[0]

    zero.fit(X)
    z_preds = zero.predict(X)
    z_preds = min_max([z_preds])[0]

    p, r, _ = precision_recall_curve(sign_change(labels), np.negative(o_preds))
    auc_ = auc(r,p)
    tpk = top_k(np.negative(o_preds), labels, out_cnt)
    print("OCSVM:", auc_, tpk)


    p, r, _ = precision_recall_curve(sign_change(labels), np.negative(z_preds))
    auc_ = auc(r,p)
    tpk = top_k(np.negative(z_preds), labels, out_cnt)
    print("ZERO:", auc_, tpk)

    return idx



def print_performance(preds, labels, member_list, out_cnt):
    """
    Takes the predictions from the ensemble and prints the ROC AUC,
    PR AUC and top-k ratio score
    """


    for i in range(len(member_list)):
        print("*****", member_list[i], "*****")
        ROC_AUC = roc_auc_score(labels, preds[i])
        p, r, _ = precision_recall_curve(labels, np.negative(preds[i]),
        pos_label=-1)
        PR_AUC = auc(r, p)
        tp_k = top_k(np.negative(preds[i]), labels, out_cnt)

        print("ROC AUC score:", round(ROC_AUC, 3))
        print("PR AUC score", round(PR_AUC, 3))
        print("TOP K ratio", round(tp_k, 3))
        print()

    print("****** MM-AVG ensemble ********")
    ROC_AUC = roc_auc_score(labels, comb_by_avg(min_max(preds)))
    p, r, _ = precision_recall_curve(labels,
    comb_by_avg(min_max(np.negative(preds))), pos_label=-1)
    PR_AUC = auc(r, p)
    tp_k = top_k(comb_by_avg(min_max(np.negative(preds))), labels, out_cnt)
    print("ROC AUC score:", round(ROC_AUC, 3))
    print("PR AUC score", round(PR_AUC, 3))
    print("TOP K ratio", round(tp_k, 3))
    print()


def run_ensemble(data_name, out_cnt, df_bin, df_str):
    """
    Takes the data set as dataframes in binary and string format.
    Then it runs the ensemble.
    """

    #split labels from data and make it np arrays
    data = np.array(df_bin.drop("label", axis=1))
    data_str = np.array(df_str.drop("label", axis=1))
    labels = np.array(df_bin['label'])

    print("**** Analyzing", data_name, "with shape:", data.shape, " *****")


    # prep ensemble w. parameters
    parameters = choose_params(data_name)
    estimators = {'ocsvm': OneClassSVM, 'zero': Zero, 'iForest': IsolationForest,
     'LOF': LocalOutlierFactor, 'fpof':FPOFSampler}
    member_list = ['LOF', 'fpof', 'iForest', 'zero', 'ocsvm']
    ensemble = init_members(member_list, parameters)


    #fit and predict
    ensemble, fpof_patterns = fit_ensemble(member_list, ensemble, data, data_str)
    preds = predict_ensemble(member_list, ensemble, data, data_str, fpof_patterns)

    #print out the performance metrics
    print_performance(preds, labels, member_list, out_cnt)


    #rfe(df_bin, labels, out_cnt)







if __name__ == '__main__':

    path1 = sys.argv[1]
    path2 = sys.argv[2]
    frac = 1

    #if running a sample of data only
    if len(sys.argv) > 3:
        frac = int(sys.argv[3]) / 100

    data_name = path1.split("/")[-1].split(".")[0]

    seed = 2019
    df_bin = pd.read_csv(path1).sample(frac=frac, random_state=seed)
    df_str = pd.read_csv(path2).sample(frac=frac, random_state=seed)

    #shuffling the data frames
    idx = np.random.permutation(df_bin.index)
    df_bin = df_bin.reindex(idx)
    df_str = df_str.reindex(idx)

    #number of outliers in the data set
    out_cnt = df_bin.loc[df_bin['label'] == -1].shape[0]

    run_ensemble(data_name, out_cnt, df_bin, df_str)
