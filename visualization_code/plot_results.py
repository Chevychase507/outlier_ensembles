import numpy as np
import pandas as pd
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve
sys.path.insert(0, '/home/ahoj/Documents/ITU/4._semester/thesis/analysis/implementations/')
from utils import *


def bxplt_auc(df, auc):
    algs = list(df.loc[df['ensemble'] == 0]['algoritm'].unique())
    #algs = list(df.loc[df['ensemble'] == 1]['algoritm'].unique()) + ['ocsvm']


    data = [None] * len(algs)

    for i in range(len(algs)):
        print(i, algs[i])
        data[i] = np.array(df[auc].loc[df['algoritm'] == algs[i]])

    x_coor = np.arange(len(algs))

    data = np.array(data).T
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_axisbelow(True)
    ax.boxplot(data, sym='+', labels=algs)
    #plt.xticks(x_coor, algs, fontsize=8)
    #plt.yticks(np.arange(0,1.1,step=0.1))
    plt.show()

def bar_auc(file_paths, auc):

    color = "bgrycmw"
    ax = plt.subplot(111)
    ax.grid()
    ax.set_axisbelow(True)
    x_coor = None
    for i in range(len(file_paths)):
        df = pd.read_csv(file_paths[i])

        algs = list(df.loc[df['ensemble'] == 0]['algoritm'].unique()) #+ ['ocsvm']
        #algs = list(df.loc[df['ensemble'] == 1]['algoritm'].unique()) + ['ocsvm']

        #algs = list(df['algoritm'].unique())

        algs_dict = {}
        algs_data = [None] * len(algs)
        for j in range(len(algs)):
            sub_df = df.loc[df['algoritm'] == algs[j]]
            algs_dict[algs[j]] =  np.array(sub_df[auc])
            algs_data[j] = np.array(sub_df[auc])

        w = 0.1
        x = np.array(algs_data)

        x_mean = np.mean(x, axis=1)
        x_std = np.std(x, axis=1)

        if x_coor is None:
            x_coor = np.arange(len(algs))
        else:
            x_coor = np.add(x_coor, w)

        ax.bar(x_coor,height=x_mean, yerr=x_std, edgecolor="k", color=color[i], width=w, label=df['data_set'][1], align='center')
        #ax.title(auc +" of methods on " + df['data_set'][1])
        #plt.title("AUROC" + " of methods on Nursery data set")
    plt.ylabel("Avg. "+ auc + " scores w. std. dev")
    plt.xlabel("Algorithm")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08),
          ncol=3, fancybox=True, shadow=True)
    plt.xticks(x_coor, algs, fontsize=8)
    plt.yticks(np.arange(0,1.1,step=0.1))

    plt.show()

def bivar_plot(path):
    df = pd.read_csv(path)
    fig = plt.figure(1)
    cols = list(df.columns)
    members = cols[:-1]
    members = members[1:]
    color = "rg"
    columns = 4
    rows = 3

    plot_tuples = []
    for i in range(len(members)):
        for j in range(i+1, len(members)):
            plot_tuples.append((members[i], members[j]))

    #plt.title("Bivariate scatter plots of anomaly scores from " + df["data_name"][0])
    for i in range(1, columns*rows+1):
        if i < len(plot_tuples):
            fig.add_subplot(rows, columns, i)
            plt.scatter(df[plot_tuples[i][0]], df[plot_tuples[i][1]], c=df['label'])
            plt.xlabel(plot_tuples[i][0] + " score")
            plt.ylabel(plot_tuples[i][1] + " score")
        else:
            break

    plt.tight_layout()
    plt.show()


def plot_3d(path, members):
    df = pd.read_csv(path)

    df_in = df.loc[df['label'] == 1]
    df_out = df.loc[df['label'] == -1]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.title("3D plot of results from nurse")


    ax.scatter(df_in[members[0]], df_in[members[1]], df_in[members[2]], c="g", label="inliers")
    ax.scatter(df_out[members[0]], df_out[members[1]], df_out[members[2]], c="r", label="outliers")

    ax.set_xlabel(members[0] + " score")
    ax.set_ylabel(members[1] + " score")
    ax.set_zlabel(members[2] + " score")

    ax.legend(loc="best")

    plt.show()



def auc_pr_curves(labels, preds):
    preds = [preds]
    for i in range(len(preds)):
        pred = preds[i]
        tpr,fpr, _= roc_curve(labels, pred, pos_label=-1)
        #tpr,fpr, _= roc_curve(labels, pred)

        p, r, _ = precision_recall_curve(labels,pred, pos_label=-1)
        #p, r, _ = precision_recall_curve(labels,pred)


        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('PR Curve')
        ax1.text(0.3, 0.5, 'PR AUC =' + str(round(auc(r, p), 2)))
        ax1.set_xlim([-0.05,1.05])
        ax1.set_ylim([-0.05,1.05])

        ax2 = fig.add_subplot(1,2,2)
        ax2.set_xlabel('FPR')
        ax2.set_ylabel('TPR')
        ax2.set_title('ROC Curve')
        ax2.text(0.55, 0.5, 'ROC AUC =' + str(round(auc(tpr, fpr), 2)))
        ax2.set_xlim([-0.05,1.05])
        ax2.set_ylim([-0.05,1.05])

        ax1.plot(r,p, color="r")
        ax2.plot(tpr,fpr, color="b")

        #ax1.grid(True)
        #ax2.grid(True)
        plt.show()



def anomaly_hist(path):
    df = pd.read_csv(path)


    members = ['LOF', 'fpof', 'iForest', 'ocsvm', 'zero']
    members = ['ocsvm']
    members = ['fpof']

    data_name = "Nursery Data"


    for i in range(len(members)):
        mean_ = np.mean(df[members[i]])
        std_pos = mean_ + np.std(df[members[i]]) * 1
        std_neg = mean_ - np.std(df[members[i]]) * 1
        n, bins, patches = plt.hist(df[members[i]], bins='auto', facecolor='b')
        #title = "Anomaly scores of " + members[i] + " on " + data_name
        title = "FPOF: Anomaly scores from " + data_name
        plt.title(title)
        plt.axvline(mean_, color='r', linestyle='solid', linewidth=1)
        plt.axvline(std_neg, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(std_pos, color='k', linestyle='dashed', linewidth=1)
        plt.xlabel('Anomaly score')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()



def print_topk(df):
    algs = list(df.loc[df['ensemble'] == 1]['algoritm'].unique())
    data = sorted(list(df['data_set'].unique()), reverse=True)

    print(df)
    for i in range(len(algs)):
        tot = []
        for j in range(len(data)):
            print("**********", algs[i], ":::", data[j], "**********")
            topk = df.loc[(df['algoritm'] == algs[i]) & (df['data_set'] == data[j])]['top_k'].values
            tot.append(topk)
            print("topk:", round(topk[0], 3))
            print()
            print()
        print("OVERALL MEAN of", algs[i],":", round(np.mean(np.array(tot)), 3), " :::::: std:", round(np.std(tot), 3))





def print_auc(df, auc):

    algs = list(df.loc[df['ensemble'] == 0]['algoritm'].unique())
    data = sorted(list(df['data_set'].unique()), reverse=True)

    for i in range(len(algs)):
        tot = []
        for j in range(len(data)):
            print("**********", algs[i], ":::", data[j], "**********")
            auprc = df.loc[(df['algoritm'] == algs[i]) & (df['data_set'] == data[j])][auc].values
            tot.append(np.mean(auprc))
            print("mean:", round(np.mean(auprc),3))
            print("std:", round(np.std(auprc),3))
            print()
        print("OVERALL MEAN of", algs[i],":", round(np.mean(np.array(tot)), 3), " :::::: std:", round(np.std(tot), 3))



def penalize(vals, med):
    for i in range(len(vals)):
        if vals[i] < med:
            vals[i] = vals[i] - (med - vals[i])
            #vals[i] = vals[i] - (np.exp(med - vals[i]) * .5)


    return vals

def calc_robust(df_auc, df_tpk, auc):
    ind_auc = df_auc.loc[df_auc['ensemble'] == 0]
    ind_tpk = df_tpk.loc[df_tpk['ensemble'] == 0]
    algs = list(df_auc['algoritm'].unique())
    data = sorted(list(df_auc['data_set'].unique()), reverse=True)



    robust = [None] * len(algs)
    med_auc = {}
    med_tpk = {}
    for i in range(len(data)):
        med_auc[data[i]] = np.median(ind_auc.loc[ind_auc['data_set'] == data[i]][auc])
        med_tpk[data[i]] = np.median(ind_tpk.loc[df_tpk['data_set'] == data[i]]['top_k'])


    for i in range(len(algs)):
        scores = np.array([])
        pr_aucs = np.array([])
        tp_ks = np.array([])
        for j in range(len(data)):

            tmp_aucs = df_auc.loc[(df_auc['algoritm'] == algs[i]) & (df_auc['data_set'] == data[j])][auc].values
            tmp_ks = df_tpk.loc[(df_tpk['algoritm'] == algs[i]) & (df_tpk['data_set'] == data[j])]['top_k'].values

            pen_auc = np.array([np.mean(penalize(tmp_aucs, med_auc[data[j]]))])
            pen_tpk = np.array([np.mean(penalize(tmp_ks, med_auc[data[j]]))])

            #tmp_aucs[tmp_aucs < med_auc[data[j]]] = 0
            #tmp_ks[tmp_ks < med_tpk[data[j]]] = 0
            tmp = np.concatenate([pen_auc, pen_tpk], axis=0)
            scores = np.concatenate([scores, tmp], axis=0)
            pr_aucs = np.concatenate((pr_aucs, pen_auc), axis=0)
            tp_ks = np.concatenate((tp_ks, pen_tpk), axis=0)


        print(algs[i], round(np.mean(pr_aucs) + np.mean(tp_ks), 3), round(np.mean(scores), 3))

    return robust, algs

def plot_robust(df_auc, df_tpk, auc):
    robust, algs = calc_robust(df_auc, df_tpk, auc)
    x_coor = np.arange(len(algs))
    plt.grid(True)
    plt.bar(algs, height=robust, label=algs, color="rbgyc")

    #plt.set_axisbelow(True)
    plt.show()

def plot_feat_select(df1, df2):
    #print(df.columns)


    ax1 = plt.subplot(111)
    ax2 = plt.subplot(111)
    ax3 = plt.subplot(111)

    baseline = np.full(len(df1['top_k_mean']), 0.213)
    std = np.full(len(df1['top_k_mean']), 0)
    ax1.errorbar(df1['n_feats'], df1['top_k_mean'], df1['top_k_std'], label="OCSVM", color="b")
    ax2.errorbar(df2['n_feats'], df2['top_k_mean'], df2['top_k_std'], label="ZERO++", color="y")
    ax3.errorbar(df1['n_feats'], baseline, std, label="Baseline", color="r")

    plt.legend(loc="best")
    plt.xlabel('N features')
    plt.ylabel('Top-k ratio')

    plt.show()


def violin(df):

    print(df.loc[df['Score'] > 1])

    ax = sns.violinplot(x="v", y="Score", hue="Label", data=df, scale="width", palette="muted", split=False)
    x_ticks = [x/10 for x in range(1,10)]
    #ax.set_xticklabels(['RBF','Linear','Polynomial','Sigmoid'])
    ax.set_xticklabels(x_ticks)
    plt.xlabel("v-parameter setting")
    #swarm_plot = sns.swarmplot(x=df['label'], y=df['pc1'])
    #print("done making the plot"
    #plt.yticks([x/10 for x in range(0,11)])
    plt.show()

def bar_2(df, measure, data_sets=None):
    data_sets = df['data_set'].unique()
    data_sets = sorted(data_sets)
    print(df.columns)
    print(df)
    data_sets = ['nurse_prep_diff', 'spect_heart_prep', 'solar_prep', 'mushroom_prep_10', 'chess_prep_diff', 'clean_fup_hot']
    #if data_sets is None:
    #    data_sets = ['nurse_prep_diff', 'spect_heart_prep', 'solar_prep', 'mushroom_prep_10', 'chess_prep_diff']

    algs = ['ocsvm', 'E_minmax_avg']
    alg_names = ['OCSVM', 'MM-AVG']
    ax = plt.subplot(111)
    ax.grid()
    ax.set_axisbelow(True)
    x_coor = None
    color = "br"
    data_titles = [x for x in range(20,32)]
    data_titles = ["Nursery", "Spect Heart", "Solar Flare", "Mushroom", "Chess", "Webshops"]
    print(data_titles)

    ds_data = [None] * len(data_sets)
    for i in range(len(algs)):
        sub_df = df.loc[df['algoritm'] == algs[i]]
        for j in range(len(data_sets)):
            ds_data[j] = np.array(sub_df.loc[sub_df['data_set'] == data_sets[j]][measure])

        w = 0.1
        if x_coor is None:
            x_coor = np.arange(len(data_sets))
        else:
            x_coor = np.add(x_coor, w)

        x = np.array(ds_data)

        x_mean = np.mean(x, axis=1)
        print(algs[i], x_mean)
        x_std = np.std(x, axis=1)
        ax.bar(x_coor, height=x_mean, yerr=x_std, edgecolor="k", color=color[i], width=w, label=alg_names[i], align='center')



    plt.ylabel("Top-k Ratios")
    #plt.ylabel("PR AUC scores")
    plt.xlabel("N features")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08),
          ncol=3, fancybox=True, shadow=True)
    plt.xticks(x_coor, data_titles, fontsize=8)
    plt.show()








if __name__ == '__main__':

    two = False
    if len(sys.argv) == 2:
        path1 = sys.argv[1]
    else:
        path1 = sys.argv[1]
        path2 = sys.argv[2]

        two = True


    if ".csv" in path1:


        df = pd.read_csv(path1)
        #print(df)
        preds = np.array(df['Score'].loc[df['Algorithm'] == "OCSVM"])
        labels = np.array(df['Label'].loc[df['Algorithm'] == "OCSVM"])
        #print(preds)
        #preds = np.full(100,0)
        #labels = np.random.randint(2, size=100)


        auc_pr_curves(labels, preds)
        """
        #violin(df1)
        #df2 = pd.read_csv(path2)

        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)
        plot_feat_select(df1, df2)
        """




    else:
        files1 = listdir(path1)
        li = []


        for i in range(len(files1)):
            files1[i] = path1 + files1[i]
            df_tmp = pd.read_csv(files1[i])
            li.append(df_tmp)

        df_auc = pd.concat(li, axis=0, ignore_index=True)

        if two:
            files2 = listdir(path2)
            li = []

            for i in range(len(files2)):
                files2[i] = path2 + files2[i]
                df_tmp = pd.read_csv(files2[i])
                li.append(df_tmp)


        df_tpk = pd.concat(li, axis=0, ignore_index=True)


        #plot_robust(df_auc, df_tpk, "auprc")
        print_auc(df_auc, 'top_k')
        #print_topk(df_tpk)
        #calc_robust(df_auc, df_tpk, "auprc")
        #bar_2(df_auc, 'top_k')
