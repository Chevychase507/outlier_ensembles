import numpy as np
import pandas as pd
import sys

"""
Takes the chess.csv file from the UCI repository. It one hot encodes the data set.
It returns only the largest class "fourteen" (label == 1) and the  "five"
class(label == -1) inspired by the procedure in the Pang et al. (2016) paper:
https://www.jair.org/index.php/jair/article/view/11035
"""


def remove_classes(df):
    """
    Takes the chess dataframe and removes the irrelevant classes. Returns
    a dataframe with the relevant classes "five" and "fourteen",
    """

    df_five = df.loc[df['target'] == 'five']
    df_fourteen = df.loc[df['target'] == 'fourteen']
    df_complete = pd.concat([df_fourteen, df_five], axis=0)

    return df_complete



def make_numerate(df):
    """
    Takes the chess data frame and makes feature 1, 3 and 5 numerate instead of
    enums. Returns the dataframe with numerical values.
    """
    cols = ['F1', 'F3', 'F5']
    for col in cols:
            orig_vals = df[col].unique()
            replacement_dict = {}
            for i in range(len(orig_vals)):
                if orig_vals[i] == "?":
                    replacement_dict[orig_vals[i]] = -99
                else:
                    replacement_dict[orig_vals[i]] = i #give it its index as value
            df[col].replace(replacement_dict, inplace=True)

    return df

def fix_label(df):

    """
    Changes the label column values and title such that outliers are -1 and
    inliers are 1.
    """

    label = df['target']
    label.replace({"five": -1,  "fourteen": 1}, inplace=True)
    df_complete = pd.concat([label, df], axis=1)
    df_complete.columns.values[0] = "label"

    df_complete.drop('target', axis=1, inplace=True)

    return df_complete

def one_hot_encode(df):
    """
    Takes a numerical dataframe. It one_hot_encodes all
    the variables that have more than two possible values.
    It returns a one hot encoded dataframe.
    """

    cols = list(df.columns)

    hot_df = df['label']
    for col in cols:
        if len(df[col].unique()) <= 2: #if column is already binary
            if col == "label":
                continue
            hot_df = pd.concat([hot_df, df[col]], axis=1)
        else:
            dummies = pd.get_dummies(df[col], prefix=col)
            hot_df = pd.concat([hot_df, dummies], axis=1)

    return hot_df


if __name__ == '__main__':

    path = sys.argv[1]
    df = pd.read_csv(path)
    df_binary = remove_classes(df)
    print("shape of df", df_binary.shape)
    df_num = make_numerate(df_binary)
    df_fixed = fix_label(df_num)
    df_hot = one_hot_encode(df_fixed)
    print("shape of one hot df", df_hot.shape)
    print("# outliers", df_hot.loc[df_hot['label'] == -1].shape[0])
    print(path)
    df_hot.to_csv("chess_prep.csv", index=False)
