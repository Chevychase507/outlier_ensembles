import numpy as np
import pandas as pd
import sys


"""
Takes the Solar Flare data set as flare1.csv and flare2.csv. It
transforms them to a one hot encoded data set with. X as the outlier class
and all other classes as the normal class
"""

def remove_classes(df):
    """
    Takes a data frame and makes the K class the outlier class and the rest of
    the classes the inlier class.
    """

    df.columns = ['F1','label', 'F2', 'F3', 'F4','F5','F6','F7','F8','F9','F10','F11','F12']
    df_label = df['label']
    df = df.drop('label', axis=1)
    df = pd.concat([df_label, df], axis=1)

    un = df['label'].unique()
    for u in un:
        print(u, df.loc[df['label'] == u].shape)
    df['label'].replace({"K": -1, "A": 1, "R": 1, "S": 1, "X": 1, "H": 1}, inplace=True) #modify if necessary

    return df


def make_numerate(df):
    """
    Takes a data frame and makes the values numerate. Returns a numerate data
    frame.
    """

    cols = df.columns

    for col in cols:
        if col == "label":
            continue
        else:
            orig_vals = df[col].unique()
            replacement_dict = {}
            for i in range(len(orig_vals)):
                if orig_vals[i] == "?":
                    replacement_dict[orig_vals[i]] = -99
                else:
                    replacement_dict[orig_vals[i]] = i #give it its index as value
            df[col].replace(replacement_dict, inplace=True)

    return df


def one_hot(df):

    """
    Takes a numerical dataframe. It one_hot_encodes all the variables that
    have more than two possible values. It returns a one hot encoded dataframe.
    """

    cols = list(df.columns)

    hot_df = df['label']
    for col in cols:
        if len(df[col].unique()) <= 2: #if column is already binary
            if col == "label":
                continue
            hot_df = pd.concat([hot_df, df[col]], axis=1)
        else:
            hot_col = pd.get_dummies(df[col], prefix=col)
            hot_df = pd.concat([hot_df, hot_col], axis=1)

    return hot_df

if __name__ == '__main__':
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    df1 = pd.read_csv(path1, sep= " ", header=None)
    df2 = pd.read_csv(path2, sep= " ", header=None)
    df = pd.concat([df1, df2], axis=0)
    df_class = remove_classes(df)
    num_df = make_numerate(df_class)
    print("shape of df", num_df.shape)
    df_hot = one_hot(num_df)
    print("shape of df hot", df_hot.shape)
    print("# outliers", df_hot.loc[df_hot['label'] == -1].shape[0])

    df_hot.to_csv("solar_prep.csv", index=False)
