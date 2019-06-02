import numpy as np
import pandas as pd
import sys

"""
Takes the nurse.csv file and one hot encodes the data set. It returns only
the classes very_recom and spec_prior
"""

def remove_classes(df):
    """
    Takes the nusery dataframe and removes the irrelevant classes. Returns
    a dataframe with the relevant classes "spec_prior" and "very_recomm",
    inspired by the procedure in the Pang et al. (2016) paper:
    https://www.jair.org/index.php/jair/article/view/11035
    """

    labels = list(df['label'].unique())
    df_spec = df.loc[df['label'] == 'spec_prior']
    df_very = df.loc[df['label'] == 'very_recom']

    df_complete = pd.concat([df_spec, df_very], axis=0)

    return df_complete


def make_numerate(df):
    """
    Takes a df and makes the values numerical
    """

    cols = list(df.columns)

    for col in cols:
        if col == "label":
            df[col].replace({"very_recom": -1, "spec_prior": 1}, inplace=True) # give it 1 for edible and -1 for poisnous
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

def one_hot_encode(df):
    """
    Takes a numerical dataframe from the nursery data set. It one_hot_encodes all
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

    df_bin = remove_classes(df)
    df_num = make_numerate(df_bin)
    print("shape of df", df_num.shape)
    df_hot = one_hot_encode(df_num)
    print("shape of hot df", df_hot.shape)
    print("# outliers", df_hot.loc[df_hot['label'] == -1].shape[0])
    df_hot.to_csv("nurse_prep.csv", index=False)
