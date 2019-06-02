import numpy as np
import pandas as pd
import sys


"""
Takes the mushroom.csv file and replaces the enums with integers and
one hot encodes the mushroom data set.
"""


def make_numerate(df):
    """
    Takes a dataframe of the mushroom data set and transforms the enumerations to
    integers. It returns a dataframe with numerical values.
    """
    cols = list(df.columns)

    for col in cols:
        if col == "label":
            # give it 1 for edible and -1 for poisnous
            df[col].replace({"p": -1, "e":1}, inplace=True)
        else:
            orig_vals = df[col].unique()
            replacement_dict = {}
            for i in range(len(orig_vals)):
                if orig_vals[i] == "?":
                    replacement_dict[orig_vals[i]] = -99
                else:
                    #give it its index as value
                    replacement_dict[orig_vals[i]] = i
            df[col].replace(replacement_dict, inplace=True)

    return df

def one_hot_encode(df):
    """
    Takes a numerical dataframe from the mushroom data set. It one_hot_encodes
    all the variables that have more than two possible values. It returns a one
    hot encoded dataframe.
    """

    cols = list(df.columns)
    cols.remove('stalk-root') #remove since it contains 2400 missing values

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


def undersample_pois(df):
    """
    Takes a df and undersamples the poisonous class.
    """
    df_ed = df.loc[df['label'] == 1]
    df_po = df.loc[df['label'] == -1].sample(frac=0.1, random_state=2019)# changed it to 10 percent
    df_complete = pd.concat([df_ed, df_po], axis=0).sample(frac=1, random_state=2019)

    return df_complete



if __name__ == '__main__':

    path = sys.argv[1]
    df = pd.read_csv(path)
    df_numbers = make_numerate(df)
    print("shape of df", df_numbers.shape)
    hot_df = one_hot_encode(df_numbers)
    print("shape of hot df", hot_df.shape)
    undersample_df = undersample_pois(hot_df)
    print("shape of undersampled df", undersample_df.shape)
    print("# outliers", undersample_df.loc[undersample_df['label'] == -1].shape[0])
    undersample_df.to_csv("mushroom_prep.csv", index=False)
