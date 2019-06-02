import numpy as np
import pandas as pd
import sys


"""
Takes the spect_HEART.csv file and changes the label such that outliers are -1
and inliers are 1.
"""

def fix_label(df):
    """
    Fits the label column  such that outliers are -1 and inliers are 1.
    """
    df_label = df['OVERALL_DIAGNOSIS']

    df_label.replace({0: -1, 1: 1}, inplace=True)
    df = df.drop(['OVERALL_DIAGNOSIS'], axis=1)
    df = pd.concat([df_label, df], axis=1)
    df.columns.values[0] = "label"
    return df



if __name__ == '__main__':
    path = sys.argv[1]
    df = pd.read_csv(path)
    df = fix_label(df)
    print("shape of all", df.shape)
    print("# outliers", df.loc[df['label'] == -1].shape[0])
    df.to_csv("spect_heart_prep.csv", index=False)
