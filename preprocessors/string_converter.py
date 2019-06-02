import numpy as np
import pandas as pd
import itertools
import sys
import string


"""
Takes a one hot encoded csv file and transforms it to a csv file where each
value is a unique string of lower case letters. The transformed data set can be
used for set-based methods such as FPOF or CompreX.
"""


def gen_distinct_combs(df):
    """
    Generates the necessary number of combinations of the required length, given
    the number of dimensions in the data set
    """
    letters = list(string.ascii_lowercase)

    cols = list(df.columns)
    n_unique = (len(cols) - 1) * 2

    n_combinations = len(letters)
    n_letters = 1
    #find needed length of unique string
    while n_combinations < n_unique:
        n_combinations = n_combinations * len(letters)
        n_letters += 1

    letter_list = []
    for i in range(n_letters):
        letter_list.append(letters)

    combinations = []
    for comb in itertools.product(*letter_list):
        combinations.append(comb[0] + comb[1])

    return combinations



def replace_values(combinations, df):
    """
    Takes the combinations list and replaces the binary values with
    string values. Returns a df with text strings. 
    """
    cols = list(df.columns)

    cnt = 0
    for i in range(len(cols)):
        if cols[i] == "label":
            continue
        df[cols[i]].replace({0: combinations[cnt], 1: combinations[cnt+1]}, inplace=True)
        cnt += 2

    return df


if __name__ == '__main__':
    path = sys.argv[1]
    df = pd.read_csv(path)
    combinations = gen_distinct_combs(df)
    df_distinct = replace_values(combinations, df)

    name = "distinct_" + path.split("/")[-1]
    new_path = "/".join(path.split("/")[:-1]) + "/" +  name

    df_distinct.to_csv(new_path, index=False)
