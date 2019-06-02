# Ensembles for categorical outlier detection
outlier_ensembles/ contains the code for my thesis project. Specifically,
it contains:

- master.py, the master program which runs the ensemble members.
- implementations/
- data/
- preprocessors/
- visualization_code/
- requirements.txt


## Requirements
The implementation is done in Python 3. I use additional libraries such as
pandas and numpy. This can be installed by utilizing the requirements.txt-file
and the following command:

pip3 install -r requirements.txt


## Run the program
The program can be run by the following command:

python3 master.py "binary_data" "string_data" "optional: percentage"

It will output the ROC AUC, PR AUC and the top-k ratio of the methods
when applied to the data set given.

## The Data
The preprocessed data sets can be found in data/preprocessed. The binary
data sets are in data/preprocessed/binary_data, and the string data sets are in
data/preprocessed/string_data. The original data sets can be found in
data/original. Only the UCI ML data sets are present here.

##
