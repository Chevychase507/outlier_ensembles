# Ensembles for categorical outlier detection
outlier_ensembles/ contains the code for my thesis project. Specifically,
it contains:

- master.py, the master program which runs the ensemble members.
- data/
- implementations/
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
when applied to the data set given. The "percentage" argument gives the
percentage of data which will be used when the program is run.

## data
The preprocessed data sets can be found in data/preprocessed. The binary
data sets are in data/preprocessed/binary_data, and the string data sets are in
data/preprocessed/string_data. The original data sets can be found in
data/original. Only the UCI ML data sets are present here.

## implementations
In implementations/, the implementations of Zero++ and FPOF with sampling
extension can be found. The directory also contains utils.py in which
many helper methods is defined.

## preprocessors
For each data set in data/original, there is a preprocessor which turns it into
a binary representation. It also contains string_converter.py that takes a
binarized data set and turns it into unique string variables

## visualization_code
Contains the code used for visualizations in the report. The code is not cleaned.
