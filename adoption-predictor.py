# Lily Nguyen
# Wan Fong
# Andy Wong
# CMPS 140: Artificial Intelligence
# 28 January, 2019

import argparse, csv, pandas, sys

# default file path for training data
TRAIN_DATA_PATH = "data/train.csv"

def main():
    ### read in command-line arguments ###
    parser = argparse.ArgumentParser(description = "program to predict the adoption speed of a given pet")
    parser.add_argument("--train", dest = "train_file", default = TRAIN_DATA_PATH, type = str, help = "the path to the .csv file containing the training data")
    args = parser.parse_args()

    ### read training data into a pandas dataframe ###
    try:
        #with open(args.train_file, mode = "r", encoding = "latin2") as csv_train:
        #    csv_reader = csv.DictReader(csv_train)
        #    for row in csv_reader:
        #        print(row)
        train_data_df = pandas.read_csv(args.train_file)
    except FileNotFoundError:
        print("ERROR: Training data file does not exist. Fiile must be of type .csv")
        sys.exit(1)
    except:
        print("ERROR: Unknown error occurred trying to read training data file")
        sys.exit(1)

    print(train_data_df.head(10))

    ### preprocessing and feature extraction ###

    ### training ###

    ### testing ###


if __name__ == '__main__':
    main()