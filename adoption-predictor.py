# Lily Nguyen
# Wan Fong
# Andy Wong
# CMPS 140: Artificial Intelligence
# 15 February 2019

import argparse, csv, json, math, os, pandas, preprocess, sys

# default file path for training data files
TRAIN_DATA_PATH = "data/train.csv"
BREED_LABELS_PATH = "data/breed_labels.csv"
COLOR_LABELS_PATH = "data/color_labels.csv"
# default file path for sentiment analysis json files
SENTIMENT_DATA_PATH = "data/train_sentiment/"
# default file path for dog/cat training data, created in preprocessing stage
DOG_DATA_PATH = "data/train_dog.csv"
CAT_DATA_PATH = "data/train_cat.csv"

def main():
    ### read in command-line arguments ###
    parser = argparse.ArgumentParser(description = "program to predict the adoption speed of a given pet")
    parser.add_argument("--train", dest = "train_file", default = TRAIN_DATA_PATH, type = str, help = "the path to the .csv file containing the training data")
    parser.add_argument("--p", dest = "preprocess_flag", action = "store_true", default = False, help = "whether the preprocessing stage should be run")
    args = parser.parse_args()

    ### preprocess the data & feature selection ###
    if args.preprocess_flag or DOG_DATA_PATH[5:] not in os.listdir("data/") or CAT_DATA_PATH[5:] not in os.listdir("data/"):
        preprocess.preprocess_data(args.train_file, BREED_LABELS_PATH, COLOR_LABELS_PATH, SENTIMENT_DATA_PATH, DOG_DATA_PATH, CAT_DATA_PATH)
        print("preprocess stage complete")

    ### read training data into a pandas dataframe ###
    try:
        dog_data_df = pandas.read_csv(DOG_DATA_PATH, encoding = "utf8")
        cat_data_df = pandas.read_csv(CAT_DATA_PATH, encoding = "utf8")
    except FileNotFoundError:
        print("ERROR: Dog/cat data file does not exist. File must be of type .csv")
        sys.exit(1)
    except:
        print("ERROR: Unknown error occurred trying to read dog/cat data file")
        sys.exit(1)

    ### training ###

    ### testing ###

if __name__ == '__main__':
    main()