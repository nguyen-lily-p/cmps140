# Lily Nguyen
# Wan Fong
# Andy Wong
# CMPS 140: Artificial Intelligence
# 28 January, 2019

import argparse, csv, json, os, pandas, sys

# default file path for training data
TRAIN_DATA_PATH = "data/train.csv"
# default file path for sentiment analysis json files
SENTIMENT_DATA_PATH = "data/train_sentiment/"

def main():
    ### read in command-line arguments ###
    parser = argparse.ArgumentParser(description = "program to predict the adoption speed of a given pet")
    parser.add_argument("--train", dest = "train_file", default = TRAIN_DATA_PATH, type = str, help = "the path to the .csv file containing the training data")
    args = parser.parse_args()

    ### read training data into a pandas dataframe ###
    try:
        train_data_df = pandas.read_csv(args.train_file, encoding = "utf8")
    except FileNotFoundError:
        print("ERROR: Training data file does not exist. Fiile must be of type .csv")
        sys.exit(1)
    except:
        print("ERROR: Unknown error occurred trying to read training data file")
        sys.exit(1)

    ### preprocessing and feature extraction ###

    # drop unwanted fields
    train_data_df.drop("Description", axis = 1, inplace = True)
    train_data_df.drop("Name", axis = 1, inplace = True)
    train_data_df.drop("RescuerID", axis = 1, inplace = True)
    train_data_df.drop("State", axis = 1, inplace = True)

    # add the sentiment analysis columns to dataframe
    train_data_df["DescriptionSentiment"] = None
    for file in os.listdir(SENTIMENT_DATA_PATH):
        with open(SENTIMENT_DATA_PATH + file, "r", encoding = "utf8") as sentiment_file:
            sentiment_json = json.load(sentiment_file)
            sentiment = sentiment_json["documentSentiment"]["magnitude"] * sentiment_json["documentSentiment"]["score"]
            train_data_df.loc[train_data_df["PetID"] == file[:-5], "DescriptionSentiment"] = sentiment

    # drop rows that don't have a value for sentiment
    train_data_df = train_data_df[train_data_df.DescriptionSentiment.notnull()]

    train_data_df.to_csv("data/modified_train.csv", index = False)

    ### training ###

    ### testing ###


    # TODO
    # remove columns: state, rescuer ID
    # one-hot encoding for categorical features
    # split data between dogs and cats


if __name__ == '__main__':
    main()