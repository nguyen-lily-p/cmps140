# Lily Nguyen
# Wan Fong
# Andy Wong
# CMPS 140: Artificial Intelligence
# 15 February 2019

import argparse, csv, json, math, os, pandas, sys

# default file path for training data files
TRAIN_DATA_PATH = "data/train.csv"
BREED_LABELS_PATH = "data/breed_labels.csv"
COLOR_LABELS_PATH = "data/color_labels.csv"
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
            # sentiment will be set as the overall document emotion score * the overall document magnitude score
            sentiment = sentiment_json["documentSentiment"]["magnitude"] * sentiment_json["documentSentiment"]["score"]
            train_data_df.loc[train_data_df["PetID"] == file[:-5], "DescriptionSentiment"] = sentiment

    # drop rows that don't have a value for sentiment
    train_data_df = train_data_df[train_data_df.DescriptionSentiment.notnull()]

    # replace categorical feature values w/ actual text values
    BREED_DICT = {0: None}
    COLOR_DICT = {0: None}
    GENDER_DICT = {1: "Male", 2: "Female", 3: "Mixed"}
    MATURITYSIZE_DICT = {0: "Not Specified", 1: "Small", 2: "Medium", 3: "Large", 4: "Extra Large"}
    FURLENGTH_DICT = {0: "Not Specified", 1: "Short", 2: "Medium", 3: "Long"}
    VACCINATED_DICT = {1: "Yes", 2: "No", 3: "Not Sure"}
    DEWORMED_DICT = {1: "Yes", 2: "No", 3: "Not Sure"}
    STERILIZED_DICT ={1: "Yes", 2: "No", 3: "Not Sure"}
    HEALTH_DICT = {0: "Not Specified", 1: "Healthy", 2: "Minor Injury", 3: "Serious Injury"}
    with open(BREED_LABELS_PATH, "r", encoding = "utf8") as breeds_file:
        csv_reader = csv.DictReader(breeds_file)
        for row in csv_reader:
            BREED_DICT[int(row["BreedID"])] = row["BreedName"]
    with open(COLOR_LABELS_PATH, "r", encoding = "utf8") as colors_file:
        csv_reader = csv.DictReader(colors_file)
        for row in csv_reader:
            COLOR_DICT[int(row["ColorID"])] = row["ColorName"]

    train_data_df.Breed1 = train_data_df.Breed1.replace(BREED_DICT)
    train_data_df.Breed2 = train_data_df.Breed2.replace(BREED_DICT)
    train_data_df.Color1 = train_data_df.Color1.replace(COLOR_DICT)
    train_data_df.Color2 = train_data_df.Color2.replace(COLOR_DICT)
    train_data_df.Color3 = train_data_df.Color3.replace(COLOR_DICT)
    train_data_df.Gender = train_data_df.Gender.replace(GENDER_DICT)
    train_data_df.MaturitySize = train_data_df.MaturitySize.replace(MATURITYSIZE_DICT)
    train_data_df.FurLength = train_data_df.FurLength.replace(FURLENGTH_DICT)
    train_data_df.Vaccinated = train_data_df.Vaccinated.replace(VACCINATED_DICT)
    train_data_df.Dewormed = train_data_df.Dewormed.replace(DEWORMED_DICT)
    train_data_df.Sterilized = train_data_df.Sterilized.replace(STERILIZED_DICT)
    train_data_df.Health = train_data_df.Health.replace(HEALTH_DICT)

    # combine features that have been split (Breed, Color)
    train_data_df["Breed"] = train_data_df[["Breed1", "Breed2"]].values.tolist()
    train_data_df["Breed"] = train_data_df["Breed"].apply(lambda row: [breed for breed in row if str(breed) != "nan"])
    train_data_df.drop("Breed1", axis = 1, inplace = True)
    train_data_df.drop("Breed2", axis = 1, inplace = True)
    train_data_df["Color"] = train_data_df[["Color1", "Color2", "Color3"]].values.tolist()
    train_data_df["Color"] = train_data_df["Color"].apply(lambda row: [color for color in row if str(color) != "nan"])
    train_data_df.drop("Color1", axis = 1, inplace = True)
    train_data_df.drop("Color2", axis = 1, inplace = True)
    train_data_df.drop("Color3", axis = 1, inplace = True)

    train_data_df.to_csv("data/train_modified.csv", index = False)

    # do one-hot encoding/dummies for each feature that needs it
    train_data_df = train_data_df.join(train_data_df["Breed"].str.get_dummies(sep = ","))
    train_data_df = train_data_df.join(train_data_df["Color"].str.get_dummies(sep = ","))
    train_data_df = pandas.get_dummies(train_data_df, prefix = ["Gender", "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health"], columns = ["Gender", "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health"])

    # split data between dogs/cats
    train_dog_df = train_data_df[train_data_df.Type == 1]
    train_cat_df = train_data_df[train_data_df.Type == 2]
    train_dog_df = train_dog_df.drop("Type", axis = 1)
    train_cat_df = train_cat_df.drop("Type", axis = 1)

    # write modified dataframes to new files
    train_dog_df.columns = train_dog_df.columns.str.strip().str.replace("'", "").str.replace("[", "").str.replace("]", "")
    train_cat_df.columns = train_cat_df.columns.str.strip().str.replace("'", "").str.replace("[", "").str.replace("]", "")    
    train_dog_df.to_csv("data/train_dog.csv", index = False)
    train_cat_df.to_csv("data/train_cat.csv", index = False)
    #train_data_df.to_csv("data/train_modified.csv", index = False)

    ### training ###

    ### testing ###


    # TODO
    # remove columns: state, rescuer ID
    # one-hot encoding for categorical features
    # split data between dogs and cats


if __name__ == '__main__':
    main()