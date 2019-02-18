# Lily Nguyen
# Wan Fong
# Andy Wong
# CMPS 140: Artificial Intelligence
# 15 February 2019

import csv, json, math, os, pandas, sys

# dictionaries for encoded categorical features
BREED_DICT = {0: None}
COLOR_DICT = {0: None}
GENDER_DICT = {1: "Male", 2: "Female", 3: "Mixed"}
MATURITYSIZE_DICT = {0: "Not Specified", 1: "Small", 2: "Medium", 3: "Large", 4: "Extra Large"}
FURLENGTH_DICT = {0: "Not Specified", 1: "Short", 2: "Medium", 3: "Long"}
VACCINATED_DICT = {1: "Yes", 2: "No", 3: "Not Sure"}
DEWORMED_DICT = {1: "Yes", 2: "No", 3: "Not Sure"}
STERILIZED_DICT ={1: "Yes", 2: "No", 3: "Not Sure"}
HEALTH_DICT = {0: "Not Specified", 1: "Healthy", 2: "Minor Injury", 3: "Serious Injury"}


def replace_categorical(train_data_df):
    """
        Replaces the encoded categorical features with their actual text values, using the pre-defined dictionaries
        Modified columns: Breed1, Breed2, Color1, Color2, Color3, Gender, MaturitySize, FurLength, Vaccinated, Dewormed, Sterilized, Health
    """
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

    return train_data_df


def preprocess_data(train_data_path, breed_labels_path, color_labels_path, sentiment_data_path, dog_data_path, cat_data_path):
    """
        Performs all necessary preprocessing and feature selection steps.
        Input:
            train_data_path: file path to the original training dataset
            breed_labels_path: file path to the label codes for the pet breeds
            color_labels_path: file path to the label codes for the colors
            sentiment_data_path: folder path where all sentiment analysis JSON files are kept
        Output:
            data/train_dog.csv: csv file containing the preprocessed data for dogs only 
            data/train_cat.csv: csv file contaiing the preprocessed data for cats only
    """
    ### read training data into a pandas dataframe ###
    try:
        train_data_df = pandas.read_csv(train_data_path, encoding = "utf8")
    except FileNotFoundError:
        print("ERROR: Training data file does not exist. File must be of type .csv")
        sys.exit(1)
    except:
        print("ERROR: Unknown error occurred trying to read training data file")
        sys.exit(1)

    ### add the sentiment analysis columns to dataframe ###
    train_data_df["DescriptionSentiment"] = None
    for file in os.listdir(sentiment_data_path):
        with open(sentiment_data_path + file, "r", encoding = "utf8") as sentiment_file:
            sentiment_json = json.load(sentiment_file)
            # sentiment will be set as the overall document emotion score * the overall document magnitude score
            sentiment = sentiment_json["documentSentiment"]["magnitude"] * sentiment_json["documentSentiment"]["score"]
            train_data_df.loc[train_data_df["PetID"] == file[:-5], "DescriptionSentiment"] = sentiment
    # drop rows that don't have a value for sentiment
    train_data_df = train_data_df[train_data_df.DescriptionSentiment.notnull()]

    ### create breed and color dictionaries ###
    with open(breed_labels_path, "r", encoding = "utf8") as breeds_file:
        csv_reader = csv.DictReader(breeds_file)
        for row in csv_reader:
            BREED_DICT[int(row["BreedID"])] = row["BreedName"]
    with open(color_labels_path, "r", encoding = "utf8") as colors_file:
        csv_reader = csv.DictReader(colors_file)
        for row in csv_reader:
            COLOR_DICT[int(row["ColorID"])] = row["ColorName"]

    ### replace categorical feature values w/ actual text values ###
    train_data_df = replace_categorical(train_data_df)

    ### combine features that have been split (Breed, Color) into one feature ###
    train_data_df["Breed"] = train_data_df[["Breed1", "Breed2"]].values.tolist()
    train_data_df["Color"] = train_data_df[["Color1", "Color2", "Color3"]].values.tolist()
    # remove NaN values from lists
    train_data_df["Breed"] = train_data_df["Breed"].apply(lambda row: [breed for breed in row if str(breed) != "nan"])
    train_data_df["Color"] = train_data_df["Color"].apply(lambda row: [color for color in row if str(color) != "nan"])
    # turn feature value from list into one string separated by |
    train_data_df["Breed"] = train_data_df["Breed"].apply(lambda row: "|".join(row))
    train_data_df["Color"] = train_data_df["Color"].apply(lambda row: "|".join(row))

    ### drop all unnecessary/unwanted columns ###
    train_data_df.drop("Description", axis = 1, inplace = True)
    train_data_df.drop("Name", axis = 1, inplace = True)
    train_data_df.drop("RescuerID", axis = 1, inplace = True)
    train_data_df.drop("State", axis = 1, inplace = True)
    train_data_df.drop("Breed1", axis = 1, inplace = True)
    train_data_df.drop("Breed2", axis = 1, inplace = True)
    train_data_df.drop("Color1", axis = 1, inplace = True)
    train_data_df.drop("Color2", axis = 1, inplace = True)
    train_data_df.drop("Color3", axis = 1, inplace = True)

    ### split data between dogs/cats ###
    dog_data_df = train_data_df[train_data_df.Type == 1]
    cat_data_df = train_data_df[train_data_df.Type == 2]
    # drop type column from both, which distinguishes between dog/cat
    dog_data_df = dog_data_df.drop("Type", axis = 1)
    cat_data_df = cat_data_df.drop("Type", axis = 1)

    ### do one-hot encoding/dummies for categorical features ###
    # must be done after split b/c dog breeds shouldn't appear as features in cat file, and vice versa
    # breed and color are done separately b/c instances can have multiple breeds/colors
    dog_data_df = dog_data_df.join(dog_data_df["Breed"].str.get_dummies(sep = "|"))
    dog_data_df = dog_data_df.join(dog_data_df["Color"].str.get_dummies(sep = "|"))
    dog_data_df = pandas.get_dummies(dog_data_df, prefix = ["Gender", "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health"], columns = ["Gender", "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health"])
    cat_data_df = cat_data_df.join(cat_data_df["Breed"].str.get_dummies(sep = "|"))
    cat_data_df = cat_data_df.join(cat_data_df["Color"].str.get_dummies(sep = "|"))
    cat_data_df = pandas.get_dummies(cat_data_df, prefix = ["Gender", "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health"], columns = ["Gender", "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health"])
    dog_data_df = dog_data_df.drop("Breed", axis = 1)
    dog_data_df = dog_data_df.drop("Color", axis = 1)
    cat_data_df = cat_data_df.drop("Breed", axis = 1)
    cat_data_df = cat_data_df.drop("Color", axis = 1)

    ### clean column headers ###
    dog_data_df.columns = dog_data_df.columns.str.strip().str.replace("'", "").str.replace("[", "").str.replace("]", "")
    cat_data_df.columns = cat_data_df.columns.str.strip().str.replace("'", "").str.replace("[", "").str.replace("]", "")

    ### write modified dataframes to new files ###    
    dog_data_df.to_csv(dog_data_path, index = False)
    cat_data_df.to_csv(cat_data_path, index = False)
