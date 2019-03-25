# Lily Nguyen
# Wan Fong
# Andy Wong
# CMPS 140: Artificial Intelligence
# 24 March 2019

import argparse, csv, os, pandas, preprocess, sklearn, sys, time
import sklearn.metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

# default file path for training data files
TRAIN_DATA_PATH = "data/train.csv"
BREED_LABELS_PATH = "data/breed_labels.csv"
COLOR_LABELS_PATH = "data/color_labels.csv"
# default file path for sentiment analysis json files
SENTIMENT_DATA_PATH = "data/train_sentiment/"
# default file path for dog/cat training data, created in preprocessing stage
DOG_DATA_PATH = "data/train_dog.csv"
CAT_DATA_PATH = "data/train_cat.csv"
# file path to write performance evaluation for dog/cat model
DOG_PERF_PATH = "data/perf_dog.txt"
CAT_PERF_PATH = "data/perf_cat.txt"
# number of folds for cross-fold validation
CV_VAL = 10
# list of performance metrics to calculate using scikit's cross_val_score
PERF_METRICS = ["f1_micro", "recall_micro", "precision_micro", "accuracy"]

# declare classifiers
dog_NB_model = GaussianNB()
dog_SVM_model = LinearSVC(penalty = 'l1', dual = False)
dog_LR_model = LogisticRegression(C = 3.0, solver = 'lbfgs', multi_class = 'multinomial', random_state = 1, max_iter = 1000)
cat_NB_model = GaussianNB()
cat_SVM_model = LinearSVC(penalty = 'l1', dual = False)
cat_LR_model = LogisticRegression(C = 1.5, solver = 'lbfgs', multi_class = 'multinomial', random_state = 1, max_iter = 1000)


def train_classifiers(features, labels, isDog):
    """
        Creates the ensemble model and trains all of the classifiers, given the feature data and label data.
        Returns the created ensemble classifier.
        The isDog parameter is used to determine whether to build the dog classifier or the cat classifier.
    """
    # create ensemble model for appropriate animal type
    if isDog:
        dog_ensemble_model = VotingClassifier(estimators = [('NB', dog_NB_model), ('SVM', dog_SVM_model), ('LR', dog_LR_model)], voting = 'hard', weights = [1, 11, 11])
        dog_ensemble_model = dog_ensemble_model.fit(features, labels)
        return dog_ensemble_model
    else:
        cat_ensemble_model = VotingClassifier(estimators = [('NB', cat_NB_model), ('SVM', cat_SVM_model), ('LR', cat_LR_model)], voting = 'hard', weights = [1, 11, 11])
        cat_ensemble_model = cat_ensemble_model.fit(features, labels)
        return cat_ensemble_model 


def eval_performance(model, features, labels, file_path):
    """
        Evaluates the performance of the ensemble classifer (and all its individual classifiers) using cross-fold validation.
        The performance metrics evaluated are determined by the PERF_METRICS global list.
        Writes the results to a text file.
    """
    # add timestamp to file name
    time_str = time.strftime("_%Y%m%d%H%M%S")
    new_file_path = file_path[:-4] + time_str + file_path[-4:] # add timestamp before the ".txt"
    
    out_file = open(new_file_path, "w")
    out_file.write("+++ CROSS VALIDATION (CV = " + str(CV_VAL) + ") +++\n")

    # get list of individual classifiers
    classifier_list = model.estimators_
    classifier_list.append(model) # add ensemble classifier to end of list

    # output cv performance of individual classifiers + ensemble classifier
    for classifier in classifier_list:
        class_name = str(type(classifier))[16:-2]
        out_file.write("\nCLASSIFIER: " + class_name + "\n")
        for metric in PERF_METRICS:
            cv_score = cross_val_score(classifier, features, labels, cv = CV_VAL, scoring = metric)
            out_file.write("\tAverage " + metric + ": " + str(sum(cv_score) / len(cv_score)) + "\n")

    out_file.close()


def main():
    """
        Entry point for the entire program.
    """
    ### read in command-line arguments ###
    parser = argparse.ArgumentParser(description = "program to predict the adoption speed of a given pet")
    parser.add_argument("--train", dest = "train_file", default = TRAIN_DATA_PATH, type = str, help = "the path to the .csv file containing the training data")
    parser.add_argument("--p", dest = "preprocess_flag", action = "store_true", default = False, help = "whether the preprocessing stage should be run")
    args = parser.parse_args()

    ### preprocess the data & feature selection ###
    if args.preprocess_flag or DOG_DATA_PATH[5:] not in os.listdir("data/") or CAT_DATA_PATH[5:] not in os.listdir("data/"):
        preprocess.preprocess_data(args.train_file, BREED_LABELS_PATH, COLOR_LABELS_PATH, SENTIMENT_DATA_PATH, DOG_DATA_PATH, CAT_DATA_PATH)
        print("preprocessing stage complete")

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

    ### get features ###
    dog_feat_df = dog_data_df.drop("AdoptionSpeed", axis = 1)
    dog_feat_df = dog_feat_df.drop("PetID", axis = 1)
    cat_feat_df = cat_data_df.drop("AdoptionSpeed", axis = 1)
    cat_feat_df = cat_feat_df.drop("PetID", axis = 1)

    ### training ###
    dog_model = train_classifiers(dog_feat_df, dog_data_df["AdoptionSpeed"], True)
    cat_model = train_classifiers(cat_feat_df, cat_data_df["AdoptionSpeed"], False)
    print("model training stage complete")

    ### testing ###
    # get predictions
    # don't need to get that here if usuing scikit cross validator

    ### evaluation ###
    # perform cross-validation
    eval_performance(dog_model, dog_feat_df, dog_data_df["AdoptionSpeed"].tolist(), DOG_PERF_PATH)
    print("model evaluation for dog data complete")
    eval_performance(cat_model, cat_feat_df, cat_data_df["AdoptionSpeed"].tolist(), CAT_PERF_PATH)
    print("model evaluation for cat data complete")


if __name__ == '__main__':
    main()