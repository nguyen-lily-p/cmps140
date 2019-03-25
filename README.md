# PAL (Pet Adoption Likelihood)

### Lily Nguyen, Andy Wong, Wan Fong
### CMPS 140: Artificial Intelligence
### 24 March 2019

### ABOUT
This program builds a machine learning model that can classify a pet's adoption speed. This is based on the Kaggle competition, sponsored by PetFinder.my (<https://www.kaggle.com/c/petfinder-adoption-prediction>). This program uses the data provided on the Kaggle website.

### TO RUN THE PROGRAM
1. Ensure that all required files are present:
    * adoption-predictor.py
    * preprocess.py
    * a "data" folder containing:
        * train.csv
        * breed_labels.csv
        * color_labels.csv
        * a "train_sentiment" folder containing all of the JSON files

2. Run the command "python adoption-predictor.py"

3. This produces:
    * data/train_dog.csv: a preprocessed dataset containing only dogs
    * data/trian_cat.csv: a preprocessed dataset containing only cats
    * data/perf_dog_2019XXXX.txt: a text file containing the model evaluation for the dog training data
    * data/perf_cat_2019XXXX.txt: a text file containing the model evaluation for the cat training data
