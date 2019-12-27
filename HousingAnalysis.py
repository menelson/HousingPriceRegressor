import os

# Imports for fetching the (tgz) data 
import tarfile
from six.moves import urllib

# Import Pandas to return a DataFrame object
import pandas as pd

# Import MatPlotLib for plotting
import matplotlib.pyplot as plt

# Use NumPy for e.g. constructing a test set manually
import numpy as np

# SciKit-Learn is a vital library for ML
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
# For training
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
# For K-fold cross-validation
from sklearn.model_selection import cross_val_score

# Hack untul sklearn is updated in the master branch
from categorical_encoder import CategoricalEncoder

# Global paths to the housing data
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Function to fetch the data from a tgz 
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Function to load the data with Pandas
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# Function to split a DataFrame into separate training and testing sets.
# However, best not to use this function as it has no random state, so we
# will get a different testing and training set everytime the function is called!
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data)) # Shuffle the data first
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size] # Sample upto the testing data size
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices] # Return the located indices of the training and testing sets

# A class for a DataFrameSelector object to be used in a custom transformer
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# A class for our custom transformer 
fetch_housing_data()
housing_df = load_housing_data()
    
rooms_ix, bedrooms_ix, population_ix, household_ix = [
list(housing_df.columns).index(col)
for col in ("total_rooms", "total_bedrooms", "population", "households")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def main():
    debug = False # Flag for debugging 
    unblind = False # Flag for keeping test data hidden

    fetch_housing_data()
    
    # Get the DataFrame
    housing_df = load_housing_data()
    
    # Investigate the data
    if debug:
        print(housing_df.head())
        print(housing_df.info())
        print(housing_df.describe())

    # Plot the data
    # Dump histograms of the DataFrame features, including the label (median_house_value)
    housing_df.hist(bins=50, figsize=(20,15))
    if debug: plt.show()
	
    # Get the training and testing sets using SciKit-Learn with a random state (not using stratified sampling here however; 
    # might therefore be biased!)
    train_set, test_set = train_test_split(housing_df, test_size=0.2, random_state=42)

    # Create an income category attribute and sort things such that we don't have
    # too many strata split over the DataFrame
    housing_df["income_cat"] = np.ceil(housing_df["median_income"]/1.5) # Round with ceil
    housing_df["income_cat"].where(housing_df["income_cat"] < 5, 5.0, inplace=True)
    housing_df["income_cat"].hist()
    if debug: plt.show()

    # Now let's create a stratified sample based on the income category.
    # This is just a test to make sure that the distribution of the income category
    # in the testing data set is similar to the trainign data, reducing bias.
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    strat_train_set = None
    strat_test_set = None
    for train_index, test_index in split.split(housing_df, housing_df["income_cat"]):
        strat_train_set = housing_df.loc[train_index]
        strat_test_set = housing_df.loc[test_index]

    # Remove income_cat so that the data is back to its original state
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
   
    # Explore the data 
    housing_df = strat_train_set.copy()
    housing_df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=housing_df["population"]/100, label="population", figsize=(10,7),
        c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
        sharex=False)
    plt.legend()
    if debug: plt.show()

    # Get the Pearson (linear) correlation coefficient
    corr_matrix = housing_df.corr()
    if debug: print(corr_matrix)

    # Let's create a few more interesting attributes from the existing features
    housing_df["rooms_per_household"] = housing_df["total_rooms"]/housing_df["households"]
    housing_df["bedrooms_per_room"] = housing_df["total_bedrooms"]/housing_df["households"]
    housing_df["population_per_household"] = housing_df["population"]/housing_df["households"]

    # Separate the predictors from the labels
    housing_df = strat_train_set.drop("median_house_value", axis=1) # Drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    # Drop districts with incomplete data
    housing_df.dropna(subset=["total_bedrooms"])
    imputer = SimpleImputer(strategy="median")
    housing_df_num = housing_df.drop("ocean_proximity", axis=1)
    imputer.fit(housing_df_num)
    X = imputer.transform(housing_df_num)

    # Use Panda's factorize() method to convert text to numbers
    housing_df_cat = housing_df["ocean_proximity"]
    housing_df_cat_encoded, housing_df_categories = housing_df_cat.factorize()
    if debug: 
        print(housing_df_cat_encoded[:10])
        print(housing_df_categories)
    
    # Let's create one binary attribute per category, using
    # one-hot encoding. We'll use the fit_transform() method,
    # however this expects a 2D array so we first need to reshape
    # the 1D array of housing_df_cat_encoded.
    encoder = OneHotEncoder()
    housing_df_cat_1hot = encoder.fit_transform(housing_df_cat_encoded.reshape(-1,1))
    
    # NOTE: above we have gone from text, to integer, to one-hot vector format. 
    # One can also use the CategoricalEncoder from sklearn to do this in a 
    # single bound. 
   
    # It is now important to apply a feature scaling such that the various
    # features are normaized/standardized to have values typically in the 
    # range [0,1].
    num_attribs = list(housing_df_num)
    cat_attribs = ["ocean_proximity"]
    num_pipeline = Pipeline([
  			     ('selector', DataFrameSelector(num_attribs)),
   		 	     ('imputer', SimpleImputer(strategy="median")),
		             ('attribs_adder', CombinedAttributesAdder()),
			     ('std_scaler', StandardScaler())
                           ])
    
    cat_pipeline = Pipeline([
  			     ('selector', DataFrameSelector(cat_attribs)),
   		 	     ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
                           ])
    # Combine these pipelines using a FeatureUnion
    full_pipeline = FeatureUnion(transformer_list=[
						   ("num_pipeline", num_pipeline),
						   ("cat_pipeline", cat_pipeline),
                                                  ])
    # Now run the whole pipeline simply:
    housing_prepared = full_pipeline.fit_transform(housing_df)
    if debug: print("Training data processed: ", housing_prepared)

    # It is finally time to train on the data; let's try with a decision tree regressor
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)
     
    # Use K-fold cross-validation to check the performance of the
    # validation set (a subset of the training set).
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    print("Score mean: ", scores.mean())
    print("Score standard deviation: ", scores.std())
    
    # NOTE: can later use a grid search to fine-tune the hyperparameters of
    # a particular model, and hence improve performance. It was found that the
    # RandomForest regressor offers partcularly good performance, especially after
    # fine-tuning with a grid search.

    # Evaluate on the test set
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)

    final_predictions = tree_reg.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    if unblind: print(final_mse)
if __name__ == "__main__":
    main()
