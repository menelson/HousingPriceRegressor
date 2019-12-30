import os
import pandas as pd
import numpy as np

# Helpers 
from HelperFunctions import *
from HelperClasses import *

#Sci-Kit learn imports for the ML implementation
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestRegressor # Training
from sklearn.model_selection import GridSearchCV # Grid search with K-fold cross-validation
from sklearn.metrics import mean_squared_error

# Hack until sklearn is updated in the master branch
from categorical_encoder import CategoricalEncoder

# For statistical tests on the final results
from scipy import stats

# Imports and global settings for nice plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def main(debug=False):
    fetch_housing_data()
    df = load_housing_data()
	
    df.hist(bins=50, figsize=(20,15))
    save_fig('housing_histograms')
    if debug:
		print(df.head(20))
		print(df.info())
		print(df.describe())

    
    # Create an income category attribute and sort things such that we don't have
    # too many strata split over the DataFrame. Use rounding to make these new categories
    df["income_cat"] = np.ceil(df["median_income"]/1.5) # Round with ceil
    df["income_cat"].where(df["income_cat"] < 5, 5.0, inplace=True)
    df["income_cat"].hist()
    save_fig('income_categories')
    
    '''
    Split into testing and training sets.
    Now let's create a stratified sample based on the income category.
    This is just a test to make sure that the distribution of the income category
    in the testing data set is similar to the training data, reducing bias.
    '''

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    strat_train_set = None
    strat_test_set = None
    for train_index, test_index in split.split(df, df["income_cat"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    # Remove income_cat so that the data is back to its original state
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
	
    # Now let's start exploring the new training set
    housing_df = strat_train_set.copy()
    housing_df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=housing_df["population"]/100, label="population", figsize=(10,7),
        c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
        sharex=False)
    plt.legend()
    save_fig('california_housing_prices')

    # Get the Pearson (linear) correlation coefficient
    corr_matrix = housing_df.corr()
    if debug: 
		print(corr_matrix)

    # Make a few general correlation plots
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    pd.plotting.scatter_matrix(housing_df[attributes], figsize=(12, 8))
    save_fig("scatter_matrix_plot")	
	
    # Data cleaning
    housing_df_targets = housing_df["median_house_value"].copy() # Stratified training set targets only
    housing_df.dropna(subset=["total_bedrooms"]) # Remove the NaN data
    housing_df.drop("median_house_value", axis=1) # Stratified training set without targets

    '''
    We need to do a lot of extra cleaning. We will need a SimpleImputer to take care of
    missing values. We'll then need to convert text to an integer category, and then those categories 
    to a one-hot vector format. This will prevent the ML procedure learning misleading
    patterns later on. Let's put all of this in a Pipeline for simplicity. Note that we'll use
	a categorical encoder for the one-hot conversion.
    '''

    housing_df_num = housing_df.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_df_num)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
		('selector', DataFrameSelector(num_attribs)),
		('imputer', SimpleImputer(strategy="median")),
		('std_scaler', StandardScaler()), # Rescale to be in the intevral [-1,1)
    ])

    cat_pipeline = Pipeline([
		('selector', DataFrameSelector(cat_attribs)),
		('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])

    # Combine the numerical and categorical pipelines
    full_pipeline = FeatureUnion(transformer_list = [
		("num_pipeline", num_pipeline),
		("cat_pipeline", cat_pipeline),
    ])
	
    # Now run the entire pipeline 
    housing_df_prepared = full_pipeline.fit_transform(housing_df)
    if debug: 
		print("Training data processed: ", housing_df_prepared)

    '''
    Now we move to training, and here we will make use of K-fold 
    cross-validation. We'll take advantage of ensemble learning
    techniques (e.g. bagging and boosting), using a 
    Random Forest regressor.
    '''
    forest_params = {
		'max_depth': [10, 30, None],
                'max_features': ['auto', 'sqrt', 'log2'], 
                'n_estimators': [3, 10],
    }
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid=forest_params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(housing_df_prepared, housing_df_targets)
    best = grid_search.best_estimator_
    print ('The best estimator is: ', str(best))
    
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    	print(np.sqrt(-mean_score), params)

    '''
    Move to testing with the best estimator
    '''
    y_test = strat_test_set["median_house_value"].copy()
    X_test = strat_test_set.drop("median_house_value", axis=1)
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = best.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    print('MSE with test set: ', final_mse)
    print('RMSE with test set: ', final_rmse)
    
    # Determine the 95 % confidence interval
    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    mean = squared_errors.mean()
    m = len(squared_errors)

    ci_array = np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))

    print('The 95 % confidence interval is: ', ci_array)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", help="", action="store_true", default=False)
	
    options = parser.parse_args()

    # Defining dictionary to be passed to the main function
    option_dict = dict( (k, v) for k, v in vars(options).iteritems() if v is not None)
    print option_dict
    main(**option_dict)
