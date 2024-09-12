import pandas as pd
import re
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

def burg_per_lsoa(df):
    '''
    Function to create a dataframe with the
    number of burglaries per LSOA
    '''
    df_lsoa_burg = df.groupby(['LSOA code']).size().to_frame()
    df_lsoa_burg = df_lsoa_burg.rename(columns={0: 'Number of Burglaries'}).reset_index()
    return df_lsoa_burg

def demographics_data():
    '''
    Initialize the 1000+ column dataset with demographics
    for the LSOA's in Barnet.
    '''
    # Load data into dataframe
    df_lsoa_demographics = pd.read_csv("lsoa-data-old-boundaries-DataSheet.csv", encoding= 'unicode_escape')

    # Create dataframe for only Barnet's LSOAs
    df_barnet_demographics = df_lsoa_demographics[df_lsoa_demographics['Names'].str.contains('Barnet')].reset_index().drop(columns=['index'])

    # Find the index of the row to be copied
    index_to_copy = df_barnet_demographics[df_barnet_demographics['Lower Super Output Area'] == 'E01000149'].index[0]

    # Copy the existing row and delete it
    new_row = df_barnet_demographics.loc[index_to_copy].copy()
    df_barnet_demographics = df_barnet_demographics.drop([index_to_copy])

    length = len(df_barnet_demographics)
    # Modify the column value for the new instances
    new_row['Lower Super Output Area'] = 'E01033572'
    df_barnet_demographics.loc[(length+1)] = new_row

    new_row['Lower Super Output Area'] = 'E01033573'
    df_barnet_demographics.loc[(length+2)] = new_row
    return df_barnet_demographics

def demo_no_crime(df_demo):
    '''
    Remove the 'Crime' columns from the 1000+ column demographics
    dataset for feature selection.
    '''

    crime_columns = [col for col in df_demo.columns if col.startswith('Crime')]
    df_demo_no_crime = df_demo.drop(columns=crime_columns)

    return df_demo_no_crime

def rfe(X, y):
    '''
    Function for feature selection based on RFE.
    TAKES LONG TOO RUN!
    '''
    # Select columns with only numbers as entries, remove NaN values and initialize X and y data
    X = X.select_dtypes(include=[np.number])
    X = X.dropna(axis=1)
    y = y['Number of Burglaries']

    # Create the estimator
    estimator = LogisticRegression()

    # Create the RFE object and specify the desired number of features
    rfe = RFE(estimator, n_features_to_select=50)  # Replace 50 with the desired number of features

    # Perform feature selection
    X_selected = rfe.fit_transform(X, y)

    # Get the selected feature names
    selected_feature_names = X.columns[rfe.support_]

    # Print the selected feature names
    print("Selected Features:")
    # for feature in selected_feature_names:
        # print(feature)

    return selected_feature_names

def corr_feature_selection(df_demo, df_lsoa_burg):
    '''
    Feature selection based on correlation between
    the variables and the number of burglaries
    '''
    # Add the number of burglaries to the demographics dataframe using the LSOA codes
    df = pd.merge(df_lsoa_burg, df_demo, left_on='LSOA code', right_on='Lower Super Output Area', how='inner')
    df = df.select_dtypes(include=[np.number])

    # Calculate the correlation coefficients between each variable and 'Burglaries'
    correlations = df.corr()['Number of Burglaries'].abs().sort_values(ascending=False)

    # Remove 'Burglaries' itself from the list
    correlations = correlations.drop('Number of Burglaries')

    # Print the ranked list of variables
    # print(correlations)
    return correlations

def rfr(X, y):
    '''
    Feature selection using Random Forest Regressor
    which ranks the features on their importance
    '''
    # Select columns with only numbers as entries, remove NaN values and initialize X and y data
    X = X.select_dtypes(include=[np.number])
    X = X.dropna(axis=1)
    y = y['Number of Burglaries']

    # Train the Random Forest model
    rf = RandomForestRegressor()
    rf.fit(X, y)

    # Get feature importances
    feature_importances = rf.feature_importances_

    # Create a dataframe to store feature names and their importances
    feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    # Sort the dataframe by feature importance in descending order
    feature_importances_df = feature_importances_df.sort_values('Importance', ascending=False)

    # Print the ranked list of features
    # print(feature_importances_df)
    return feature_importances_df

# Apparently, indices were in csv-file too so those have to be dropped
df_barnet = pd.read_csv('barnet.csv').drop(columns=['Unnamed: 0.1', 'index', 'Unnamed: 0'])
df_lsoa_burg = burg_per_lsoa(df_barnet)
df_demographics = demographics_data()
df_demo_no_crime = demo_no_crime(df_demographics)

# RFE feature selection
# Uncomment if you want to use. Takes long time to run, therefore commented out right now.
# features_rfe = rfe(df_demographics, df_lsoa_burg)
# print(features_rfe)

# Correlation based feature selection
# print(corr_feature_selection(df_demographics, df_lsoa_burg)[:50])

# Random Forest Regressor feature selection
#print(rfr(df_demographics, df_lsoa_burg)[:50])

# RFE feature selection without the Crime columns included as possible features
# Uncomment if you want to use. Takes long time to run, therefore commented out right now.
# features_rfe_no_crime = rfe(df_demo_no_crime, df_lsoa_burg)
# print(features_rfe_no_crime)

# Correlation based feature selection without the Crime columns included as possible features
print(corr_feature_selection(df_demo_no_crime, df_lsoa_burg)[-50:])

# Random Forest Regressor feature selection without the Crime columns included as possible features
# print(rfr(df_demo_no_crime, df_lsoa_burg)[:50])

