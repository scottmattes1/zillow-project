import pandas as pd
import os
from env import get_db_url


############################ IMPORTS #########################

#standard ds imports
import pandas as pd
import numpy as np
import os

#visualization imports
import matplotlib.pyplot as plt
import seaborn as sns


#import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

#ignore warnings
import warnings
warnings.filterwarnings("ignore")


###################### ACQUIRE ZILLOW FUNCTION ##########################

def acquire_zillow():
    ''' This function checks to see if a csv with the zillow dataset exists, and if not, runs a SQL query to pull from database and save it in a new csv file.'''
    
    filename = 'zillow.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)

    else:
        sql = '''
SELECT p.*, pr.*
FROM properties_2017 p
JOIN predictions_2017 pr ON p.parcelid = pr.parcelid
WHERE p.propertylandusetypeid = 261;'''

        df = pd.read_sql(sql, get_db_url('zillow'))

        df.to_csv(filename, index=False)

        return df
    
    
############################ PREPARE ZILLOW FUNCTION ###########################

def prep_zillow(df):
    '''
    This function takes in the zillow df then the data is cleaned and returned
    '''

    # Select the required columns
    # Bathrooms was determined to be multicolinear with bedrooms during explore so even though the curriculum scenario asked me to predict using bathrooms, it is being dropped from the features I will model off of
    df = df[['calculatedfinishedsquarefeet', 'bedroomcnt', 'taxvaluedollarcnt']]
    
   
    #change column names to be more readable
    df = df.rename(columns={'calculatedfinishedsquarefeet': 'square_feet', 'bedroomcnt': 'bedrooms', 'taxvaluedollarcnt': 'home_value'})
    
    #filters out outliers (homes above 1M), these higher values are heteroskedastic with predictive features so they will only make the model worse. Homes above $1M only make up 10% of homes bought in 2017
    df = df[df['home_value'].astype(float) < 1000000]

    #drop null values, null values make up less than 1% of the data so this won't effect modeling too much
    df = df.dropna()

    #drop duplicates
    df.drop_duplicates(inplace=True)
   
    return df


############################ WRANGLE ZILLOW RAW FUNCTION ############################

def wrangle_zillow_raw():
    """
    Retrieves the raw zillow data and cleans it for use in exploration visualizations
    """
    zillow_raw = acquire_zillow()
    zillow_raw = zillow_raw.rename(columns={'taxvaluedollarcnt': 'home_value', 'calculatedfinishedsquarefeet': 'square_feet'})
    zillow_raw = zillow_raw[['home_value', 'square_feet']]
    zillow_raw = zillow_raw.dropna()
    zillow_raw.drop_duplicates(inplace=True)
    return zillow_raw
    

############################ WRANGLE ZILLOW FUNCTION ############################

def wrangle_zillow():
    '''
    This function acquires and prepares our Zillow data
    and returns the clean dataframe
    '''
    df = prep_zillow(acquire_zillow())
    
    print("The zillow data has been acquired, cleaned, and stored in the zillow_df dataframe")
    
    return df


############################ SCALE ZILLOW FUNCTION ############################

def scale_zillow(X_train, X_validate, X_test):
    
    cols = ['square_feet', 'bedrooms', 'square_feet']
    
    scaler = MinMaxScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[cols]), columns=cols, index=X_train.index)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate[cols]), columns=cols, index=X_validate.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[cols]), columns=cols, index=X_test.index)
    
    return X_train_scaled, X_validate_scaled, X_test_scaled

############################ SPLIT ZILLOW FUNCTION ############################

def split_zillow(df):
    '''
    This function takes in the dataframe and splits it into train, validate, test datasets. This is a 60/20/20 split
    '''    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=13)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=13)
    
    return train, validate, test


############# X_Y_SPLIT FUNCTION #####################
    
    
def x_y_split(train, validate, test):
    """
    Separate the features (X) and target variable (y) for the train, validate, and test sets
    """
    X_train, y_train = train.drop(columns='home_value'), pd.DataFrame(train.home_value)
    X_validate, y_validate = validate.drop(columns='home_value'), pd.DataFrame(validate.home_value)
    X_test, y_test = test.drop(columns='home_value'), pd.DataFrame(test.home_value)
    
    # Add mean baseline columns to y_train and y_validate
    baseline = y_train.home_value.mean()
    y_train['baseline'] = baseline

    baseline = y_validate.home_value.mean()
    y_validate['baseline'] = baseline
    
    print("Train, validate, and test have been split into x_ and y_ dataframes. Mean baseline columns have been added to y_train and y_validate.")
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

    ########################################## AQUIRE ZILLOW FUNCTION #######################################

def split_clean_zillow(df):
    '''
    This function splits our clean dataset into 
    train, validate, test datasets
    '''
    train, validate, test = (split_zillow(df))
    
    print(f"train shape: {train.shape}   -- {round(train.shape[0]/df.shape[0], 1) *100}% of zillow_df")
    print(f"validate shape: {validate.shape} -- {round(validate.shape[0]/df.shape[0], 1) *100}% of zillow_df")
    print(f"test shape: {test.shape}     -- {round(test.shape[0]/df.shape[0], 1) *100}% of zillow_df")
    
    return train, validate, test
  
    
######################## IDENTIFY FIPS FUNCTION ########################

    
def identify_fips():
    """
    Wrangles zillow data and a csv with state and county data and merges them based on the fips data. Returns a dataframe.
    """
    import wrangle
    # Bring in zillow so we can append the county and state to the listings
    zdf = wrangle.acquire_zillow()

    # Bring in the home id and fips info so that these homes can be identified
    zdf = zdf[['id', 'fips']]
    zdf.head()

    # Read in the fips data
    fips_df = pd.read_csv('fips-by-state.csv', encoding='latin-1')

    # Rename the 'name' column to 'county'
    fips_df = fips_df.rename(columns= {'name':'county'})

    locations_df = zdf.merge(fips_df, on='fips', how='left')
    locations_df.shape
    
    return locations_df


############## CHECK NULLS DISPLAY FIPS DF FUNCTION #######################

def check_nulls_display_fips_df(fips_df):
    """
    Checks the fips_df for nulls and prints the results, then displays the dataframe
    """
    print('\nCount of all nulls in the dataframe by column:')
    display(fips_df.isnull().sum())
    print('\nNo nulls were found.\n\nThis is a truncated display of the transaction id dataframe with state and county info matched by fips code:')
    display(fips_df)
 
