#stats testing methods
from scipy.stats import pearsonr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

############# HOMES BEFORE AFTER OUTLIERS FUNCTION ################

def homes_before_after_outliers(zillow_raw, zillow_df):
    """
    Displays two subplots charting the distribution of the zillow_raw.home_values and the train.home_values columns
    """
    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first displot on the left subplot
    sns.histplot(data=zillow_raw, x="home_value", ax=axes[0])
    axes[0].set_title('Distribution of Home Values Before Handling Outliers')
    axes[0].set_ylim(0, 2200)
    axes[0].set_xlim(0, 5000000)  # Set the xlim for the left subplot
    
    # Plot the second displot on the right subplot
    sns.histplot(data=zillow_df, x="home_value", ax=axes[1])
    axes[1].set_title('Distribution of Home Values After Handling Outliers')

    plt.tight_layout()  # Adjust the spacing between subplots

    # Display the subplots
    plt.show()


########### STATS TEST PEARSONS FUNCTION ##############

def stat_test_pearsons(df, predictor, target):
    """
    Conducts a pearsons r stats test on two continuous columns and displays the results
    """
    # Set the alpha value
    a=.05

    # Run the Pearson R stats test
    r, p = pearsonr(df[predictor], df[target])
    print(f"""

Null hypothesis: There is no significant relationship between {predictor} and {target}

Alternative hypothesis: There is a significant relationship between {predictor} and {target}


----Results-----

R coefficient: {round(r,2)} 
P-value: {p}""")

    # Print the results
    if p < a:
        print(f"""
The p-value is less than the alpha. The relationship between {predictor} and {target} is significant\n""")
    else:
        print(f"""
The p-value is higher than the alpha. The relationship between {predictor} and {target} is not significant\n""")
