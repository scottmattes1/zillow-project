
***
[[Objectives]]
[[Project Plan]]
[[Key Findings]]
[[Modeling]]
[[Recommendations]]
[[Next Steps]]
[[Steps to Reproduce]]
[[Data Exploration]]
[[Conclusion]]
[[Data Dictionary]]
___


Objectives
-----------------
For the Zillow Data Science team:
1. Provide recommendations on how to build a better model (things that do and don't work
2. Determine the states and counties where the fips are located



Project Plan
-----------------
Follow the data science pipeline to find insights to present to the Zillow Data Science team on how to improve model performance to predict home values; identify the location of each transaction. Produce a GitHub repository and a final notebook from which to present findings.



Key Findings
-----------------
1. The bedrooms relationship to home_value is extremely weak yet significant¶

2. The square_feet relationship to home_value is very strong and significant

3. Bedrooms and square_feet have some degree of multicolinearity, however there is high variance in the relationship so both features will still be taken to modeling


Modeling
-----------------------
The best performing regression model on this data set was a LassoLars model with an alpha of .01 which performed on test data with an RMSE of $207,884 and an R squared value of .22



Recommendations
-----------------
1. Merging the bedrooms and bathrooms information into a ratio does not add value to a model because it increases feature variance when plotted against home_value. Do not combine these features in future models.

2. Eliminating outliers in the home value distribution will increase model performance. Only 10% of the transactions are above 1M, eliminating these from the data set reduces heteroskedasticity and improves model performance.

3. The dataframe I was able to create which matches state and county information to each transaction id may be added to the Zillow database server, possibly using transaction 'id' as the primary key and 'fips' as a foreign key.



Next Steps
-----------------
To further improve model performance, feature selection methods may be used to determine other columns in the data that can add predictive power to future models

It may be useful to experiment with trimming outlier values from other model features as well, besides just the home value

If desirable to the Zillow Data Science team, develop a model that works well at predicting extreme target values (potentially a clustering model)



Steps to Reproduce
-----------------------
1. Make an env.py file with your credentials to access the codeup database
2. clone this repository to a local directory
3. Follow the hyperlink in the 'State and County Data' section at the bottom of the notebook and 'curl -O' the raw file into your local directory
4. Run all cells of the notebook



Conclusion
-----------------------
The project objectives were met: recommendations to build a better model were presented and state and county information were matched and added to the list of 2017 home transactions.



Data Dictionary
------------------
##Column Name           |       Description
------------------------------------------------------------------------------
home_value                    The taxable value of each home. Given as a float.
square_feet                   The square footage of each home. Given as a float.
bedrooms                      The bedroom count of each home. Given as a float.







