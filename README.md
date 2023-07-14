# <a name="top"></a>Zillow Project - Predicting Home Values 
![]()


***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquisition and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
[[Recommendations](#recommendations)]
[[Next Steps](#next_steps)]
___



## <a name="project_description"></a>Project Description:
[[Back to top](#top)]

In this project we will be using the Zillow Data Set. Exploring the data, we will find features that are correlated with Property Tax Assesed Values (taxvaluedollarcnt) using PearsonsR, in order to run features through a model that will predict the Property Tax Assessed Values. The goal is to beat baseline using one of the four regression models: Linear Regression(OLS), LassoLars, and Polynomial Regression.



***
## <a name="project_planning"></a>Project Planning: 
[[Back to top](#top)]


### Objective

For the Zillow Data Science team:
1. Provide recommendations on how to build a better model (things that do and don't work
2. Determine the states and counties where the fips are located


### Target variable
The target variable of this project is home_value.



***

## <a name="key_findings"></a>Key Findings:
[[Back to top](#top)]

1. The bedrooms relationship to home_value is extremely weak yet significant¶

2. The square_feet relationship to home_value is very strong and significant

3. Bedrooms and square_feet have some degree of multicolinearity, however there is high variance in the relationship so both features will still be taken to modeling




***

## <a name="data_dictionary"></a>Data Dictionary  
[[Back to top](#top)]

### Data Used
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| bedroom_count|the amount of bedrooms in the home | float |
| square_feet |the calculated square footage of the home|float |
| home_value |the home's value ($) | float |
**


## <a name="data_acquisition_and_preparation"></a>Data Acquisition and Preparation
[[Back to top](#top)]



### Wrangle steps: 
- dropped unwanted columns.

- renamed columns for readability:
 'bedroomcnt': 'bedrooms',
 'calculatedfinishedsquarefeet': 'square_feet',
 'taxvaluedollarcnt': 'home_value',
 'fips': 'county_code'
 
- created function to acquire and prep data

- function created to scale certain features


*********************

## <a name="data_exploration"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:
    - wrangle.py
    - explore.py
    

### Takeaways from exploration:
- Two features were chosen for statistical testing: bedroom count, calculated square feet


***

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]

### Stats Test 1: Pearson's R

Pearson's correlation coefficient (Pearson's R) is a statistical measure used to assess the strength and direction of the linear relationship between two continuous variables.

By calculating Pearson's R, we aim to determine whether there is a significant linear association between the independent variable and the dependent variable. The coefficient helps us quantify the extent to which the variables vary together and provides insight into the direction (positive or negative) and strength (magnitude) of the relationship.

To calculate Pearson's R in Python, we can use the corrcoef function from the numpy module. This function takes the two variables as input and returns the correlation matrix, where the coefficient of interest is the element in the [0, 1] or [1, 0] position. Pearson's R ranges from -1 to 1, where -1 indicates a perfect negative linear relationship, 0 indicates no linear relationship, and 1 indicates a perfect positive linear relationship.


### Hypothesis

In summary, the hypotheses for the PearsonsR test can be stated as follows:

### 1st Hypothesis

Null hypothesis: There is no significant relationship between bedrooms and home_value
Alternative hypothesis: There is a significant relationship between bedrooms and home_value

----Results-----

R coefficient: 0.24 
P-value: 0.0

The p-value is less than the alpha. The relationship between bedrooms and home_value is significant



### 2nd Hypothesis

Null hypothesis: There is no significant relationship between square_feet and home_value

Alternative hypothesis: There is a significant relationship between square_feet and home_value


----Results-----

R coefficient: 0.46 
P-value: 0.0

The p-value is less than the alpha. The relationship between square_feet and home_value is significant



#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05


#### Summary: 
bathroom_count and calc_sqr_ft have a moderate correlation with the target(tax_value). However, yearbuilt, bedroom_count and the county code dummies have a very week correlation with the target.



***

## <a name="modeling"></a>Modeling:
[[Back to top](#top)]



| model | rmse_train | rmse_validate | r_validate | model_difference
| ---- | ----| ---- | ---- | ---- 
| Mean Baseline     | 235901.40 | 234884.00 | 0.00 | 1017.64
| OLS               | 208789.54 | 207870.05 | 0.22 | 919.49
| LassoLars_a0.01   | 208788.06 | 207884.91 | 0.22 | 903.16
| LassoLars_a0.02   | 208788.06 | 207884.91 | 0.22 | 903.16
| LassoLars_a0.03   | 208788.06 | 207884.90 | 0.22 | 903.16
| LassoLars_a0.04   | 208788.06 | 207884.90 | 0.22 | 903.16
| LassoLars_a0.05   | 208788.06 | 207884.90 | 0.22 | 903.16
| Polynomial_deg2   | 208329.19 | 206904.01 | 0.22 | 1425.18
| Polynomial_deg3   | 208116.63 | 206708.08 | 0.23 | 1408.55
| Polynomial_deg4   | 208054.86 | 206568.79 | 0.23 | 1486.07


##### LassoLars Regression preformed best with the lowest train-validate spread


## Testing the Model

- Model Testing Results

| model | RMSE_train | RMSE_validate | RMSE_test | model_difference 
| ---- | ----| ---- | ---- | ---- | ---- 
|LassoLars_a0.01 | 208788.06 | 207884.91 | 207884.91 | 0.22 | 877.13

The best performing regression model on this data set was a LassoLars model with an alpha of .01 which performed on test data with an RMSE of $207,884 and an R squared value of .22



***

## <a name="conclusions"></a>Conclusions:
[[Back to top](#top)]

#### The project objectives were met: recommendations to build a better model were presented and state and county #### information were matched and added to the list of 2017 home transactions.

***
## <a name="recommendations"></a>Recommendations:
[[Back to top](#top)]

#### Merging the bedrooms and bathrooms information into a ratio does not add value to a model because it increases feature variance when plotted against home_value. Do not combine these features in future models.

#### Eliminating outliers in the home value distribution will increase model performance. Only 10% of the transactions are above 1M, eliminating these from the data set reduces heteroskedasticity and dramatically improves model performance

#### The dataframe I was able to create which matches state and county information to each transaction id may be added to the Zillow database server, possibly using transaction 'id' as the primary key and 'fips' as a foreign key


***
## <a name="next_steps"></a>Next Steps:
[[Back to top](#top)]
#### To further improve model performance, feature selection methods may be used to determine other columns in the data that can add predictive power to future models
#### It may be useful to experiment with trimming outlier values from other model features as well, besides just the home value
#### If desirable to the Zillow Data Science team, a model may be trained that specifically with predicting extreme target values (potentially a clustering model), this could quite possibly be more accurate than this model would be with high value homes


***