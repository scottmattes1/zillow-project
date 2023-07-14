import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.metrics import explained_variance_score
import warnings


############ APPLY HIGHLIGHTS FUNCTION ###############

def apply_highlights(df, column_name, value_list):
    """
    A helper function used in the build_regression_models function to stylize the results_df dataframe to highlight poor performing models in red, good models in yellow, and great models in green.
    """
    
    # Create a Styler object for the DataFrame
    styled_df = df.style

    # Create a boolean mask to identify rows with values in the specified list
    mask = df[column_name].isin(value_list)
    
    # variables to calculate red rows
    baseline = df.rmse_validate[0]
    one_percent = df.rmse_validate[0] * .01
    
    # variables to calculate green rows
    lowest_difference_val = df.model_difference.sort_values().head(1).tolist()[0]
    rmse_of_lowest_diff_val = df.groupby('model_difference').min().head(1).rmse_validate.tolist()[0]
    
    # Apply the desired styling to the rows in the mask
    styled_df = styled_df.apply(lambda row: ['background-color: yellow'] * len(row) if row[column_name] in value_list else [''] * len(row), axis=1)
    
    styled_df = styled_df.apply(lambda row: ['background-color: limegreen'] * len(row) if (row.model_difference <= value_list[0] + (value_list[0] * .000001)) and (row.rmse_validate <= rmse_of_lowest_diff_val) else [''] * len(row), axis=1)
    
    styled_df = styled_df.apply(lambda row: ['background-color: red'] * len(row) if (row.rmse_validate > baseline - one_percent) and (row.rmse_validate < baseline + one_percent) else [''] * len(row), axis=1)


    return styled_df


############ BUILD REGRESSION MODELS FUNCTION ###############

def build_regression_models(X_train, y_train, X_validate, y_validate):
    """
    Builds an OLS model, 5 LassoLars models, and 3 polynomial models and outputs them in a dataframe that highlights the good and bad performers with a legend to understand what each color means
    """
    # Calculate the RMSE for the baseline model
    rmse_train_mu = mean_squared_error(y_train.home_value, y_train.baseline) ** 0.5
    rmse_validate_mu = mean_squared_error(y_validate.home_value, y_validate.baseline) ** 0.5

    # Initialize a dataframe to store the evaluation metrics and add the baseline RMSE
    results_df = pd.DataFrame(data=[{
    'model': 'Mean Baseline',
    'rmse_train': round(rmse_train_mu,2),
    'rmse_validate': round(rmse_validate_mu),
    'r_validate': round(explained_variance_score(y_validate.home_value, y_validate.baseline),2),
    'model_difference': round(abs(rmse_validate_mu - rmse_train_mu),2)
    }])
    
    # Build, evaluate, and append the OLS model to the results dataframe
    OLSmodel = LinearRegression()
    OLSmodel.fit(X_train, y_train.home_value)
    y_train['value_pred_ols'] = OLSmodel.predict(X_train)
    rmse_train = mean_squared_error(y_train.home_value, y_train.value_pred_ols) ** 0.5
    y_validate['value_pred_ols'] = OLSmodel.predict(X_validate)
    rmse_validate = mean_squared_error(y_validate.home_value, y_validate.value_pred_ols) ** 0.5
    results_df = results_df.append({
        'model': 'OLS',
        'rmse_train': round(rmse_train,2),
        'rmse_validate': round(rmse_validate,2),
        'r_validate': round(explained_variance_score(y_validate.home_value, y_validate.value_pred_ols),2),
        'model_difference': round(abs(rmse_validate - rmse_train),2)
    }, ignore_index=True)
    
    # Build, evaluate, and append 5 LassoLars models to the results dataframe
    for i in range(1, 6):
        #Make the thing
        lassomodel = LassoLars(alpha=i/100)
        #Fit the thing
        lassomodel.fit(X_train, y_train.home_value)
        #Use the thing
        y_train['value_pred_lasso'] = lassomodel.predict(X_train)
        #Calculate performance metrics
        rmse_train = mean_squared_error(y_train.home_value, y_train.value_pred_lasso) ** 0.5
        #Repeat steps on the validate set
        y_validate['value_pred_lasso'] = lassomodel.predict(X_validate)
        rmse_validate = mean_squared_error(y_validate.home_value, y_validate.value_pred_lasso) ** 0.5
        results_df = results_df.append({
            'model': 'LassoLars_a' + str(i/100),
            'rmse_train': round(rmse_train,2),
            'rmse_validate': round(rmse_validate,2),
            'r_validate': round(explained_variance_score(y_validate.home_value, y_validate.value_pred_lasso),2),
            'model_difference': abs(rmse_validate - rmse_train)
        }, ignore_index=True)
    
    # Build, evaluate, and append 3 Polynomial models to the results dataframe
    for i in range(2, 5):
        poly = PolynomialFeatures(degree=i, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_validate_poly = poly.transform(X_validate)
        poly_reg = LinearRegression()
        poly_reg.fit(X_train_poly, y_train.home_value)
        y_train['value_pred_poly'] = poly_reg.predict(X_train_poly)
        rmse_train = mean_squared_error(y_train.home_value, y_train.value_pred_poly) ** 0.5
        y_validate['value_pred_poly'] = poly_reg.predict(X_validate_poly)
        rmse_validate = mean_squared_error(y_validate.home_value, y_validate.value_pred_poly) ** 0.5
        results_df = results_df.append({
            'model': 'Polynomial_deg' + str(i),
            'rmse_train': round(rmse_train,2),
            'rmse_validate': round(rmse_validate,2),
            'r_validate': round(explained_variance_score(y_validate.home_value, y_validate.value_pred_poly),2),
            'model_difference': round(abs(rmse_validate - rmse_train),2)
        }, ignore_index=True)
    
    
    # create styled_df
    lowest_differences = results_df.model_difference.sort_values()[:3].tolist()
    styled_df = apply_highlights(results_df, 'model_difference', lowest_differences)


    # styling variables for legend string
    HIGHLIGHT_YELLOW = '\033[1;33;93m'  
    HIGHLIGHT_RED = '\033[1;33;91m'
    HIGHLIGHT_GREEN = '\033[1;33;92m'
    HIGHLIGHT_END = '\033[0m;'


    print(f"""\n
10 regression models were successfully generated.
    
{HIGHLIGHT_YELLOW}Yellow: the three models with the lowest train-validate difference (green may overlap with these and leave only two yellow rows){HIGHLIGHT_END}\n{HIGHLIGHT_RED}Red: the baseline plus all models within one percent of the baseline rmse{HIGHLIGHT_END}\n{HIGHLIGHT_GREEN}Green: either the model with the lowest train-validate difference or a model with a very low train-validate difference and the lowest validate rmse{HIGHLIGHT_END}""")
    display(styled_df)
    print("\nExamine the above models and determine the best one for your purposes.\nGenerally, a green model will perform the best.\n")
    
    return results_df, styled_df



############ VISUALIZE MODEL PERFORMANCE FUNCTION ###############

def visualize_model_performance(results_df):
    """
    Creates a line graph of the train and validate performance of the models listed in the results_df
    """
    
    # Store performance scores from the results_df dataframe as lists for a matplotlib plot
    train_scores = results_df.rmse_train.tolist()
    val_scores = results_df.rmse_validate.tolist()
    models = results_df.model

    # Plot the train performance and validate performance of each model, with formatting
    plt.figure(figsize=(26, 13))
    plt.plot(models, train_scores, marker='o', markersize=10, label='Train Scores', linewidth=4)
    plt.plot(models, val_scores, marker='o', markersize=10, label='Validation Scores', linewidth=4)

    plt.xticks(rotation=45, fontsize=20, fontweight='bold')
    plt.tick_params(axis='x', which='both', length=15, width=2)

    plt.yticks(fontsize=20, fontweight='bold')
    plt.gca().set_yticklabels(["${:,.0f}K".format(label / 1000) for label in plt.gca().get_yticks()])

    plt.xlabel('Models', fontsize=24, labelpad=20)
    plt.ylabel("RMSE", fontsize=24, labelpad=20)
    plt.title('Train vs. Validation Performance of the Different Models', fontsize=30, fontweight='bold', pad=30)

    plt.legend(prop={'size': 25})
    plt.grid(axis='y')

    print('''
    ''')
    plt.show()
    
    return


################# EVALUATE BEST MODEL FUNCTION ##################

def evaluate_best_model(X_train, y_train, X_validate, y_validate, X_test, y_test):
    """
    Rebuilds the best model and evaluates its performance on the test data then displays a dataframe with the results
    """
    #Make the thing
    lassomodel = LassoLars(alpha=.01)
    #Fit the thing
    lassomodel.fit(X_train, y_train.home_value)
    #Use the thing
    y_train['value_pred_lasso'] = lassomodel.predict(X_train)
    #Calculate performance metrics
    rmse_train = mean_squared_error(y_train.home_value, y_train.value_pred_lasso) ** 0.5

    #Repeat steps on the validate set
    y_validate['value_pred_lasso'] = lassomodel.predict(X_validate)
    rmse_validate = mean_squared_error(y_validate.home_value, y_validate.value_pred_lasso) ** 0.5

    #Repeat steps on the validate set
    y_test['value_pred_lasso'] = lassomodel.predict(X_test)
    rmse_test = mean_squared_error(y_test.home_value, y_test.value_pred_lasso) ** 0.5

    final_df = pd.DataFrame(data={
        'model': ['LassoLars_a0.01'],
        'rmse_train': [round(rmse_train, 2)],
        'rmse_validate': [round(rmse_validate, 2)],
        'rmse_test': [round(rmse_validate, 2)],
        'r_test': [round(explained_variance_score(y_test.home_value, y_test.value_pred_lasso), 2)],
        'model_difference': [abs(rmse_test - rmse_train)]
    })

    final_df.set_index('model', inplace=True)  # Set 'model' column as the index
    
    styled_df = final_df.style.applymap(lambda _: 'background-color: yellow').format('{:.2f}')

    print("\nModel performace:")
    display(styled_df)


################### VISUALIZE BEST MODEL FUNCTION #####################

    
def visualize_best_model():
    """
    Creates a graph that plots the baseline, model line, and average data point, displays the RMSE and TSS as well
    """

    # Define the x-axis range
    x = np.linspace(200, 7500, 100)
    # Set the slope of the line to be 45 degrees across the chart
    slope = 1000000 / (7500 - 200)
    # Compute the corresponding y-values
    y = slope * (x - 200)
    # # Create the figure and axes
    fig, ax = plt.subplots(figsize=(15, 10))
    # Set title
    plt.title('Model predictions are 22% closer than baseline', weight='bold', fontsize=20)
    # Plot the diagonal line with the desired slope
    ax.plot(x, y-70000, linestyle='-', color='red', label='LassoLars_a.01 Model', linewidth=4)

    # Plot the dotted line at 368500
    ax.axhline(y=368500, linestyle=':', color='blue', label='Baseline Model', linewidth=4)

    # Set the x and y-axis limits
    ax.set_xlim(800, 7500)
    ax.set_ylim(2000, 1100000)

    # Hide the x and y-ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add labels and a legend
    ax.set_xlabel('Features', weight= 'bold', fontsize=12)
    ax.set_ylabel('Home Value', weight= 'bold', fontsize=12)
    ax.legend()

    # Green vertical line that represents RMSE across the middle third of the screen
    ax.axvline(x=4050, ymin=0.445, ymax=0.60, color='limegreen', linewidth=5)

    # blue vertical line that represents TSS across the middle third of the screen
    ax.axvline(x=3950, ymin=0.36, ymax=0.60, color='blue', linewidth=5)

    # Add a text box at position
    ax.text(2800, 630000, 'RMSE = $208K', fontsize=12, weight='bold', color='yellow', bbox=dict(facecolor='limegreen', edgecolor='limegreen'))

    # Add a text box at position
    ax.text(2800, 570000, ' TSS = $254K  ', fontsize=12, weight='bold', color='yellow', bbox=dict(facecolor='blue', edgecolor='blue'))

    # Add a text box at position
    ax.text(3400, 810000, 'Average Home Value', fontsize=12, weight='bold', color='white', bbox=dict(facecolor='grey', edgecolor='grey'))

    # Plot a large dot
    large_dot_x = 3995
    large_dot_y = slope * (large_dot_x - 200) + 207884
    ax.plot(large_dot_x, large_dot_y, marker='o', markersize=45, color='orange', label='Large Dot')
    plt.figure(figsize=(40,20))

    # Display the plot
    print('\n')
    plt.show()
