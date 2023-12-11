import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import os



def plot_original():
    """
    Plot the original data features versus time, Rain, Temperature, Relative Humidity.

    Parameters:
    None

    Returns:
    - Displays plot, doesn't return
    """


   # load the dataset
    df = pd.read_csv('./data_0_1/Both.csv')

    # convert 'day', 'month', and 'year' columns to datetime
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # set the date as the index
    df.set_index('date', inplace=True)

    # create a figure with subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18))

    # plot the 'Temperature' column over time
    df['Temperature'].plot(ax=axes[0], title='Temperature over time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Temperature (degrees Celsius)')

    # plot the 'Rain' column over time
    df['Rain'].plot(ax=axes[1], title='Rain over time')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Rain')

    # plot the 'RH' column over time
    df['RH'].plot(ax=axes[2], title='Relative Humidity over time')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Relative Humidity')

    # show the plots
    plt.tight_layout()
    plt.show()

def linear_regression_plot(start_date, end_date, feature):
    """
    Perform linear regression for a specified time span and plot the regression line.

    Parameters:
    - start_date: str, start date of the time span in the format 'YYYY-MM-DD'
    - end_date: str, end date of the time span in the format 'YYYY-MM-DD'
    - feature: str, the feature for which linear regression is to be performed

    Returns:
    - Displays plot, doesnt return
    """

    # read the dataset
    df = pd.read_csv('./data_0_1/both.csv')

    # convert 'day', 'month', and 'year' columns to datetime
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # set the date as the index and sort the DataFrame by the index
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # convert start_date and end_date to datetime 
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # get data for the specified time span
    subset_df = df.loc[start_date:end_date]

    # get features and target variable
    X = (subset_df.index - subset_df.index.min()).days.values.reshape(-1, 1)  # use days since the minimum date
    y = subset_df[feature].values.reshape(-1, 1)

    # do a linear regression
    model = LinearRegression()
    model.fit(X, y)
    slope = round(model.coef_[0][0], 4)
    print(f"Slope of the linear regression for {feature}: {slope}")

    # plot the original data
    plt.figure(figsize=(10, 6))
    plt.scatter(subset_df.index, subset_df[feature], label=feature)
    
    # plot the regression line
    plt.plot(subset_df.index, model.predict(X), color='red', label='Linear Regression')
    
    # set plot labels and title
    plt.xlabel('Date')
    plt.ylabel(feature)
    plt.title(f'{feature} Linear Regression from 2012-08-15 to 2012-09-15')
    plt.savefig(os.path.join('figures', f'{feature}_linear_regression.png'))
    plt.legend()
    plt.show()

def permuted_linear_regression_slopes(start_date, end_date, feature, num_permutations=1000):
    """
    Perform linear regression for a specified time span on permuted datasets
    and plot a histogram of the distribution of slopes.

    Parameters:
    - start_date: str, start date of the time span in the format 'YYYY-MM-DD'
    - end_date: str, end date of the time span in the format 'YYYY-MM-DD'
    - feature: str, the feature for which linear regression is to be performed
    - num_permutations: int, the number of permutations to perform

    Returns:
    - displays histogram of slope distribution
    """

    # eead the dataset
    df = pd.read_csv('./data_0_1/Both.csv')

    # convert 'day', 'month', and 'year' columns to datetime
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # set the date as the index and sort the DataFrame by the index
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # convert start_date and end_date to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # get data for the specified time span
    subset_df = df.loc[start_date:end_date]

    # get features and target variable
    X = (subset_df.index - subset_df.index.min()).days.values.reshape(-1, 1)  # Use days since the minimum date
    y = subset_df[feature].values

    # do a linear regression on the original dataset
    model = LinearRegression()
    model.fit(X, y)
    original_slope = model.coef_[0]
    
    # perform permutation and store slopes in a list
    permutation_slopes = []

    for _ in range(num_permutations):

        # dandomly shuffle the target variable (feature)
        shuffled_y = np.random.permutation(y)

        # perform linear regression on the permuted dataset
        model.fit(X, shuffled_y)
        permutation_slope = model.coef_[0]
        permutation_slopes.append(permutation_slope)

    # plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(permutation_slopes, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(original_slope, color='red', linestyle='dashed', linewidth=2, label='Original Slope')
    plt.xlabel('Slope')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Slopes for {feature} from {start_date} to {end_date}')
    plt.legend()
    plt.savefig(os.path.join('figures', f'{feature}_slopes_distribution.png'))
    plt.show()



plot_original() 
linear_regression_plot('2012-08-15', '2012-09-15', 'Temperature')
permuted_linear_regression_slopes('2012-08-15', '2012-09-15', 'Temperature', num_permutations=1000)
