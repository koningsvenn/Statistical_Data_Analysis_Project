
import pandas as pd
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_analysis(file_path):

    # load the data
    data = pd.read_csv(file_path)

    # preparing the data for the Probit model
    X = data['Temperature']  # independent variable
    y = data['Classes']      # dependent variable (binary: 0 = no fire, 1 = fire)

    # add a constant to the independent variable
    X_const = sm.add_constant(X)

    # fit the Probit model
    probit_model = Probit(y, X_const).fit()

    print(probit_model.summary())

    # setting up the range of temperature values for plotting
    temperature_range = np.linspace(data['Temperature'].min(), data['Temperature'].max(), 500)
    X_range = sm.add_constant(temperature_range)

    # getting the predicted probabilities from the Probit model
    predicted_probabilities = probit_model.predict(X_range)

    # plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Temperature', y='Classes', alpha=0.6, label='Observed Data')
    plt.plot(temperature_range, predicted_probabilities, color='red', label='Predicted Probability')
    plt.title('Probit Model: Probability of Forest Fire vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Probability of Forest Fire')
    plt.legend()
    plt.grid(True)
    plt.show()

file_path = './data_0_1/Both.csv'  # Replace with your actual file path
run_analysis(file_path)
