import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the file into a DataFrame
bejaia = pd.read_csv('data/Bejaia.csv')
sidi = pd.read_csv('data/Sidi_Bel-abbes.csv')

# These files contain the following columns:
#   - day: the date of the observation
#   - month: the month of the observation
#   - year: the year of the observation
#   - Temperature: (temperature at noon) in Celsius degrees: 22 to 42
#   - RH: Relative Humidity in %: 21 to 90
#   - Ws: Wind speed in km/h: 6 to 29
#   - Rain: total day in mm: 0 to 16.8
#   - FFMC: Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5
#   - DMC: Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9
#          DCDrought Code (DC) index from the FWI system: 7 to 220.4
#   - ISI: Initial Spread Index (ISI) index from the FWI system: 0 to 18.5
#   - BUI: Buildup Index (BUI) index from the FWI system: 1.1 to 68
#   - FWI: Fire Weather Index (FWI) Index: 0 to 31.1
#   - Classes: two classes, namely fire and not fire


print(list(bejaia.keys()))

# sperate the data of class: fire or no fire
classes = bejaia['Classes']
b_fire, b_no_fire = [], []
for i in range(len(classes)):
    if classes[i] == 'fire':
        b_fire.append(bejaia.iloc[i])
    else:
        b_no_fire.append(bejaia.iloc[i])

# sperate the data of class: fire or no fire
classes = sidi['Classes']
s_fire, s_no_fire = [], []
for i in range(len(classes)):
    if classes[i] == 'fire':
        s_fire.append(sidi.iloc[i])
    else:
        s_no_fire.append(sidi.iloc[i])

def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient between two lists
    """
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    x_std = (sum([(i - x_mean) ** 2 for i in x]) / len(x)) ** 0.5
    y_std = (sum([(i - y_mean) ** 2 for i in y]) / len(y)) ** 0.5
    return sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x))]) / (len(x) * x_std * y_std)\

def plot_correlation(b_fire, b_no_fire, s_fire, s_no_fire):
    """
    Plot the correlation between temperature and relative humidity for the two areas
    """
     # Calculate the correlation between temperature and relative humidity
    b_f_temp = [i['Temperature'] for i in b_fire]
    b_f_rh = [i['RH'] for i in b_fire]
    b_f_corr = pearson_correlation(b_f_temp, b_f_rh)
    print(f'Bejaia Fire: Pearson correlation coefficient between temperature and relative humidity ={b_f_corr.round(4)}')

    b_nf_temp = [i['Temperature'] for i in b_no_fire]
    b_nf_rh = [i['RH'] for i in b_no_fire]
    b_nf_corr = pearson_correlation(b_nf_temp, b_nf_rh)
    print(f'Bejaia No Fire: Pearson correlation coefficient between temperature and relative humidity ={b_nf_corr.round(4)}')

    s_f_temp = [i['Temperature'] for i in s_fire]
    s_f_rh = [i['RH'] for i in s_fire]
    s_f_corr = pearson_correlation(s_f_temp, s_f_rh)
    print(f'Sidi Fire: Pearson correlation coefficient between temperature and relative humidity ={s_f_corr.round(4)}')

    s_nf_temp = [i['Temperature'] for i in s_no_fire]
    s_nf_rh = [i['RH'] for i in s_no_fire]
    s_nf_corr = pearson_correlation(s_nf_temp, s_nf_rh)
    print(f'Sidi No Fire: Pearson correlation coefficient between temperature and relative humidity ={s_nf_corr.round(4)}')

    # calculate the regression line
    b_f_m, b_f_b = np.polyfit(b_f_temp, b_f_rh, 1)
    b_nf_m, b_nf_b = np.polyfit(b_nf_temp, b_nf_rh, 1)
    s_f_m, s_f_b = np.polyfit(s_f_temp, s_f_rh, 1)
    s_nf_m, s_nf_b = np.polyfit(s_nf_temp, s_nf_rh, 1)

    # Plot the correlation between temperature and relative humidity and the regression line
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.scatter(b_f_temp, b_f_rh, c='red', marker='o', label='fire')
    plt.scatter(b_nf_temp, b_nf_rh, c='blue', marker='x', label='no fire')
    plt.plot(b_f_temp, [b_f_m * i + b_f_b for i in b_f_temp], c='red', label='fire')
    plt.plot(b_nf_temp, [b_nf_m * i + b_nf_b for i in b_nf_temp], c='blue', label='no fire')
    plt.xlabel('Temperature')
    plt.ylabel('Relative Humidity')
    plt.title('Bejaia')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.scatter(s_f_temp, s_f_rh, c='red', marker='o', label='fire')
    plt.scatter(s_nf_temp, s_nf_rh, c='blue', marker='x', label='no fire')
    plt.plot(s_f_temp, [s_f_m * i + s_f_b for i in s_f_temp], c='red', label='fire')
    plt.plot(s_nf_temp, [s_nf_m * i + s_nf_b for i in s_nf_temp], c='blue', label='no fire')
    plt.xlabel('Temperature')
    plt.ylabel('Relative Humidity')
    plt.title('Sidi')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_correlation(b_fire, b_no_fire, s_fire, s_no_fire)

