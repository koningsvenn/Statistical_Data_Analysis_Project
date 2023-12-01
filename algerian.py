import pandas as pd

# Read the data from the file into a DataFrame
bejaia = pd.read_csv('data/Bejaia.csv')
sidi = pd.read_csv('data/Sidi_Bel-abbes.csv')

# These files contain the following columns:
#   - day: the date of the observation
#   - month: the month of the observation
#   - year: the year of the observation
#   - temperature noon (temperature max) in Celsius degrees: 22 to 42
#   - RH: Relative Humidity in %: 21 to 90
#   - Ws: Wind speed in km/h: 6 to 29
#   - Rain: total day in mm: 0 to 16.8
#   - FFMC: Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5
#   - DMC: Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9
#          DCDrought Code (DC) index from the FWI system: 7 to 220.4

# Print the first five rows of the DataFrame
print(bejaia.head())
print("----")
print(sidi.head())













