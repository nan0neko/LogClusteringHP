import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Read the CSV file into a DataFrame
data = pd.read_csv(r"C:\Users\toutu\Downloads\ifinfobase10.csv")

# Create a new column with a constant value
data['class'] = 'all'

# Create the parallel coordinates plot
parallel_coordinates(data, 'class')

plt.xticks(rotation='vertical', fontsize='xx-small')

# Show the plot
plt.show()