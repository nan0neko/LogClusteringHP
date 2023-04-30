import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# get the absolute path of the current file
file_path = os.path.abspath(__file__)

# extract the directory from the file path
dir_path = os.path.dirname(file_path)

# Specify the file name
file_name = 'logtable10.csv'

# Join the current directory with the file name to get the full file path
file_path = os.path.join(dir_path, file_name)

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Create a new column with a constant value
data['class'] = 'all'

# Create the parallel coordinates plot
parallel_coordinates(data, 'class')

plt.xticks(rotation='vertical', fontsize='xx-small')

# Show the plot
plt.show()
