import seaborn as sns
import pandas as pd
import os

# get the absolute path of the current file
file_path = os.path.abspath(__file__)

# extract the directory from the file path
dir_path = os.path.dirname(file_path)

# Specify the file name
file_name = 'ifinfobase10.csv'

# Join the current directory with the file name to get the full file path
file_path = os.path.join(dir_path, file_name)

# Load your data into a DataFrame
data = pd.read_csv(file_path)

# Create the heatmap
sns.heatmap(data)

# Show the plot
import matplotlib.pyplot as plt
plt.show()
