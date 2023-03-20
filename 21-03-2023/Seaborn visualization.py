import seaborn as sns
import pandas as pd

# Load your data into a DataFrame
data = pd.read_csv(r"C:\Users\toutu\Downloads\ifinfobase10.csv")

# Create the heatmap
sns.heatmap(data)

# Show the plot
import matplotlib.pyplot as plt
plt.show()