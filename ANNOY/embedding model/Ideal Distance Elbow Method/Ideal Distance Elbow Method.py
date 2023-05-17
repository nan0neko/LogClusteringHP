import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np

# Read the CSV file
df = pd.read_csv('unique_angular_csvfile_embedding_model.csv')

# Extract the distance and neighbor value columns
distances = df['Distance']
neighbors = df['Neighbor Value']

# Calculate the distortion for different values of k
distortions = []
K = range(1, 10)  # Adjust the range as needed
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(distances.values.reshape(-1, 1))
    distortions.append(sum(np.min(cdist(distances.values.reshape(-1, 1), kmeans.cluster_centers_, 'euclidean'), axis=1)) / distances.shape[0])


# Find the ideal distance based on the elbow graph
ideal_distance = None
for i in range(1, len(distortions)-1):
    slope1 = distortions[i] - distortions[i-1]
    slope2 = distortions[i+1] - distortions[i]
    if slope1 > slope2:
        ideal_distance = distances[i]
        break

# Print the ideal distance
print("Ideal Distance:", ideal_distance)
