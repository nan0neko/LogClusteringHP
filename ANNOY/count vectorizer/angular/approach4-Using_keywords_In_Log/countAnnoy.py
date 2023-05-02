import annoy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import psutil
import os

import time
start_time = time.time()

# Load log file into a DataFrame
log_df = pd.read_csv('LogClusteringHP-main\datasets\input.csv')

# Preprocess log file to generate numerical vectors
vectorizer = CountVectorizer()
log_text = log_df['Message'].values.tolist()
log_vectors = vectorizer.fit_transform(log_text).toarray()

# Create Annoy index
f = log_vectors.shape[1]  # Number of dimensions
index = annoy.AnnoyIndex(f, 'angular')  # Angular distance metric
n_trees = 10  # Number of trees in the index

# Add log vectors to index
for i, vec in enumerate(log_vectors):
    index.add_item(i, vec)

# Build the index
index.build(n_trees)

# Query the index for nearest neighbors
# query_text = "peer in state Established"
query_text = "notification"
# query_text = "notification connection"
query_vec = vectorizer.transform([query_text]).toarray()[0]
n_neighbors = 10  # Number of nearest neighbors to retrieve
nearest_neighbors = index.get_nns_by_vector(query_vec, n_neighbors)


# Printing nearest neighbors
print(log_df.iloc[nearest_neighbors])


distances = []
for nn in nearest_neighbors:
    nn_vec = index.get_item_vector(nn)
    distance = np.linalg.norm(query_vec - nn_vec)  # Euclidean distance
    # Alternatively, use cosine distance:
    # distance = 1 - np.dot(query_vec, nn_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(nn_vec))
    distances.append(distance)

print(distances)

# Store the result in a DataFrame
result_df = log_df.iloc[nearest_neighbors]
# Save the result DataFrame to a CSV file
result_df.to_csv('ANNOY\count vectorizer\\angular\\approach4-Using_keywords_In_Log\output\output.csv', index=True)

print(f'Physical Memory usage: {int(psutil.Process(os.getpid()).memory_info().rss / 1024 * 2)} MB') # in MB 
print(f'Virtual Memory usage: {int(psutil.Process(os.getpid()).memory_info().vms / 1024 * 2)} MB') # in MB

end_time = time.time()
print("Execution time = ",end_time-start_time, "seconds")

# Plot the distances
plt.plot(np.arange(n_neighbors), distances)
plt.xlabel('Nearest Neighbor Index')
plt.ylabel('Distance')
plt.show()