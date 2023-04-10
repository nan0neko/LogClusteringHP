import pandas as pd
import numpy as np
from annoy import AnnoyIndex
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


# get the absolute path of the current file
file_path = os.path.abspath(__file__)

# extract the directory from the file path
dir_path = os.path.dirname(file_path)

# Specify the file name
file_name = 'logtable3.csv'

# Join the current directory with the file name to get the full file path
file_path = os.path.join(dir_path, file_name)

# Open the file and read the lines
with open(file_path) as f:
    log_lines = f.readlines()

# Preprocess the data
# The re.findall() function is used to extract all alphanumeric tokens from the line, and these tokens are then joined together into a single string with spaces between each token.
# Converting log lines into word tokens
def preprocess(line):
    tokens = re.findall('\w+', line)
    return ' '.join(tokens)

preprocessed_lines = [preprocess(line) for line in log_lines]

# Convert preprocessed log lines into a count vector representation
vectorizer = CountVectorizer()
line_vectors = vectorizer.fit_transform(preprocessed_lines)

# Set up ANNOY
num_trees = 10
tree_size = 2
# Annoy index only supports few metrics such as euclidean, cosine similarity(angular), dot, manhatten, hamming
annoy_index = AnnoyIndex(line_vectors.shape[1], metric='angular')

# Add items to ANNOY index
for i in range(len(preprocessed_lines)):
    vector = line_vectors[i].toarray()[0]
    annoy_index.add_item(i, vector)

# Build the ANNOY index
annoy_index.build(num_trees)

# Perform nearest neighbor searches
query_index = 1
num_neighbors = 100000
neighbor_indices, neighbor_distances = annoy_index.get_nns_by_item(query_index, num_neighbors, include_distances=True)

# Save the output to a file
output_file_name = 'angular.txt'
output_file_path = os.path.join(dir_path, output_file_name)
with open(output_file_path, 'w') as output_file:
    for i in range(num_neighbors):
        output_file.write(f"Neighbor {i+1}: Index = {neighbor_indices[i]}, Distance = {neighbor_distances[i]:.2f}\n")
        output_file.write(log_lines[neighbor_indices[i]])



plt.bar(range(num_neighbors), neighbor_distances)
plt.title("Distances to Nearest Neighbors")
plt.xlabel("Neighbor Index")
plt.ylabel("Distance")
plt.show()

plt.hist(neighbor_distances, bins=50)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('Histogram of Nearest Neighbor Distances')
plt.show()