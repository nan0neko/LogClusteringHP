import pandas as pd
import numpy as np
from annoy import AnnoyIndex
import re
import os
import gensim.downloader as api
import time
from memory_profiler import memory_usage
import psutil
import csv

# get the absolute path of the current file
file_path = os.path.abspath(__file__)

# extract the directory from the file path
dir_path = os.path.dirname(file_path)

# Specify the file name
file_name = 'logtable4.csv'

# Join the current directory with the file name to get the full file path
file_path = os.path.join(dir_path, file_name)

# Read the file into a pandas dataframe and sort it by the first column in ascending order
df = pd.read_csv(file_path, header=None)
df_sorted = df.sort_values(by=0, ascending=True)

# Preprocess the data
def preprocess(line):
    tokens = re.findall('\w+', line)
    return ' '.join(tokens)

preprocessed_lines = [preprocess(line) for line in df_sorted[0]]

# Load pre-trained word embedding model
embedding_model = api.load("glove-wiki-gigaword-100")

# Convert preprocessed log lines into a numeric vector
vector_size = embedding_model.vector_size
line_vectors = np.zeros((len(preprocessed_lines), vector_size))
for i in range(len(preprocessed_lines)):
    tokens = preprocessed_lines[i].split()
    for token in tokens:
        try:
            vector = embedding_model[token]
            line_vectors[i] += vector
        except KeyError:
            pass

# Set up ANNOY
num_trees = 10
tree_size = 2
annoy_index = AnnoyIndex(vector_size, metric='angular')

# Add items to ANNOY index
for i in range(len(preprocessed_lines)):
    vector = line_vectors[i]
    annoy_index.add_item(i, vector)

# Build the ANNOY index
annoy_index.build(num_trees)

# Perform nearest neighbor searches
num_lines = len(preprocessed_lines)
unique_queries = df_sorted[0].unique()
for query in unique_queries:
    query_index = np.where(df_sorted[0] == query)[0][0]
    neighbor_indices, neighbor_distances = annoy_index.get_nns_by_item(query_index, num_lines, include_distances=True)

    # Write result to a csv file
    output_file_name = 'output_{}.csv'.format(query_index)
    output_file_path = os.path.join(dir_path, output_file_name)
    with open(output_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['Distance', 'Value'])
        writer.writerow([0, preprocessed_lines[query_index]]) # Add the query to the output file
        for i in range(num_lines):
            if neighbor_distances[i] < 1:
                writer.writerow([neighbor_distances[i], preprocessed_lines[neighbor_indices[i]]])
        if len(neighbor_indices) == 0:
            writer.writerow([0, preprocessed_lines[query_index]]) # Add the query to the output file if it has no neighbors below distance 1
