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
file_name = 'logtable3.csv'

# Join the current directory with the file name to get the full file path
file_path = os.path.join(dir_path, file_name)

# Read the file into a pandas dataframe and sort it by the first column in ascending order
df = pd.read_csv(file_path, header=None)
df_sorted = df.sort_values(by=0, ascending=True)

# Preprocess the data
# The re.findall() function is used to extract all alphanumeric tokens from the line, and these tokens are then joined together into a single string with spaces between each token.
# Converting log lines into word tokens
def preprocess(line):
    tokens = re.findall('\w+', line)
    return ' '.join(tokens)

preprocessed_lines = [preprocess(line) for line in df_sorted[0]]

# This line loads a pre-trained word embedding model named "glove-wiki-gigaword-100" using the api.load() method from the gensim package. 
# This word embedding model was trained on a large corpus of text and contains vector representations of words.
embedding_model = api.load("glove-wiki-gigaword-100")

# converts each preprocessed log line into a numeric vector by summing the vector representations of each word token in the line obtained from a pre-trained word embedding model
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
annoy_index = AnnoyIndex(vector_size, metric='hamming')

# Add items to ANNOY index
for i in range(len(preprocessed_lines)):
    vector = line_vectors[i]
    annoy_index.add_item(i, vector)

# Build the ANNOY index
annoy_index.build(num_trees)

# Perform nearest neighbor searches
num_lines = len(preprocessed_lines)
queries = set(preprocessed_lines)
for query in queries:
    query_index = preprocessed_lines.index(query)
    num_neighbors = num_lines
    neighbor_indices, neighbor_distances = annoy_index.get_nns_by_item(query_index, num_neighbors, include_distances=True)

    #Save the result into a csv file
    output_file_name = 'output{}.csv'.format(query_index)
    output_file_path = os.path.join(dir_path, output_file_name)
    with open(output_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['Neighbor Value'])
        writer.writerow([preprocessed_lines[query_index]]) # Add the query to the output file
        if 0 in neighbor_distances:
            for i in range(num_neighbors):
                if neighbor_distances[i] == 0:
                    writer.writerow([preprocessed_lines[neighbor_indices[i]]])

        
    # Remove query and neighbors from the data
    df_query = df_sorted[df_sorted[0] == df_sorted[0].iloc[query_index]]