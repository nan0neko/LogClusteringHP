import pandas as pd
import numpy as np
from annoy import AnnoyIndex
import re
# gensim is a natural language processing library
import gensim.downloader as api

# Load the network log file into a list
log_file_path = r"C:\Users\toutu\Downloads\logtable.csv"
with open(log_file_path) as f:
    log_lines = f.readlines()

# Preprocess the data
# The re.findall() function is used to extract all alphanumeric tokens from the line, and these tokens are then joined together into a single string with spaces between each token.
def preprocess(line):
    tokens = re.findall('\w+', line)
    return ' '.join(tokens)

preprocessed_lines = [preprocess(line) for line in log_lines]

# This line loads a pre-trained word embedding model named "glove-wiki-gigaword-100" using the api.load() method from the gensim package. This word embedding model was trained on a large corpus of text and contains vector representations of words.
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
num_trees = 100
tree_size = 10
annoy_index = AnnoyIndex(vector_size, metric='euclidean')

# Add items to ANNOY index
for i in range(len(preprocessed_lines)):
    vector = line_vectors[i]
    annoy_index.add_item(i, vector)

# Build the ANNOY index
annoy_index.build(num_trees)

# Perform nearest neighbor searches
query_index = 1
num_neighbors = 10
neighbor_indices = annoy_index.get_nns_by_item(query_index, num_neighbors)

# Print out the results of the nearest neighbor searches
for i in range(num_neighbors):
    print(f"Neighbor {i+1}: Index = {neighbor_indices[i]}")
    print(log_lines[neighbor_indices[i]])
