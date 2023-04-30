import pandas as pd
import numpy as np
from annoy import AnnoyIndex
import re
import os
# gensim is a natural language processing library
import gensim.downloader as api
import time
from memory_profiler import memory_usage
import psutil
import csv
import os

# Define a function to measure memory usage
def mem_usage():
    mem = memory_usage()[0]
    return mem

start_time = time.time()
# Measure memory usage before running the code
start_mem = mem_usage()


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
query_index = 9026
num_neighbors = num_lines
neighbor_indices, neighbor_distances = annoy_index.get_nns_by_item(query_index, num_neighbors, include_distances=True)

#Save the result into a text file
output_file_name = 'hamming_textfile_embedding_model.txt'
output_file_path = os.path.join(dir_path, output_file_name)
with open(output_file_path, 'w') as output_file:
    for i in range(num_neighbors):
        output_file.write(f"Neighbor {i+1}: Index = {neighbor_indices[i]}, Distance = {neighbor_distances[i]:.2f}\n")
        output_file.write(df_sorted.iloc[neighbor_indices[i], :][0] + '\n')

#Save the result into a csv file
output_file_name = 'hamming_csvfile_embedding_model.csv'
output_file_path = os.path.join(dir_path, output_file_name)
with open(output_file_path, 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(['Distance', 'Neighbor Value'])
    for i in range(num_neighbors):
        writer.writerow([neighbor_distances[i], preprocessed_lines[neighbor_indices[i]]])


file_name = 'hamming_csvfile_embedding_model.csv'

# Join the current directory with the file name to get the full file path
file_path = os.path.join(dir_path, file_name)

# Open the input file and read the data
with open(file_path, 'r') as input_file:
    reader = csv.reader(input_file)
    input_data = list(reader)

# Create a dictionary to keep track of the unique values and their counts
unique_values = {}

# Loop through the input data, adding unique values to the dictionary and incrementing their count
for row in input_data:
    value = tuple(row[:2])  # Consider both columns as a single value
    if value in unique_values:
        unique_values[value] += 1
    else:
        unique_values[value] = 1

# Join the current directory with the output file name to get the full file path
output_file_path = os.path.join(dir_path, 'unique_hamming_csvfile_embedding_model.csv')

# Open the output file and write the unique values and their counts
with open(output_file_path, 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    for value, count in unique_values.items():
        writer.writerow(list(value) + [count])


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {int(execution_time/60)} minutes and {int(execution_time%60)} seconds")

# Measure memory usage after running the code
end_mem = mem_usage()

# Print the max memory usage
print(f"Max memory usage: {end_mem - start_mem} MB")
print(f'Physical Memory usage: {int(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)} MB')
print(f'Virtual Memory usage:  {int(psutil.Process(os.getpid()).memory_info().vms / 1024 ** 2)} MB')
