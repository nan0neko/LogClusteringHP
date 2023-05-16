import pandas as pd
import numpy as np
from annoy import AnnoyIndex
import re
import os
import gensim.downloader as api
import csv
import time
from memory_profiler import memory_usage
import psutil

#Records the starting time of the code execution.
start_time = time.time()

#Retrieves the absolute path of the current file
file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
file_name = 'logtable3.csv'

file_path = os.path.join(dir_path, file_name)
df = pd.read_csv(file_path, header=None)
df_sorted = df.sort_values(by=0, ascending=True)

# Removes numbers from the each line
def preprocess(line):
    tokens = re.findall('[^\d\W]+', line) 
    return ' '.join(tokens)

#Applies the preprocess function to each line in the first column of the sorted DataFrame, creating a list of preprocessed lines.
preprocessed_lines = [preprocess(line) for line in df_sorted[0]]
#Loads the GloVe word embedding model from the Gensim library.
embedding_model = api.load("glove-wiki-gigaword-100")
#Retrieves the dimensionality of the word vectors in the embedding model.
vector_size = embedding_model.vector_size
#Initializes a NumPy array of zeros with dimensions (number of preprocessed lines, vector size).
line_vectors = np.zeros((len(preprocessed_lines), vector_size))

# Iterates over the indices of the preprocessed lines.
for i in range(len(preprocessed_lines)):
    #Splits the preprocessed line into individual tokens.
    tokens = preprocessed_lines[i].split()
    for token in tokens:
        try:
            #Tries to retrieve the word vector for the token from the embedding model.
            vector = embedding_model[token]
            #Adds the word vector to the corresponding row of the line_vectors array.
            line_vectors[i] += vector
        except KeyError:
            pass

#Specifies the number of trees to build in the Annoy index.
num_trees = 10
#Specifies the size of each tree in the Annoy index.
tree_size = 2
#Creates an Annoy index with the specified vector size and angular distance metric.
annoy_index = AnnoyIndex(vector_size, metric='angular')

for i in range(len(preprocessed_lines)):
    vector = line_vectors[i]
    #Adds the vector to the Annoy index.
    annoy_index.add_item(i, vector)

#Builds the Annoy index with the specified number of trees.
annoy_index.build(num_trees)

num_lines = len(preprocessed_lines)
#Creates a set of unique preprocessed lines as queries.
queries = set(preprocessed_lines)
# Store the indices of queries and their neighbors that have been used
used_indices = set() 

for query in queries:
    #Retrieves the index of the query in the preprocessed lines.
    query_index = preprocessed_lines.index(query)

    # Skip if the query index or its neighbors have been used before
    if query_index in used_indices:
        continue
    #Adds the query index to the set of used indices.
    used_indices.add(query_index)

    num_neighbors = num_lines
    #Retrieves the indices and distances of the nearest neighbors of the query from the Annoy index.
    neighbor_indices, neighbor_distances = annoy_index.get_nns_by_item(query_index, num_neighbors, include_distances=True)

    #Generates the output file name based on the query index.
    output_file_name = 'output{}.csv'.format(query_index)
    #Joins the directory path and output file name to get the complete output file path.
    output_file_path = os.path.join(dir_path, output_file_name)
    #Opens the output file in write mode.
    with open(output_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['Neighbor Value'])
        # Add the query to the output file
        writer.writerow([preprocessed_lines[query_index]])  
        for i in range(num_neighbors):
            if neighbor_distances[i] == 0:
                neighbor_index = neighbor_indices[i]
                #Checks if the neighbor index has not been used before.
                if neighbor_index not in used_indices:
                    writer.writerow([preprocessed_lines[neighbor_index]])
                    used_indices.add(neighbor_index)
    
    #Retrieves the rows from the sorted DataFrame that match the query.
    df_query = df_sorted[df_sorted[0] == df_sorted[0].iloc[query_index]]

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {int(execution_time/60)} minutes and {int(execution_time%60)} seconds")

print(f'Physical Memory usage: {int(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)} MB')
print(f'Virtual Memory usage:  {int(psutil.Process(os.getpid()).memory_info().vms / 1024 ** 2)} MB')
