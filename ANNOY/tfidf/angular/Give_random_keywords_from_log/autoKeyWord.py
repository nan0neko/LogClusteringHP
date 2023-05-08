import psutil
import os
import time
start_time = time.time()
import numpy as np
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from annoy import AnnoyIndex

# Load log file into a DataFrame
log_df = pd.read_csv('datasets\input.csv')

# Choose a random log from the DataFrame
random_log = random.choice(log_df['Message'])

# Preprocess log file to generate numerical vectors
vectorizer = TfidfVectorizer()
log_text = log_df['Message'].values.tolist()
log_vectors = vectorizer.fit_transform(log_text).toarray()

# Create Annoy index
f = log_vectors.shape[1]  # Number of dimensions
index = AnnoyIndex(f, 'angular')  # Angular distance metric
n_trees = 10  # Number of trees in the index

# Add log vectors to index
for i, vec in enumerate(log_vectors):
    index.add_item(i, vec)

# Build the index
index.build(n_trees)

# Define function to extract top words from query text
def extract_top_words(query_text, n=5):
    # Preprocess query text using TF-IDF vectorizer
    query_vec = vectorizer.transform([query_text]).toarray()[0]
    # Extract top 5 words from query vector
    top_word_indices = query_vec.argsort()[::-1][:n]
    top_words = [vectorizer.get_feature_names_out()[i] for i in top_word_indices]
    return top_words
# Query the index for nearest neighbors
query_text = random_log
# Extract top words from query text
top_words = extract_top_words(query_text)
# Use top words to find nearest neighbors in Annoy index
query_vec = vectorizer.transform([" ".join(top_words)]).toarray()[0]
n_neighbors = 100  # Number of nearest neighbors to retrieve
nearest_neighbors = index.get_nns_by_vector(query_vec, n_neighbors)

# Retrieve log entries corresponding to nearest neighbors
result_df = log_df.iloc[nearest_neighbors]

# Print the random log and nearest neighbor logs
print("Random log:")
print(random_log)
print("Nearest neighbor logs:")
print(result_df)


# Save the result DataFrame to a CSV file
result_df.to_csv('ANNOY\\tfidf\\angular\Give_random_keywords_from_log\output.csv', index=True)

print(f'Physical Memory usage: {int(psutil.Process(os.getpid()).memory_info().rss / 1024 * 2)} MB') # in MB 
print(f'Virtual Memory usage: {int(psutil.Process(os.getpid()).memory_info().vms / 1024 * 2)} MB') # in MB

end_time = time.time()
print("Execution time = ",end_time-start_time, "seconds")


