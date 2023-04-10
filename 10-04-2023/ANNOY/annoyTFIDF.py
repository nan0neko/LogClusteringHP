import annoy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load log file into a DataFrame
log_df = pd.read_csv('output2.csv')

# Preprocess log file to generate numerical vectors
vectorizer = TfidfVectorizer()
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
# query_text = "Error: failed to connect to server"
#query_vec = vectorizer.transform([query_text]).toarray()[0]
query_vec = log_vectors[2]
n_neighbors = 10  # Number of nearest neighbors to retrieve
nearest_neighbors = index.get_nns_by_vector(query_vec, n_neighbors)

# Printing nearest neighbors
print(log_df.iloc[nearest_neighbors])
# Retrieve log entries corresponding to nearest neighbors
# for idx in nearest_neighbors:
#     print(log_df.iloc[idx])
