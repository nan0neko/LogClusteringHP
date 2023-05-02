import pandas as pd

df = pd.read_csv('LogClusteringHP-main\datasets\input.csv')
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Message'])
# X = vectorizer.fit_transform(df['Message']).toarray()

#create annoy index
from annoy import AnnoyIndex

n_trees = 10
n_dimensions = X.shape[1]

index = AnnoyIndex(n_dimensions, metric='angular')

for i in range(X.shape[0]):
    v = X[i].toarray().flatten()
    index.add_item(i, v)
# for i,v in enumerate(X):
#     index.add_item(i, v)

index.build(n_trees)
log_id = 3 #Index of the log to find nearest neighbor
n_neighbors = 10

# Get nearest neighbors
similar_ids = index.get_nns_by_item(log_id, n_neighbors)
similar_logs = df.iloc[similar_ids]

print(similar_logs)
