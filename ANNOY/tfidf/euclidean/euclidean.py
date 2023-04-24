from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
data=pd.read_csv("logtable.csv",names=['date','message'],sep=" - ")
#date field 
df_y=data['date']

#message field
df_x=data['message']
uni=df_x.unique()
tf=TfidfVectorizer(stop_words='english')
data_3=tf.fit_transform(uni).toarray()
#transforming the message using tfidf
data_2=tf.fit_transform(df_x).toarray()
dim=data_2.shape[1]

index=AnnoyIndex(dim,metric='euclidean')
for i,val in enumerate(data_2):
    index.add_item(i,val)
index.build( n_trees=100)
index.save("tree.ann")
u=AnnoyIndex(dim,metric='euclidean')
u.load("tree.ann")

print(data.iloc[u.get_nns_by_vector(data_3[141], 12)]) 
print(uni[11])
# num_neighbors = len(data_2)# Retrieve all items in dataset
# for dam in data_3:
#     neighbor_indices = index.get_nns_by_vector(dam,2)
#     similar_items = data.iloc[neighbor_indices]
#     print(similar_items)
