import faiss
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
#readin the file and separating the date and the message part
data=pd.read_csv("logtable.csv",names=['date','message'],sep=" - ")
data_value=data.values
#date field 
df_y=data['date']
#message field
df_x=data['message']
uni=df_x.unique()
#initializing tfidf
tf=TfidfVectorizer()
#transforming the message using tfidf
data_2=tf.fit_transform(df_x).toarray()
data_3=tf.fit_transform(uni).toarray()
dim=data_2.shape[1]
print(dim)
m=35
n_bits=8
#using pq
index=faiss.IndexPQ(dim,m,n_bits)
index.train(data_2)
index.add(data_2)
data_4=data_3[0]
#finding all the neighbours of the any log type
D,I=index.search(data_4.reshape(-1,dim),data_2.shape[0])
print(len(I))
print(D)
#after finding the neighbours we group the logs with similar distance together
dic={}
for i in range(len(I)):
    for j in range(data_2.shape[0]):
        distance = D[i][j]
        item_id = I[i][j]
        if distance not in dic:
            dic[distance] = []
        dic[distance].append(item_id)
print(len(dic))

#finding the average cosine similiarity where for each file the first log is compared to all the other logs and for each file we find the average
simi=0
for op,l in dic.items():
    x=cosine_similarity( [data_2[l[0]]],data_2[l])
    y=x.mean()
    
    simi=simi+y
    print(y)

#code to group the given data present in the dictionary into different csv files  
for i in dic.keys():
    name=str(i)+'_clusternnnumber.csv'
    arra=[]
    numb=[]
    deta=[]
    for j in dic[i]:
        arra.append(df_y[j])
        numb.append(j)
        deta.append(df_x[j])
    dict={'log_no':numb,'date':arra,'time':deta}
    data_f=pd.DataFrame(dict)
    data_f.to_csv(name)        
     


    
