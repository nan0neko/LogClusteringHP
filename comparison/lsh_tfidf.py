from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import faiss
import re
from sklearn.metrics.pairwise import cosine_similarity
import random
#reading the file and separating the date and the message part
data=pd.read_csv("logtable.csv",names=['date','message'],sep=" - ")
#date field 
df_y=data['date']
#message field
df_x=data['message']
#initializing Tfidf
tf=TfidfVectorizer(stop_words='english')
#transforming the message using tfidf
data_2=tf.fit_transform(df_x)

dim=data_2.shape[1]

data_2=data_2.toarray()
n_bits=8
#using lsh
index = faiss.IndexLSH(dim,n_bits)
index.train(data_2)
index.add(data_2)
#getting the bucket number of each of the item
arr=faiss.vector_to_array(index.codes)
#storing the data in terms of a dictionary where key is the bucket number and value is the items stored in that bucket
dic={}
for i,val in enumerate(arr):
    if val not in dic:
        dic[val] =[i]
    else:
        dic[val].append(i)
#finding the average cosine similiarity where for each file the first log is compared to all the other logs and for each file we find the average
simi=0
for op,l in dic.items():
    x=cosine_similarity( [data_2[l[0]]],data_2[l])
    y=x.mean()
    
    simi=simi+y
#average of all the files 
print(f"cosine similarity of lsh using tfidf :{simi/len(dic)}")   

#code to group the given data present in the dictionary into different csv files
# # for i in dic.keys():
# #     name=str(i)+'_bucketnumber1.csv'
# #     arra=[]
# #     numb=[]
# #     deta=[]
# #     for j in dic[i]:
# #         arra.append(df_y[j])
# #         numb.append(j)
# #         deta.append(df_x[j])
# #     dict={'log_no':numb,'date':arra,'time':deta}
# #     data_f=pd.DataFrame(dict)
# #     data_f.to_csv(name)
      


        
        
         

    








