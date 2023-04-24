from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np
import faiss
import re
from sklearn.metrics.pairwise import cosine_similarity
import random
data=pd.read_csv("logtable.csv",names=['date','message'],sep=" - ")

df_y=data['date']

df_x=data['message']
tf=TfidfVectorizer(stop_words='english')
data_2=tf.fit_transform(df_x)

dim=data_2.shape[1]


data_2=data_2.toarray()
n_bits=8
index = faiss.IndexLSH(dim,n_bits)
index.train(data_2)
index.add(data_2)
arr=faiss.vector_to_array(index.codes)

dic={}
for i,val in enumerate(arr):
    if val not in dic:
        dic[val] =[i]
    else:
        dic[val].append(i)
   


for i in dic.keys():
    name=str(i)+'_bucketnumber1.csv'
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
      


        
        
         

    








