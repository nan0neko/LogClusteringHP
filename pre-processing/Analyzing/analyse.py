from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import faiss
import re
import random
data=pd.read_csv("logtable.csv",names=['date','message'],sep=" - ")

df_y=data['date']

df_x=data['message']
tf=TfidfVectorizer(stop_words='english')
data_2=tf.fit_transform(df_x)
dim=data_2.shape[1]
n_bits=100
index = faiss.IndexLSH(dim,n_bits)
index.train(data_2.toarray())
index.add(data_2.toarray())
lis=[]
for deta in df_x:
    if deta not in lis:
        lis.append(deta)
print(len(lis))

data_3=tf.transform(lis)
print(data_3.shape)
D, I = index.search(data_3.toarray(), k=10)


for z,i in enumerate(I):
    
    with open('analyse.txt','a') as f: 
           
        dat_lis=[]
       
        for j in i:
            dat_lis.append(df_x[j])
            
            dat_sen='\n'.join(map(str,dat_lis ))
      
            
        f.write((f"actual log:\n{lis[z]}\n nearest logs: \n{dat_sen}\n\n\n\n"))
        