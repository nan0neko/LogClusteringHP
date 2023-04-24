import pandas as pd
import numpy as np
import matplotlib.pyplot as mlp
from sklearn.cluster import KMeans,DBSCAN,MeanShift,AgglomerativeClustering  
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,davies_bouldin_score
le=LabelEncoder()

data=pd.read_csv("ifinfo.csv")

data['date']=le.fit_transform(data['date'])
data['time']=le.fit_transform(data['time'])
#data['ifnfo']=le.fit_transform(data['ifnfo'])
#data['ifinfo_name']=le.fit_transform(data['ifinfo_name'])
#data['ifd']=le.fit_transform(data['ifd'])
data['type']=le.fit_transform(data['type'])
data.drop('software/module',axis=1,inplace=True)
data.drop('type_log',axis=1,inplace=True)
data.drop('Interface_Descriptor',axis=1,inplace=True)
deta=data.iloc[:,:]
scal=StandardScaler()
scaled=scal.fit_transform(deta)
pca=PCA(n_components=2)
fit=pca.fit_transform(scaled)
fit=pd.DataFrame(fit)
fit.columns=['p1','p2']

label=deta["means"]=KMeans(n_clusters=2).fit_predict(fit)
print(silhouette_score(fit,deta['means']),davies_bouldin_score(fit,deta['means']))
"""deta["dbs"]=DBSCAN().fit_predict(fit)
deta["hi"]=MeanShift().fit_predict(fit)
deta["agg"]=AgglomerativeClustering().fit_predict(fit)
print(silhouette_score(fit,deta['means']),davies_bouldin_score(fit,deta['means']))
print(silhouette_score(fit,deta['dbs']),davies_bouldin_score(fit,deta['dbs']))
print(silhouette_score(fit,deta['hi']),davies_bouldin_score(fit,deta['hi']))
print(silhouette_score(fit,deta['agg']),davies_bouldin_score(fit,deta['agg']))
"""




#scal=StandardScaler()
#scaled=scal.fit_transform(data)

#pca=PCA(n_components=2)
"""fit=pca.fit_transform(scaled)
fit=pd.DataFrame(fit)
fit.columns=['p1','p2']

data["kmeans"]=KMeans(n_clusters=3).fit_predict(fit)
label=KMeans(n_clusters=3).fit_predict(fit)
#Getting unique labels
 """
u_labels = np.unique(label)
filtered_label0 = fit[label == 0]
for i in u_labels:
    mlp.scatter(fit.iloc[label == i , 0] , fit.iloc[label == i , 1] , label = i)
mlp.legend()
mlp.show()

 

   









