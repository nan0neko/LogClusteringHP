import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('usr.csv')
#print(data.shape)
print(len(data['cmd name'].unique()))
data.drop(['x', 'tag for log typ','usr','command'], axis=1, inplace=True)
data.info()
x=data
y=data['process and process id']
a=data['cmd name']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x['process and process id'] = le.fit_transform(x['process and process id'])
y = le.transform(y)
x['cmd name'] = le.fit_transform(x['cmd name'])
a = le.transform(a)
x.info()

cols=x.columns
from sklearn.preprocessing import MinMaxScaler
ms=MinMaxScaler()
x=ms.fit_transform(x)
x=pd.DataFrame(x,columns=[cols])
#print(x.head())

kmeans=KMeans(n_clusters=5,random_state=0)
kmeans.fit(x)
label=kmeans.fit_predict(x)
#print(label)
'''cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()'''
u_labels = np.unique(label)
filtered_label0 = x[label == 0]
for i in u_labels:
    plt.scatter(x.iloc[label == i , 0] , x.iloc[label == i , 1] , label = i)
plt.legend()
plt.show()