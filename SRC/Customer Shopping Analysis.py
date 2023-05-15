#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np


# In[82]:


df=pd.read_csv('marketing_campaign.csv',sep='\t')


# In[83]:


df.head()


# In[84]:


df.shape


# In[85]:


df.info()


# In[86]:


df.isnull().sum()


# In[87]:


mean_value=df['Income'].mean()
df['Income'].fillna(value=mean_value, inplace=True)


# Since there are many categories which basically means same we all group them to a single category

# In[88]:


df['Marital_Status']=df['Marital_Status'].replace(['Absurd','Alone','YOLO'],['Single','Single','Single'])
df['Marital_Status']=df['Marital_Status'].replace(['Together'],['Married'])


# whether parent or not

# In[89]:


df['Is_Parent'] = (df['Kidhome'] + df['Teenhome'] > 0).astype(int)


# instead of DOB, age could be helpful to know the distribution

# In[90]:


df['Age'] = 2023 - df['Year_Birth']


# creating a column to know how much amount does customers spent on total  

# In[91]:


df['Amt_Spent'] = df['MntWines'] +df['MntFruits']+ df['MntMeatProducts']+ df['MntFishProducts']+df['MntSweetProducts']+ df['MntGoldProds']


# In[92]:


df['Average_Spend_deals'] = (df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts','MntSweetProducts', 'MntGoldProds']].sum(axis=1)) / df['NumDealsPurchases']


# In[93]:


df['Total_CampsAccepted'] = df[['AcceptedCmp1', 'AcceptedCmp2', 
                                     'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)


# Calculating avaerage vists to the company website

# In[94]:


df['avg_web_visits'] = df['NumWebVisitsMonth'] / 12


# In[95]:


df_drop = [ 'Dt_Customer','Z_CostContact', 'Z_Revenue', 'Year_Birth', 'ID','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain','Response']
df = df.drop(df_drop, axis=1)


# ## Machine learning

# In[96]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# In[97]:


df.drop(['Average_Spend_deals'],axis=1,inplace=True)


# In[98]:


cluster=df.copy()


# In[99]:


cat_col = ['Education','Marital_Status']


# In[100]:


for i in cat_col:
    cluster[i] = cluster[i].astype('category')
    cluster[i] = cluster[i].cat.codes


# In[101]:


cluster['avg_web_visits'] = cluster['avg_web_visits'].astype(int)


# In[102]:


cluster['Income'] = cluster['Income'].astype(int)


# In[103]:


scaler=StandardScaler()
df_S=pd.DataFrame(scaler.fit_transform(cluster),columns=cluster.columns)


# In[104]:


sum_sqr={}
for k in range(1,11):
    kmean=KMeans(n_clusters=k,random_state=1).fit(df_S)
    lables=kmean.predict(df_S)
    sum_sqr[k]=kmean.inertia_
plt.figure()
plt.plot(list(sum_sqr.keys()), list(sum_sqr.values()), 'bx-')
plt.xlable='clusters'
plt.ylable= 'sum_sqr'
plt.show()


# In[105]:


Shil_Score={}
for k in range(2,10):
    km=KMeans(n_clusters=k,random_state=1).fit(df_S)
    l=km.predict(df_S)
    Shil_Score[k]=silhouette_score(df_S,l)
plt.figure()
plt.plot(list(Shil_Score.keys()), list(Shil_Score.values()), 'bx-')
plt.xlable='no. of clusters'
plt.ylable= 'SSE'
plt.show()


# In[106]:


kmeans=KMeans(n_clusters=2, random_state=1)
kmeans.fit(df_S)
# to predict cluster for each entity of cluster dataset 
cluster['cluster_no']=kmeans.predict(df_S) 
# to predict cluster for each entity of original "data" dataset 
df['cluster']=kmeans.predict(df_S)


# In[107]:


df.cluster.value_counts()


# ## Cluster Profiling

# In[108]:


means=df.groupby('cluster').mean()


# In[109]:


df_mean=pd.DataFrame(means)
df_mean.index=['cluster1','cluster2']
df_mean.T


# In[110]:


cluster1=df.loc[df.cluster==0]
cluster2=df.loc[df.cluster==1]


# # Hierarchical Clustering

# In[111]:


df_h=df.select_dtypes(np.number)


# In[112]:


import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder


# In[113]:


plt.figure(figsize=(10, 6))
plt.title("Customers Dendrogram")

# Selecting Annual Income and Spending Scores by index
clusters = shc.linkage(df_h, 
            method='ward', 
            metric="euclidean")
shc.dendrogram(Z=clusters)
plt.show()


# In[114]:


hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(df_h) 
print(labels)


# In[115]:


df_h['Clust'] = labels
encoder = LabelEncoder()
cols_encoded = pd.DataFrame(encoder.fit_transform(df_h['Clust']))
cols_encoded.index = df_h.index
cols_encoded.columns = ['Clust_enc']
df_h = pd.concat([df_h, cols_encoded], axis=1)
df_h = df_h.drop(columns = 'Clust')


# In[116]:


df_h.Clust_enc.value_counts()


# In[117]:


df_h_mean=pd.DataFrame(means)
df_h_mean.index=['clust1','clust2']
df_h_mean.T


# In[118]:


clust1=df_h.loc[df.cluster==0]
clust2=df_h.loc[df.cluster==1]


# In[119]:


total_amt_wines = cluster1['MntWines'].sum()
total_amt_meat  = cluster1['MntMeatProducts'].sum()
total_amt_fruits  = cluster1['MntFruits'].sum()
total_amt_fish  = cluster1['MntFishProducts'].sum()
total_amt_gold  = cluster1['MntGoldProds'].sum()
total_amt_sweet  = cluster1['MntSweetProducts'].sum()
total_revenue  =  cluster1['Amt_Spent'].sum()


# In[120]:


vals = ['total_amt_wines','total_amt_meat','total_amt_fruits','total_amt_fish','total_amt_gold','total_amt_sweet']


# In[121]:


print(total_amt_wines,total_amt_meat,total_amt_fruits,total_amt_fish,total_amt_gold,total_amt_sweet)


# In[122]:


value = 263537
part_values = [145503, 52670, 9475, 13833, 32419, 9637]


# In[123]:


labels = ['total_amt_wines','total_amt_meat','total_amt_fruits','total_amt_fish','total_amt_gold','total_amt_sweet']


# In[124]:


from math import *
hue_values = [145503, 52670, 9475, 13833, 32419, 9637]
plt.figure(figsize=(6, 6))  # Adjust the figure size as desired
plt.pie(part_values, labels=labels, autopct='%1.1f%%', startangle = 90, 
        wedgeprops = {"edgecolor" : "black",
                      'linewidth': 2,
                      'antialiased': True})
plt.axis('equal')
plt.show()


# In[125]:


total_amt_wines2 = cluster2['MntWines'].sum()
total_amt_meat2  = cluster2['MntMeatProducts'].sum()
total_amt_fruits2  = cluster2['MntFruits'].sum()
total_amt_fish2  = cluster2['MntFishProducts'].sum()
total_amt_gold2  = cluster2['MntGoldProds'].sum()
total_amt_sweet2  = cluster2['MntSweetProducts'].sum()
total_revenue2  =  cluster2['Amt_Spent'].sum()


# In[126]:


print(total_amt_wines2,total_amt_meat2,total_amt_fruits2,total_amt_fish2,total_amt_gold2,total_amt_sweet2)


# In[127]:


part_values2=[535101,316323,49406,70190,66160,50974]


# In[128]:


labels2 = ['total_amt_wines2','total_amt_meat2','total_amt_fruits2','total_amt_fish2','total_amt_gold2','total_amt_sweet2']


# In[129]:


plt.figure(figsize=(6, 6))  # Adjust the figure size as desired
plt.pie(part_values2, labels=labels2, autopct='%1.1f%%', startangle = 90, 
        wedgeprops = {"edgecolor" : "black",
                      'linewidth': 2,
                      'antialiased': True})
plt.axis('equal')
plt.show()


# In[ ]:




