


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np





df=pd.read_csv('marketing_campaign.csv',sep='\t')




df.head()





df.shape





df.info()





df.isnull().sum()





mean_value=df['Income'].mean()
df['Income'].fillna(value=mean_value, inplace=True)





df['Marital_Status']=df['Marital_Status'].replace(['Absurd','Alone','YOLO'],['Single','Single','Single'])
df['Marital_Status']=df['Marital_Status'].replace(['Together'],['Married'])







df['Is_Parent'] = (df['Kidhome'] + df['Teenhome'] > 0).astype(int)




df['Age'] = 2023 - df['Year_Birth']




df['Amt_Spent'] = df['MntWines'] +df['MntFruits']+ df['MntMeatProducts']+ df['MntFishProducts']+df['MntSweetProducts']+ df['MntGoldProds']





df['Average_Spend_deals'] = (df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts','MntSweetProducts', 'MntGoldProds']].sum(axis=1)) / df['NumDealsPurchases']





df['Total_CampsAccepted'] = df[['AcceptedCmp1', 'AcceptedCmp2', 
                                     'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)





df['avg_web_visits'] = df['NumWebVisitsMonth'] / 12





df_drop = [ 'Dt_Customer','Z_CostContact', 'Z_Revenue', 'Year_Birth', 'ID','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain','Response']
df = df.drop(df_drop, axis=1)





from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score





df.drop(['Average_Spend_deals'],axis=1,inplace=True)




cluster=df.copy()





cat_col = ['Education','Marital_Status']





for i in cat_col:
    cluster[i] = cluster[i].astype('category')
    cluster[i] = cluster[i].cat.codes





cluster['avg_web_visits'] = cluster['avg_web_visits'].astype(int)





cluster['Income'] = cluster['Income'].astype(int)





scaler=StandardScaler()
df_S=pd.DataFrame(scaler.fit_transform(cluster),columns=cluster.columns)





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




kmeans=KMeans(n_clusters=2, random_state=1)
kmeans.fit(df_S)
 
cluster['cluster_no']=kmeans.predict(df_S) 
df['cluster']=kmeans.predict(df_S)





df.cluster.value_counts()





means=df.groupby('cluster').mean()




df_mean=pd.DataFrame(means)
df_mean.index=['cluster1','cluster2']
df_mean.T





cluster1=df.loc[df.cluster==0]
cluster2=df.loc[df.cluster==1]


# # Hierarchical Clustering




df_h=df.select_dtypes(np.number)





import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder




plt.figure(figsize=(10, 6))
plt.title("Customers Dendrogram")

clusters = shc.linkage(df_h, 
            method='ward', 
            metric="euclidean")
shc.dendrogram(Z=clusters)
plt.show()





hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(df_h) 
print(labels)





df_h['Clust'] = labels
encoder = LabelEncoder()
cols_encoded = pd.DataFrame(encoder.fit_transform(df_h['Clust']))
cols_encoded.index = df_h.index
cols_encoded.columns = ['Clust_enc']
df_h = pd.concat([df_h, cols_encoded], axis=1)
df_h = df_h.drop(columns = 'Clust')





df_h.Clust_enc.value_counts()





df_h_mean=pd.DataFrame(means)
df_h_mean.index=['clust1','clust2']
df_h_mean.T





clust1=df_h.loc[df.cluster==0]
clust2=df_h.loc[df.cluster==1]





total_amt_wines = cluster1['MntWines'].sum()
total_amt_meat  = cluster1['MntMeatProducts'].sum()
total_amt_fruits  = cluster1['MntFruits'].sum()
total_amt_fish  = cluster1['MntFishProducts'].sum()
total_amt_gold  = cluster1['MntGoldProds'].sum()
total_amt_sweet  = cluster1['MntSweetProducts'].sum()
total_revenue  =  cluster1['Amt_Spent'].sum()





vals = ['total_amt_wines','total_amt_meat','total_amt_fruits','total_amt_fish','total_amt_gold','total_amt_sweet']





print(total_amt_wines,total_amt_meat,total_amt_fruits,total_amt_fish,total_amt_gold,total_amt_sweet)




value = 263537
part_values = [145503, 52670, 9475, 13833, 32419, 9637]





labels = ['total_amt_wines','total_amt_meat','total_amt_fruits','total_amt_fish','total_amt_gold','total_amt_sweet']





from math import *
hue_values = [145503, 52670, 9475, 13833, 32419, 9637]
plt.figure(figsize=(6, 6))  # Adjust the figure size as desired
plt.pie(part_values, labels=labels, autopct='%1.1f%%', startangle = 90, 
        wedgeprops = {"edgecolor" : "black",
                      'linewidth': 2,
                      'antialiased': True})
plt.axis('equal')
plt.show()




total_amt_wines2 = cluster2['MntWines'].sum()
total_amt_meat2  = cluster2['MntMeatProducts'].sum()
total_amt_fruits2  = cluster2['MntFruits'].sum()
total_amt_fish2  = cluster2['MntFishProducts'].sum()
total_amt_gold2  = cluster2['MntGoldProds'].sum()
total_amt_sweet2  = cluster2['MntSweetProducts'].sum()
total_revenue2  =  cluster2['Amt_Spent'].sum()




print(total_amt_wines2,total_amt_meat2,total_amt_fruits2,total_amt_fish2,total_amt_gold2,total_amt_sweet2)





part_values2=[535101,316323,49406,70190,66160,50974]





labels2 = ['total_amt_wines2','total_amt_meat2','total_amt_fruits2','total_amt_fish2','total_amt_gold2','total_amt_sweet2']





plt.figure(figsize=(6, 6))  
plt.pie(part_values2, labels=labels2, autopct='%1.1f%%', startangle = 90, 
        wedgeprops = {"edgecolor" : "black",
                      'linewidth': 2,
                      'antialiased': True})
plt.axis('equal')
plt.show()







