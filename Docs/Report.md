# Customer Shopping Analysis
![image](https://github.com/Sukesh-bitragunta/Sukesh_Data606/assets/98784645/15734b0e-29d3-423a-b6a3-9095c09667d7)


## Presentation Link: https://youtu.be/14R9wM-9orM

## PPT link: https://github.com/Sukesh-bitragunta/Sukesh_Data606/blob/main/Docs/Customer%20Shopping%20Analaysis.pptx


## Introduction

Custer shopping analysis is used by every company and organization, this helps the company to changes its marketing strategy from every customer to customer whom would be willing to buy. Instead of paying  marketing every product of the company to  every customer in the company database and to wait whether the product would be successful or not. This will help the company a lot more, by performing this segregation companies may determine which segment of customers most likely to purchase the product and then market that product to those customers.

## Data 

To understand the shopping analysis of the consumers and segregate them into products, the dataset is very important. The dataset was taken from Kaggle, and the entries are from 2014, It has 2240 rows and 29 columns 

* ID: Customer's unique identifier
* Year_Birth: Customer's birth year
* Education: Customer's education level
* Marital_Status: Customer's marital status
* Income: Customer's yearly household income
* Kidhome: Number of children in customer's household
* Teenhome: Number of teenagers in customer's household
* Dt_Customer: Date of customer's enrollment with the company
* Recency: Number of days since customer's last purchase
* Complain: 1 if the customer complained in the last 2 years, 0 otherwise
* MntWines: Amount spent on wine in last 2 years
* MntFruits: Amount spent on fruits in last 2 years
* MntMeatProducts: Amount spent on meat in last 2 years
* MntFishProducts: Amount spent on fish in last 2 years
* MntSweetProducts: Amount spent on sweets in last 2 years
* MntGoldProds: Amount spent on gold in last 2 years
* NumDealsPurchases: Number of purchases made with a discount
* AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
* AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
* AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
* AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
* AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
* Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
* NumWebPurchases: Number of purchases made through the company’s website
* NumCatalogPurchases: Number of purchases made using a catalogue
* NumStorePurchases: Number of purchases made directly in stores
* NumWebVisitsMonth: Number of visits to company’s website in the last month

## Explorartary Data Analysis 

In the above list of columns, some of them are not useful for the analysis suchs as ID, Recency, complain and some other columns. Some features are added to the dataset such as age, ,by adding the columns of amount spent on the each product is calculated and created a column called total amount spent and instead of kepping with both kids and teenagers in the house created a column whether the customer is a parent or not. Outliers are also removed from the freatues and also created the columns with average store purchases and average web sales.

After performing the EDA, some of the graphs are plotted to find out the whether there are any relations between features and also any pattern with spending behaviour and their income and their social staus.

These are some plots 

![image](https://github.com/Sukesh-bitragunta/Sukesh_Data606/assets/98784645/e8549d39-c88f-40bc-a62c-90796e4fe88a)

![image](https://github.com/Sukesh-bitragunta/Sukesh_Data606/assets/98784645/68ea1854-5565-4244-9c34-e49adbd898d3)

![image](https://github.com/Sukesh-bitragunta/Sukesh_Data606/assets/98784645/1407d5cd-7800-4cfb-aae1-f04cd2dc7ebd)

![image](https://github.com/Sukesh-bitragunta/Sukesh_Data606/assets/98784645/008552f3-04c1-4118-aac1-918b2ef3d714)


The above graph indicates the age distribution of the customers, which we can tell there are more customers in the age group between 71 to 73. The other graph indicates the amount spent by the customers whether they are a parent or not and how much they spent. The scatter plot between amount spent on the fruits and amount spent of the meat have a linear relation. There is also a relation between number of the web purchase and number of times  the customer visted the company websites. There are customers who visted the website only once and odered more times from the website and there are customers who visited numerous times to website and still ddin't odered.


![image](https://github.com/Sukesh-bitragunta/Sukesh_Data606/assets/98784645/79d7a681-1de3-42fe-85c9-40daf295daef)

This scatter plot shows the realtion between 3 variables, x-axis represnt the amount spent on the wines, y-axis is amount spent on the sweeets and the size is the number of webvisits per month of each customer. The color spectrum repsents the income. The point that pointed out on the graph tells us that the, customer has income $110k per anum and spent $6 on wines and $262 on sweets and visted the company website atelast one time in a month.

 ## Machine learning
 
 Two models are used for clustering, they are k-means clustering and heirarchial clustering. K means clustering is used for grouping  the dataset, we need to define the number of centroids in the dataset. This centroid will allocate every data point to each cluster while keeping the centroids as soon as possible.  To perform the model every attribute should be a continuous variable. Education and martial status are the only variables that are categorical and transformed to continous. 
 
 The next step would be determining the value of k, There are two methods two perform this, one is silhouette score and plot are used to evaluate the quality of a clustering solution    produced by the k-means algorithm. The highest and closest point to one is considered the optimal number of clusters. The below graph shows us clearly 
 
 ![image](https://github.com/Sukesh-bitragunta/Sukesh_Data606/assets/98784645/9b659464-8fa3-4cf4-9382-6a00f55b842f)

 
The other one is  elbow method, In the Elbow method, we are actually varying the number of clusters (K) from 1 – 10. For each value of K, we are calculating Within-Cluster Sum of Square.

![image](https://github.com/Sukesh-bitragunta/Sukesh_Data606/assets/98784645/f9eff815-8f52-4132-b2d1-d26a9165f3c2)

Heirarchial Clustering  groups similar objects into a dendrogram. It merges similar clusters iteratively, starting with each data point as a separate cluster.
This creats a tree like structure that shows the relations between slusters and their heirarchay. To get the number of clusters for hierarchical clustering, we make use a concept      called   Dendrogram.

## Dendogram

![image](https://github.com/Sukesh-bitragunta/Sukesh_Data606/assets/98784645/7df25687-15f2-4b00-8755-b996c16daab0)

After performing the cluster analysis, The clsuter1 and cluster2 are resulted by performing the analysis of k-means and clust1 and clust2 are resuylted by heirarchial clustering. The table tells us clearly. 

![image](https://github.com/Sukesh-bitragunta/Sukesh_Data606/assets/98784645/9d44f17f-5eb7-4a96-9dce-c2e8be2ecc2a)

After performing cluster analysis on both the algorithms, the observations showed that the both algorithms are similarly clustered the dataset. They are almost indetical in some features.

## Conclusion

![image](https://github.com/Sukesh-bitragunta/Sukesh_Data606/assets/98784645/65f370bb-a3f9-48dd-9a7c-d7563f56ec19)

![image](https://github.com/Sukesh-bitragunta/Sukesh_Data606/assets/98784645/85f1e224-5e2e-4738-9e0d-78f1e7ed0fad)

![image](https://github.com/Sukesh-bitragunta/Sukesh_Data606/assets/98784645/e28723c7-6c74-426d-91f0-000e4e775813)
                      Amount spent on the each product in the clsuter1

![image](https://github.com/Sukesh-bitragunta/Sukesh_Data606/assets/98784645/8accc848-1217-4a22-b953-c838674cf6f9)
                      Amount spent on the each product in the cluster2


* On total the amount spent on cluster1 is around $260k and the amount spent on cluster1 is around million dollars. On both clusters the wines were more consumed by customers. 

* Sweets were least consumed goods and meat has equal contribution in both the clusters. 

* Also customers that are grouped in cluster1 have accepted more campaigns compared to cluster2 and In cluster0 we have more parents compared to cluster1.

As per the clustering, wines will be more reliable income for the company compared to other products. Wines are consumned by both the clustering groups and they occupy 50% income in both the clusters. This means both high income and low income customers favours wines. Cluster1 accepted more campigns this tells us they are loyal customers to the company and by making priority of these customers they can have strong base of customers whom will visit and purchase from the company regularly.



