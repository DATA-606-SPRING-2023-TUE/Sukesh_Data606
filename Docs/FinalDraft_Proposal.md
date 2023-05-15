# Final Draft Propsal

# Customer Shopping Analysis 


## Introduction:
A comprehensive investigation of a company's ideal clients is known as customer shopping analysis. It allows a company to better understand its consumers and makes it simpler to change goods to meet the unique requirements, habits of various sorts of customers.Customer shopping analysis enables a company to change its product based on its customers from various customer categories. Instead of paying money to promote a new product to every consumer in the firm's database, a company may determine which customer group is most likely to purchase the product and then market the product just to that segment.


## Related Questions:

* How is the distribution of  customers in differents categories?
* Which model is best suitable for analysis?
* What products are most profitable?
* Which customer group would more like to repurchase from the company?

## Dataset

This dataset is taken from kaggle, it has 2240 rows and 29 columns. The source of the dataset https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

Some of these columns are unncessary for the analysis, such as ID, AcceptedCmp1,2,3,4,5 and other columns are removed. After removing the null values and unwated columns these attributes are I would like to focus on.

### Attributes
* Year_Birth: Customer's birth year
* Education: Customer's education level
* Marital_Status: Customer's marital status
* Income: Customer's yearly household income
* Kidhome: Number of children in customer's household
* Teenhome: Number of teenagers in customer's household
* MntWines: Amount spent on wine in last 2 years
* MntFruits: Amount spent on fruits in last 2 years
* MntMeatProducts: Amount spent on meat in last 2 years
* MntFishProducts: Amount spent on fish in last 2 years
* MntSweetProducts: Amount spent on sweets in last 2 years
* MntGoldProds: Amount spent on gold in last 2 years
* NumDealsPurchases: Number of purchases made with a discount
* NumWebPurchases: Number of purchases made through the company’s website
* NumCatalogPurchases: Number of purchases made using a catalogue
* NumStorePurchases: Number of purchases made directly in stores
* NumWebVisitsMonth: Number of visits to company’s website in the last month
 
I would like to add some more columns to the daatset such as total amount spent by the each customer and other columns that would be beneficial to analyze the dataset.

## ML Modeles:

I plan on using K-means culstering, Agglomerative hierarchial clustering and observe which algorthim would be able to segregate the customers and make useful the dataset to get a clear understanding of the customer shopping analysis.


