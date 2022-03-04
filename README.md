# SampleSuperstore-EDA
Sample superstore dataset Exploration
<h2 align=center>Exploratory Data Analysis on very famous Sample Superstore Dataset</h2>


Link to data source: https://www.kaggle.com/aungpyaeap/supermarket-sales

**Context**

The growth of supermarkets in most populated cities are increasing and market competitions are also high. The dataset is one of the historical sales of supermarket company which has recorded in 3 different branches for 3 months data.

**Data Dictionary**

1. ***Invoice id:*** Computer generated sales slip invoice identification number

2. ***Branch:*** Branch of supercenter (3 branches are available identified by A, B and C).

3. ***City:*** Location of supercenters

4. ***Customer type:*** Type of customers, recorded by Members for customers using member card and Normal for without member card.

5. ***Gender:*** Gender type of customer

6. ***Product line:*** General item categorization groups - Electronic accessories, Fashion accessories, Food and beverages, Health and beauty, Home and lifestyle, Sports and travel

7. ***Unit price:*** Price of each product in USD

8. ***Quantity:*** Number of products purchased by customer

9. ***Tax:*** 5% tax fee for customer buying

10. ***Total:*** Total price including tax

11. ***Date:*** Date of purchase (Record available from January 2019 to March 2019)

12. ***Time:*** Purchase time (10am to 9pm)

13. ***Payment:*** Payment used by customer for purchase (3 methods are available – Cash, Credit card and Ewallet)

14. ***COGS:*** Cost of goods sold

15. ***Gross margin percentage:*** Gross margin percentage

16. ***Gross income:*** Gross income

17. ***Rating:*** Customer stratification rating on their overall shopping experience (On a scale of 1 to 10)

#### OK, lets start with Boilerplate stuff, import the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import the warnings.
import warnings
warnings.filterwarnings('ignore')

### Lets explore the data now, shall we?

##### Load the Dataset

df = pd.read_csv("supermarket_sales.csv")

##### Take a glimpse of the dataset


df.head()

 df.columns

##### Check Datatypes of columns

df.dtypes

##### Date column appears as object, check that column's format

df.Date

##### Change Date Column to DateTime format

df.Date=pd.to_datetime(df.Date)

##### Verify the changed format

df.Date

#### COOL, date format is better now...

##### OK, lets see the dataset again.

df.head()

##### Invoice Id is of no use for us, so lets make Date as our index column

#make a copy of dataset, in case something goes wrong
#Copying is not recommended for large datasets

df2= df.copy()
df.set_index("Date", inplace=True)

##### Lets see how our dataset looks now..

df.head()

##### Lets check some Statistics for our data

df.describe()

### Lets see if data cleaning is required

##### check for duplicated rows in the dataset

df.duplicated().sum()

df[df.duplicated()==True]

##### Lets remove the duplicated rows

df.drop_duplicates(inplace=True)

##### Check for duplicated row count again

df.duplicated().sum()

##### check columnwise Null values

df.isna().sum()

##### Graphically, null values can be seen using heatmap

sns.heatmap(df.isnull())

##### Filling null values

##### For numerical variables, we use the mean(choice may vary dependent on the spread of data)


df.fillna(df.mean(),inplace=True)

##### For categorical values, we use mode to fill the most frequest occurance

df.fillna(df.mode().iloc[0],inplace=True)

##### Lets visualize using heatmap again

sns.heatmap(df.isnull())

##### Lets start with taking a look at the branchwise transactions 

sns.countplot(df.Branch)

##### now, lets see the transaction methods used for sales

sns.countplot(df.Payment)

##### Lets check the distribution of customer ratings...

#plotting a hostogram
plt.hist(df.Rating)
plt.show()

##### Okay, that was a basic distribution, lets exlpore a detailed one..

#using distribution plot from seaborn library
sns.distplot(df.Rating)

#lets check some percentile values as well.

plt.axvline(x=np.mean(df.Rating), c="r", ls="-.",label="Mean")
plt.axvline(x=np.percentile(df.Rating,25),c="g",ls="--",label="25th,\n75th \npercentile")
plt.axvline(x=np.percentile(df.Rating,75),c="g",ls="--")
plt.legend()
plt.show()

##### Lets check distributions of all variables at once

df.hist(figsize=(12,10))
plt.show()

##### Seeing the distribution graphs, it seems some values are really correlated with each other

##### Lets check the correlation as well.

# Using pairplot for correlations
sns.pairplot(data=df2,vars=["Total","gross income","cogs","Tax 5%"])

#### Oh, all these colums are extremely correlated with each other.


##### A better way for checking correlation is using heatmaps

sns.heatmap(df.corr(),annot=True,cmap="BuGn")
plt.show()

### Okay, lets move further,

##### Lets check for a relationship between gross income and customer ratings

#sns.scatterplot(df.Rating,df["gross income"])
sns.regplot(df.Rating,df["gross income"])

sns.pairplot(data=df2,vars=["Rating","gross income"])

##### What can we say about branchwise gross income??

sns.boxplot(x=df.Branch,y=df["gross income"])

##### What about genderwise gross income?

sns.boxplot(x=df.Gender,y=df["gross income"])

##### Lets see how gross income varies over time..

sns.lineplot(x=df2.Date,y=df2["gross income"])

##### Okay, lets see correlation of all numeric variables

##### This May take time for large datasets

sns.pairplot(df2)

### Helpful Links

1. More visualizations: https://www.data-to-viz.com/
2. Seaborn gallery: https://seaborn.pydata.org/examples/index.html
3. Pandas profiling documentation: https://pypi.org/project/pandas-profiling/
