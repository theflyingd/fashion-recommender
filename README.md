
# Fashion recommernder capstoneproject

## Disclaimer: 

!!! This notebook is under construction. Cleaning of the Notebooks, Documentation and Descriptions are in process. Thank you for your understanding. !!!!

## Introduction

Online retailers for fashion products suffer the paradox of choice: With increasing number of articles/choices, customers tend to make less or wrong purchases. This results in less customer satisfaction, more returns and less turnover.

This capstone project deals with the challenging task of creating a recommendation system for a fashion label. This increases customer satisfaction and the sustainability of the sales process, and a targeted recommendation generates significant economic value. In addition, a recommendation system provides valuable insights into customer preferences and forecasting opportunities for stakeholders.
The task is to read customers’ minds without evaluation information based solely on customers’ purchase decisions using cleaver feature engineering and targeted models. An in-depth error analysis was conducted to select an appropriate model, and the winning model was further trained on the proper metric.
In addition to the challenging nature of a recommender system, several additional obstacles were overcome. Starting from a fast adaptation to an unfamiliar subject area, dealing with Big-Data, and targeted feature engineering to acquaint with the yet foreign methodology.
All these efforts were made because “you have the right to look fabulous.”

## Data given

The data was provided by [H&M](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/) by their fashion recommendations Kaggel challenge. 

The dataset contains 3 Tables : 

* article meta data
    * 105,542 articles and 25 features
* customer data
    * 1,371,980 customers and 7 features
* transactions table
    * 31,788,324 transactions and 5 features
        20.09.2018 - 22.09.2020

### Transaction table :

The main table was the transactions table, which connects the article table with the customer table. There every single purchased item is present. An overview is given in the following:

|Column name| Datatyp| Meaning | nunique| annotation|
|----------|-------| ---- |----|---|
|tdat |	object |	Transaction date|	734 | needs to be transformed|
|customerid |	object |	Custumer ID|	1362281| primary key|
|articleid |	int64|	article_id |	104547| secondary key|
|price |	float64 |	price | 	9857| is transformed or different currency|
|saleschannelid |	int64|	2 is online and 1 store|	2| encoding to 0 &1 ?|


### Article Metadata

The article table is describing the 105.542 articles by 25 features in more detail these should be shown in the following: 

|Column name| Datatyp| Meaning | nunique| annotation|
|----------|-------| ---- |----|---|
| [article_id](#articleid) 	|int64 | Id connected to image name| 105542| |
| [product_code](#productcode)	|int64| Overlying product category| 45875| Takes the first 7 digits of the article ID|
| [prod_name](#prodname)	|object |  overall product name | 47224| General product name|
|[product_type_no](#producttypeno)	|int64 | Classificaton after product type number| 132| -1 = unknown = NaN ? = 121||
|[product_type_name](#producttypename)	|object |Classificaton product type label| 131| 131 values + -1 for unknown |
|[product_group_name](#productgroupname)	|object | Product typ categories by name|19| 121 unknown |
| [graphical_appearance_no](#graphicalappearanceno)	| int64| Appearance no| 30| -1 = unknown = NaN ? = 52|
|[graphical_appearance_name](#graphicalappearancename)	|object| Appearance label| 30| |
| [colour_group_code](#colourgroupcode)	|int64| (dominating)Colour group number|50| No closed numbers i.e. 18, 24-29, 34-39 etc. is missing|
| [colour_group_name](#colourgroupname)	|object| (dominating)Colour group label| 50| |
|[perceived_colour_value_id](#perceivedcolourvalueid)	|int64| "gray" scale or lightning condition of the image| 8 | -1 = unknown = NaN? = 28|
|[perceived_colour_value_name](#perceivedcolourvaluename)	|object| description of the lightning condition label| 8 | | |
|[perceived_colour_master_id](#perceivedcolourmasterid)	|int64|color master id | 20 | 17 is missing and -1 = unknown|
|[perceived_colour_master_name](#perceivedcolourmastername)	|object|  label of the color master id |20 | 
|[department_no](#departmentno)	|int64| Department number| 299| More than one department might be responsible for  a certain topic| 
|[department_name](#departmentname)	|object| Department label (topics) |250| |
|[index_code](#indexcode)	|object| "target group classification" index code| 10| A-F|
|[index_name](#indexname)	|object| "target group classification"  index label | 10 | Ladieswear, Menswear etc.| 
|[index_group_no](#indexgroupno)	|int64| Corse "target group classification"  |5| 1,2,3,4,26|
| [index_group_name](#indexgroupname)	|object| Corse "target group classification"  label | 5|  'Ladieswear' = 1 ,'Divided' =2,'Menswear'=3 'Baby/Children' = 4 , 'Sport' =26|
| [section_no](#sectionno)	|int64| "target group classification" + usage numbering| 57| not a closed numbering|
| [section_name](#sectionname)	|object| "target group classification" + usage label| 56| -1 is missing|
| [garment_group_no](#garmentgroupno)	|int64| garnet group no| 21|1002,  1003, 1007 ...1025| 
| [garment_group_name](#garmentgroupname)	|object|garnet group label|21 | | 
| [detail_desc](#detaildesc)	|object| more detailed description| 43404| so not every item as an individual description and 416 NaN values|


### Customer Table

The third table, is the Customer Table. It contains  1,371,980 customers and 7 features. and should be shown in the following. 

|Column name| Datatyp| Meaning | nunique| annotation|
|----------|-------| ---- |----|---|
|[customer_id](#customerid) |	object|	Custumer id|	1371980| hashed Customer Id|
|[FN](#fn)	|float64	|Recievs Fashion News|	1|
|[Active](#active) |	float64	|customer is active for communication|	1| Active = ACTIVE $\downarrow$|
|[club_member_status](#clubmemberstatus)	|object|	club status |	3| ACTIVE = Active ? $\uparrow$|
|[fashion_news_frequency](#fashionnewsfrequency)|	object	|Fashion News frequency	|4| interested in topics |
|[age](#age())	|float64|	age |	84| |
|[postal_code](#postalcode) |object|	Zip code |	352899| Hashed postalcode|

Further investiagtion to the basic data can be found in the notebook Columsdescription.ipynb. 


## Big Data handling

To deal with the big data a cloud service (```Google Cloud Platform)``` was used. The datasets were transformed to [```.parquet files```](https://parquet.apache.org/) ([pyarrow](https://arrow.apache.org/docs/python/index.html)) and loaded into ```Google cloud storage```. In addition ```Bigquery``` was enabled as a Data warehouse, to be adressed with SQL-Queries. For paralizsation of the data wrangling and EDA, the Python library [```dask```](https://dask.org/) was used. 

## Explorative Data Analysis

## Models

¡[](./images/Models.jpg)
