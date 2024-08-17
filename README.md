# Predict Clicked Ads Customer Classification Using Machine Learning

Mini Project by Rakamin Academy

Wika Rabila Putri (DS 41 & JAP 28)

## Overview
A company in Indonesia wants to determine the effectiveness of the ads they run. This is important for the company to understand how well their ads are performing and how effectively they attract customers to view them. By analyzing historical advertisement data and uncovering insights and patterns, the company can better target their marketing efforts. The focus of this case is to build a machine learning classification model that aims to identify the right customer targets.

## Problem
The company wants to understand how effective their ads are in capturing customer attention. By understanding the characteristics of customers who are likely to click on ads, the company can optimize its marketing strategy to improve conversions.

## Goals

1. Identify characteristics and patterns of customers who are more likely to click on the ads shown by the company.
2. Develop a machine learning classification model to predict whether a customer will click on an ad based on historical data.
3. Provide recommendations to the company on more effective marketing targets based on analysis and predictive model results.

## Objective

1. Perform data preprocessing.
2. Conduct exploratory data analysis.
3. Identify and create new features that can enhance model performance.
4. Develop several machine learning classification models and perform hyperparameter tuning to optimize model performance.
5. Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC, and compare the performance of various models to select the best one.
6. Provide recommendations to the company.

## TASK 1: Customer Type And Behaviour Analysis On Advertisement

### Conversion Rate Analysis Based on Income, Spending and Age

![image](https://github.com/user-attachments/assets/19ca0464-2489-4d8d-93bb-117872e4420d)

![image](https://github.com/user-attachments/assets/a761bde4-4af6-44b4-b1c2-3f0ad8b83a1b)

- Total Spending vs Conversion Rate: There is no clear relationship between the amount spent and the frequency of purchases. Thus, spending decisions do not always result in more conversions.
- Conversion Rate vs Income: Ironically, the conversion rate tends to decrease with higher income. So, money does not always buy customer loyalty.
- Conversion Rate vs Age: There is no linear relationship between customer age and conversion rate. However, the data distribution shows a fairly even spread across age ranges, indicating that other factors are more influential in determining conversion rates.
- Total Spending vs Income: The higher the income, the greater the total spending. This means that more money allows for more spending.

![image](https://github.com/user-attachments/assets/b0c68df9-30ac-4308-a8c0-433ccc8bbad2)

- For the features Income, Total Spending, and Conversion Rate, there is no clear pattern or distinct age group.
- The age distribution tends to be fairly even and not significantly clustered. This suggests that, in this context, age is not a dominant factor affecting the relationship between income, spending, and conversion rate. Factors such as income and spending patterns have a more significant impact on determining customer conversion rates. However, it is still important to consider age in marketing strategies and conduct more in-depth analysis.

## TASK 2: Data Cleaning & Preprocessing

Data Overview: 2012 – 2014
- Total Data Points: 2,240
- Number of Columns/Features: 30
- Data Types: Includes object, int64, and float64
- Duplicate Data: None
-------------------------------------------------------------------------------------------------------------------------------------
- Missing Values in Income: There are 24 missing values in the Income feature, which will be handled by filling them with the median.
- Outliers Handling: Features tend to have outliers, which will be managed using the Interquartile Range (IQR) method.

1. Categorical Feature Encoding: Categorical features will be encoded to be processed by machine learning algorithms.
2. Feature Standardization: Features will be standardized to avoid bias in the model.

## TASK 3: Data Modeling

From the Distortion Score and Elbow Method, the optimal number of clusters is 4.

![image](https://github.com/user-attachments/assets/762881d1-4811-45ab-965d-72d953cc176a)

Clustering using the K-Means Algorithm:
The results from modeling and clustering the data using the K-Means algorithm show that the clusters formed clearly and effectively separate the data into distinct groups. This plot illustrates how the data can be well-differentiated into unique segments, reflecting the diversity of patterns and characteristics present.

![image](https://github.com/user-attachments/assets/0e87041f-78cf-414b-86af-ab46d095a1f3)

**Evaluation**

Based on the silhouette score, the optimal number of clusters appears to be either 2 clusters (0.542) or 3 clusters (0.484), as they provide the highest silhouette scores.

![image](https://github.com/user-attachments/assets/86afe4e3-b004-4b02-b483-98294a9c6c68)

## TASK 4: Customer Personality Analysis for Marketing Retargeting

![image](https://github.com/user-attachments/assets/124b5e22-3a1e-400a-a851-c8f99c5d2974)

**Cluster 0: Loyal Spenders**
- Key Characteristics: Loyal customers with high transaction frequency but moderate spending.
- Note: Good income and older age suggest brand loyalty.

**Cluster 1: Emerging Users**
- Key Characteristics: Young customers with low transaction frequency and low spending.
- Note: Price-sensitive and have a low conversion rate.

**Cluster 2: High Rollers**
- Key Characteristics: Customers with very high transaction frequency and spending.
- Note: Highest income and highly responsive to marketing campaigns.

**Cluster 3: Top Engagers**
- Key Characteristics: Customers with the highest transaction frequency and spending almost equal to Cluster 2.
- Note: High income but slightly lower conversion rate.

![image](https://github.com/user-attachments/assets/96e5cdad-34f9-4e17-8ccf-591118e58ac6)

- The majority of customers are in Cluster 1, the Emerging Users, who require a special strategy to increase loyalty and transactions.
- Cluster 2 and Cluster 3, although smaller, consist of customers with high spending and intense transaction activity, making them worthy of focus on retention and up-sell efforts.
- Cluster 0, while not as large as Cluster 1, shows significant potential in terms of loyalty, which can be further optimized.

![image](https://github.com/user-attachments/assets/736f7cb7-8eb5-4426-a6ea-e1576a0e3b42)

![image](https://github.com/user-attachments/assets/bea9e5c5-804f-43ca-899f-e76c2650df14)

![image](https://github.com/user-attachments/assets/6395658c-d687-490e-b5a6-bb3da5396a07)

- Cluster 0 customers have a relatively stable number of transactions and moderate spending. They tend to be older and have relatively high income.
- Cluster 1 consists of younger customers with low transaction frequency and spending. They frequently visit the website but are less responsive to campaigns.
- Cluster 2 has very high spending and a positive response to campaigns. These customers do not visit the website frequently but have a high conversion rate. Their income is also the highest among the clusters.
- Cluster 3 is very active with the highest transaction frequency and spending. The customers in this cluster tend to be older and have fairly high income.

![image](https://github.com/user-attachments/assets/f123abcb-0797-4482-8bf8-91c86ff6b877)

- In Cluster 0, the data points are scattered with spending ranging from low to moderate, and income consistently at a mid-to-high level.
- In Cluster 1, the data points are concentrated at the lower part of the plot, indicating low spending and relatively lower income.
- In Cluster 2, the data points are spread out in the upper-right part of the plot, indicating high spending and very high income.
- In Cluster 3, the data points are scattered with high spending and high income, indicating very active transaction activity.
- Income and Spending Polarization: There is a clear polarization between the clusters, with Cluster 2 and Cluster 3 showing higher income and spending compared to Cluster 0 and Cluster 1.
- Clear Market Segmentation: These clusters help identify clear market segmentation, allowing the company to target more specific and effective marketing campaigns.
- Tailored Strategies: The company can develop tailored strategies based on the characteristics and needs of each cluster. For example, Cluster 2 could be offered premium deals, while Cluster 1 could be provided with more in-depth education and promotions to enhance their engagement.

### Recommendation

**Enhance User Engagement**
- Recommendation: Increase user engagement with more interactive and engaging content. The "Daily Internet Usage" and "Daily Spent Time on Site" features highlight the importance of engagement.
- Action: Implement features such as personalized content, notifications, and interactive elements to encourage users to be more active on your site.

**Target High-Income Segments**
- Recommendation: Tailor advertisements for users in high-income areas, as "Area Income" influences ad clicks.
- Action: Use demographic data to target advertising campaigns at high-income segments with premium products or services.

**Personalize Ads Based on Age**
- Recommendation: Customize ads to the interests and needs of different age groups. The "Age" feature indicates the importance of personalization.
- Action: Create age-specific campaigns and ad content to increase relevance and engagement.

**Optimize Ad Placement and Timing**
- Recommendation: Determine the best times and places to display ads based on "Daily Internet Usage" and "Daily Spent Time on Site."
- Action: Analyze user activity patterns to schedule ads during peak times and place ads in strategic positions for maximum visibility.
