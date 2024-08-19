# Customer Churn Automation
#### © 2024 Sudarshan Zunja 

The objective of this project is to identify the plausible cause for Churn among customers using various analytical approaches. We first analyze the data and extract patterns and other metrics that 
affect the churn rate amongst customers.

The data used here is IBM's Telco Customer Churn Data. It boasts around 7043 customer details and their churn label.

## Workflow

- Data Collection and Preprocessing
- Exploratory Data Analysis
- Train a Classifier Model for classification
- Deploy Model as Streamlit App

## Preprocessing

- There are no NULL records, however some of the columns consist of NULL values such as Total Charges.
- Total Charges can be filled by assumig:  Total Charges = Contract * Monthly Charges

## [Exploratory Data Analysis](https://github.com/sudarshan710/Customer-Churn-Analytics/blob/main/analysis.ipynb)

Perform Exploratory Data Analysis based on available customer details. It includes features like gender, seniority, subscriptions, contract period, different services and their access data. Plot necessary plots 
and other visualizations to show how different aspect influence churn rate.

## [Classifier Model](https://github.com/sudarshan710/Customer-Churn-Analytics/blob/main/classifier.ipynb)

Train a classifier model to predict churn label for unseen data. Try different types of classifiers if necessary.

## HuggingFace Deployment - [Streamlit App](https://huggingface.co/spaces/Sudarshan710/churnAutomation)

Deploy the trained model using Streamlit framework on HuggingFace Spaces. 

![image](https://github.com/user-attachments/assets/16e39f35-ca9d-4e5c-92f4-54a3ec4d22e4)


© 2024 | Sudarshan Zunja
