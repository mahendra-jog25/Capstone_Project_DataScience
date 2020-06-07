# Capstone_Project_DataScience

## Predicting the Outcome of H-1B Visa Applications

### Introduction
In our project, we aim to predict the outcome of H-1B visa applications that are filed by many high-skilled foreign nationals every year. We framed the problem as a classification problem and applied supervised classification models like Naive Bayes, Logistic Regression, KNN and ensemble technique like Random Forest, Decision Tree, in order to output a predicted case status of the application.

### Limitations/challenges

1. The dataset was highly imbalanced and the reason for the model to not perform well enough was the imbalance nature of the dataset, given a balanced data would have performed really well.
2. Also, this imbalanced nature of dataset affected Cohen kappa score, F1 score in the models that we built to know the accuracy.
3. The highly imbalance nature of dataset could not be purely solved even by using techniques such as SMOTE (Oversampling, Under sampling)
4. The original dataset before any EDA, Feature Transformation or SMOTE is highly biased towards predicting only a particular value, for eg: in the case status predicting more 1’s (Certified) as compared to 0’s (Denied).
5. Also the data was provided was from a single financial year 2017.Getting more data from previous financial years would have provided us with rich quality of data to be dealt with.
6. There are a lot of features created due to dummification. Extensive H-1B visa domain knowledge is required for selecting the features and limiting the dummified values.
