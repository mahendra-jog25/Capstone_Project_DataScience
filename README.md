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

### Conclusion 
1. In the end, it is indeed possible to predict the outcomes of H-1B visa applications based on the attributes of the applicant using machine learning. Out of the models we tried, Random Forest Classifier with best parameters obtained using GridSearchCV outperformed all the other models with 98% training accuracy and 86% test accuracy on the under sampled balanced test data.
2.  That’s likely because Random Forest Classifier are inherently better at explaining the complexities in the dataset. Overall, this model performed better after hyperparameter tuning and the predictive power of our model substantially increased as compared to the model built before hyperparameter tuning. This gave us a better F1 Score, Recall, Precision Score. The validation score that we obtained was a clear indication that overfitting problem of our model has been mitigated purely. This overfitting was caused due to the highly biased nature of the variables in our dataset. This cross validation score also was the best technique for assessing the effectiveness of the model built to accurately predict the outcome of H-1B visa applications.
3. If we had more time and computational resources, there are several directions we could take to improve our prediction algorithm. First of all, we would try Random Forest with Lasso (L1 Norm) regularization, since we believe that some features are actually irrelevant to the output as previously discussed.
4. Also, we could adjust the depth of the Random Forest, tune the hyperparameters a bit more precisely and possibly obtain the best model which would fairly predict the outcomes of H-1B visa applications and would give an optimal solution to the business problem favoured.
5. In addition, we could convert more features such as SOC NAME into one-hot-k representation to achieve better accuracy. Finally, we could create more informative features such as Standard Industrial Classification codes of the companies through web crawling instead of using the given EMPLOYER NAME and SOC NAME features directly.
