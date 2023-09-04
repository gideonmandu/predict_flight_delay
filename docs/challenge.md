# Model selection criteria
Based on the classification reports for models trained with the top 10 features and class balancing, we have:

### 1. XGBoost (with Feature Importance and with Balance):
- Precision for class 1 (delays): 0.25
- Recall for class 1 (delays): 0.69
- Accuracy: 0.55

### 2. Logistic Regression (with Feature Importance and with Balance):
- Precision for class 1 (delays): 0.25
- Recall for class 1 (delays): 0.69
- Accuracy: 0.55

Both models exhibit almost identical performance metrics. 

However, when deciding between the two, additional factors might be considered:

1. **Model Complexity**: XGBoost is a gradient boosting algorithm, which is typically more complex than logistic regression. If considering a simpler model that's easier to understand, maintain, and interpret, Logistic Regression would be preferable.
  
2. **Scalability**: If the dataset size increases substantially in the future, XGBoost might scale better and potentially provide improved performance over logistic regression. However, XGBoost would also typically require more computational resources.

3. **Feature Importance**: One advantage of XGBoost is its ability to provide feature importance, which can give insights into which features are most influential in making predictions. If understanding the significance of features is crucial, XGBoost might be the better choice.

4. **Flexibility and Future Tuning**: XGBoost has numerous hyperparameters that can be tuned, offering the potential for performance optimization in the future.

Given the identical performance metrics, the choice between XGBoost and Logistic Regression will largely depend on the above considerations and any specific requirements or constraints we might have.

I will be considering a combination of performance, interpretability, and potential for future optimization, **XGBoost with Feature Importance and with Balance** is slightly more favorable given my choose metrics choice.

# Modifications to model testing


# Modification to API testing