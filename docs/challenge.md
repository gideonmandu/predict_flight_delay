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

I have a modified the model testing process to include the following:

```python
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'data.csv')
self.data = pd.read_csv(
    filepath_or_buffer=data_path.replace('tests/model/', ''),
    low_memory=False
)
```
This helps load the test data from the correct path, regardless of where the test is run from.
unlike the one below which is not flexible and will only work if the test is run from the root directory.

```python
self.data = pd.read_csv(filepath_or_buffer="../data/data.csv")       
```

I have also added more test cases to test other featured added to the model 
class.



# Modification to API testing

I have a modified the API testing process to include the following:

```python
with patch("xgboost.XGBClassifier.predict", return_value=np.array([0])):
```

This created a mock of the predict method of the XGBClassifier class and returns a numpy array of 0s. This is to ensure that the API test is not dependent on the model's prediction.

I also added other tests to test other features added to the API implementation.

# Deployment

Deployment was done using Google Cloud Run. The following steps were taken:
setting up a service account that serve the ci cd pipeline, creating a docker 
image, pushing the image to docker hub, and deploying the image to Cloud Run.

Due to cloud runs default security features the stress test experienced a 
restriction due to too many request from one IP address.