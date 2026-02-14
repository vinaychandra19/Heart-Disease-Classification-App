### a. Problem statement
To build an end-to-end machine learning classification pipeline to predict the likelihood of heart disease in patients based on clinical features, and deploy this as an interactive web application.

### b. Dataset description
The Heart Failure Prediction dataset contains 918 observations and 12 clinical features (including Age, Sex, Cholesterol, RestingBP, etc.). It is a binary classification problem where the target variable is `HeartDisease`.

### c. Models used
| Model               |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.8478 | 0.9008 |      0.9072 |   0.8224 | 0.8627 | 0.6971 |
| Decision Tree       |     0.788  | 0.7941 |      0.8617 |   0.757  | 0.806  | 0.5804 |
| KNN                 |     0.8478 | 0.9223 |      0.9072 |   0.8224 | 0.8627 | 0.6971 |
| Naive Bayes         |     0.8424 | 0.909  |      0.8824 |   0.8411 | 0.8612 | 0.6801 |
| Random Forest       |     0.8804 | 0.9419 |      0.8972 |   0.8972 | 0.8972 | 0.7543 |
| XGBoost             |     0.8696 | 0.9366 |      0.9192 |   0.8505 | 0.8835 | 0.7387 |

### Observations
| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Performed well as a baseline model with an accuracy of 84.78% and AUC of 0.90, indicating the dataset has a reasonable degree of linear separability. |
| Decision Tree | This was the weakest model with the lowest accuracy (78.8%) and AUC (0.79), suggesting it likely overfitted to the training data and failed to generalize as well as the ensemble methods. |
| KNN | Achieved the same accuracy as Logistic Regression (84.78%) but with a higher AUC (0.92), showing it was effective at capturing local patterns in the feature space. |
| Naive Bayes | Delivered consistent performance (AUC 0.91) despite the strong assumption of feature independence, proving robust for this clinical dataset. |
| Random Forest (Ensemble) | The Best Performer. It achieved the highest Accuracy (88.04%) and AUC (0.94). The ensemble approach effectively reduced variance, handling non-linear relationships better than single decision trees. |
| XGBoost (Ensemble) | Very competitive with Random Forest (Accuracy 86.96%, AUC 0.94), demonstrating strong predictive power, though slightly lower recall in this specific test split compared to Random Forest. |