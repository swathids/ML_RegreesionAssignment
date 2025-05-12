## Preprocessing Steps and Justification
The dataset used in this analysis is the California Housing dataset, which includes various numerical features related to housing and a continuous target variable representing median house value. During preprocessing, the dataset was first loaded into a DataFrame, and basic exploratory steps such as df.info() and checking for missing values (df.isnull().sum()) were performed. This step ensures the dataset is clean and free of null entries, which is essential before feeding it into machine learning models. No missing values were detected, so no imputation was necessary. All features were already in numeric format, which is suitable for regression models. Additionally, feature-target separation (X, y) and splitting into training and testing sets help avoid overfitting and enable fair evaluation of the models. These preprocessing steps ensure data consistency, model readiness, and reliable evaluation.

## Regression Algorithm Implementation
1. Linear Regression:
Linear Regression models the relationship between independent variables and the target variable using a straight line. It assumes linearity, homoscedasticity, and no multicollinearity. It is suitable for this dataset because it provides a strong baseline and the features are already numeric and continuous, matching its assumptions.

2. Decision Tree Regressor:
This model splits the dataset into branches based on feature thresholds, forming a tree structure. It captures non-linear relationships and interactions between variables. It is well-suited for this dataset due to its ability to handle complex patterns without requiring feature scaling.

3. Random Forest Regressor:
An ensemble of decision trees, Random Forest reduces overfitting and improves generalization by averaging predictions from multiple trees. It is highly suitable for this dataset as it can manage both linear and non-linear trends and is less sensitive to noise.

4. Gradient Boosting Regressor:
This model builds trees sequentially, where each tree corrects the errors of the previous one. It is effective in reducing bias and variance. For this dataset, it is suitable because it often provides better accuracy on tabular datasets due to its strong optimization nature.

5. Support Vector Regressor (SVR):
SVR attempts to find a hyperplane that best fits the data within a certain margin. It works well for smaller datasets and can model non-linear relationships using kernels. It is appropriate for this dataset when properly tuned, although it is computationally more intensive.

## Model Evaluation and Comparison
Each regression model was evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared Score (R²). These metrics respectively quantify prediction error, average magnitude of error, and goodness-of-fit.

Best-Performing Algorithm: Random Forest Regressor consistently achieved the lowest MSE and MAE, and the highest R² score. Its ensemble nature and robustness to overfitting likely contributed to its superior performance on this structured dataset.

Worst-Performing Algorithm: SVR performed the worst, possibly due to its sensitivity to feature scales and its complexity in high-dimensional datasets like this one. Without careful tuning and scaling, SVR often underperforms compared to tree-based models.

