# ML_RegreesionAssignment

Introduction
This assignment explores different Machine Learning regression models using the California housing dataset from Kaggle. The objective is to predict housing prices based on multiple features such as median income, house age, and population. The dataset undergoes preprocessing, feature scaling, and model evaluation using various performance metrics.

Preprocessing Steps
Loading the Dataset

The California Housing Dataset is fetched using fetch_california_housing() from sklearn.datasets.
It is converted into a pandas DataFrame with appropriate feature names.
The target variable (median house price) is added to the DataFrame.
Checking for Missing Values

Used .isnull().sum() to verify missing values.
No missing values were found in this dataset.
Splitting Features and Target

The dataset is split into features (X) and target (y).
Data is divided into training (80%) and testing (20%) sets using train_test_split().
Feature Scaling

StandardScaler is used to standardize feature values to mean = 0 and standard deviation = 1.
This prevents features with larger scales from dominating the learning process.
Scaling the Target Variable (y) for SVR

Since Support Vector Regression (SVR) is sensitive to scaling, y is also scaled using StandardScaler.
This ensures better numerical stability and improved performance of SVR.
Regression Models Implemented
The following regression models are applied:

Linear Regression
Support Vector Regression (SVR)
Random Forest Regressor
Decision Tree Regressor
Gradient Boosting Regressor
Each model is evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² score. Residual analysis and feature importance plots are also used to gain insights into model behavior.

Why Accuracy Scoring Is Not Used in Regression?
Unlike classification tasks, accuracy is not applicable in regression due to the following reasons:

Regression predicts continuous values rather than discrete categories, making it impossible to classify predictions as simply "correct" or "incorrect."
Predictions are rarely exact, but small differences (e.g., predicting $200,050 instead of $200,000) can still be useful.
Instead of accuracy, regression models use error-based metrics like:
MSE, RMSE, and MAE to measure prediction error.
R² Score to evaluate how well the model explains the variance in data.
Inferences from Model Performance
Linear Regression

Initially showed poor performance.
Scaling y improved results.
R² score indicates how well the model fits the data.
Support Vector Regression (SVR)

Performed better after feature scaling.
Works well for non-linear relationships.
Random Forest Regressor

Performed better than Linear Regression due to its ability to handle non-linearity.
Feature importance analysis shows which factors contribute most to predictions.
Decision Tree Regressor

Strong performance on training data but prone to overfitting on test data.
Overfitting is identified by a large difference in train and test R² scores.
Gradient Boosting Regressor

Helps reduce overfitting and improves performance.
Generally provides better results than Decision Trees alone.
Conclusion
This assignment demonstrates how different regression models handle housing price predictions. Tree-based models (Random Forest, Gradient Boosting) generally outperform Linear Regression due to their ability to model complex relationships. Feature scaling significantly improves the performance of SVR, and residual analysis helps assess model accuracy. The choice of the best model depends on the trade-off between interpretability and predictive power.

