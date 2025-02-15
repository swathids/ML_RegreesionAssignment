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

Inference for Linear Regression Performance

Metric	Value
Mean Squared Error (MSE)	0.5389
Root Mean Squared Error (RMSE)	0.7341
Mean Absolute Error (MAE)	0.5353
R² Score	0.5888


 Here with R² = 0.5888, the model explains ~59% of the variance in house prices, but 41% remains unexplained.
This indicates missing features or non-linearity in the relationship between predictors and house prices.
Prediction Errors Are High

RMSE (0.7341) means the typical prediction error is around $73,000 (if house values are scaled in 100,000s).
MAE (0.5353) suggests the average absolute error is ~$53,000 per house, meaning many predictions are significantly off.
Linear Assumptions Are Limiting the Model

Housing prices are influenced by complex, non-linear interactions (e.g., neighborhood trends, income levels, and house age effects).
A simple linear relationship struggles to capture these dependencies, leading to suboptimal accuracy.
Conclusion 
 --Linear Regression is not the best choice for this dataset.
--Tree-based models like Gradient Boosting or XGBoost will likely perform better due to their ability to model non-linearity.
--Feature engineering (polynomial terms, interaction features) may help improve Linear Regression, but it will still be outperformed by ensemble methods.
