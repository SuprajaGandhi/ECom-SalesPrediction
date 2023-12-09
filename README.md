# ECommerce Sales Prediction


**Problem Statement:**
To develop machine learning models for predicting sales in an e-commerce platform

**Data Exploratory Analysis:**
Analyze the trends on the weekly sales in different months, rolling average and identify the correlation between dependent and independent variables.

<img width="532" alt="image" src="https://github.com/SuprajaGandhi/ECom-SalesPrediction/assets/137209418/2959b4e8-7173-49cc-b663-a93b1f452f2d">

<img width="441" alt="image" src="https://github.com/SuprajaGandhi/ECom-SalesPrediction/assets/137209418/420a09a0-374d-4232-91cb-8a5a8f90e67e">

<img width="441" alt="image" src="https://github.com/SuprajaGandhi/ECom-SalesPrediction/assets/137209418/9b7b6420-5560-4f2a-872a-b22f81d59e4e">

**Feature Engineering:**
Lag features to capture the effect of past values on the target variable
![image](https://github.com/SuprajaGandhi/ECom-SalesPrediction/assets/137209418/bebe7d21-83a8-47c7-afb2-53d9706c1eb8)

**ML models:**

**KNN Regressor:** It predicts the continuous target variable by considering the average of the K-nearest neighbors' output values in the feature space.
Accuracy:89.59%

<img width="507" alt="image" src="https://github.com/SuprajaGandhi/ECom-SalesPrediction/assets/137209418/f4ec7638-a9ab-4709-b89c-258428bc5690">

**Random Forest:** Sturdy and adaptable, Random Forest is well-known for its robustness and adaptability when working with a wide range of data types.
Accuracy:88.06%

<img width="507" alt="image" src="https://github.com/SuprajaGandhi/ECom-SalesPrediction/assets/137209418/3bf0201a-5c31-4029-a40f-7c257843981a">

**XGBoost:** It employs gradient boosting, building trees sequentially to correct errors from previous iterations and focusing on challenging examples
Accuracy: 75.75%

<img width="507" alt="image" src="https://github.com/SuprajaGandhi/ECom-SalesPrediction/assets/137209418/f88a271c-7158-4e0e-853e-8353d85975ca">

**Arima:** This model combines moving average, differencing, and autoregression components. By using past data trends, it is possible to predict future values.

<img width="507" alt="image" src="https://github.com/SuprajaGandhi/ECom-SalesPrediction/assets/137209418/b8ff58fd-74d2-45ca-a1b2-98c5164631b6">

**Holiday Impact Analysis:**
There is a significant boost in holiday sales compared to non-holiday weeks. With the power of applying marketing and promotional strategies on holidays, companies can take advantage of consumer spending.

<img width="507" alt="image" src="https://github.com/SuprajaGandhi/ECom-SalesPrediction/assets/137209418/41cdfcdf-4d52-4521-b4c6-f52a749f292e">









