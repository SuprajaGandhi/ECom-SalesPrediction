
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima_model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

#Read the input files
working_dr=os.getcwd()
training_dataset = pd.read_csv(working_dr+'/Data/train.csv')
testing_dataset = pd.read_csv(working_dr+'/Data/test.csv')
store = pd.read_csv(working_dr+'/Data/stores.csv')
feature = pd.read_csv(working_dr+'/Data/features.csv')

training_dataset['Date'] = pd.to_datetime(training_dataset['Date'])
testing_dataset['Date'] = pd.to_datetime(testing_dataset['Date'])
feature['Date']=pd.to_datetime(feature['Date'])

# Splitting month, year and day - train
training_dataset['Month']=training_dataset['Date'].dt.month
training_dataset['Year']=training_dataset['Date'].dt.year
training_dataset['Dayofweek']=training_dataset['Date'].dt.dayofweek

# Splitting month, year and day - test
testing_dataset['Month']=testing_dataset['Date'].dt.month
testing_dataset['Year']=testing_dataset['Date'].dt.year
testing_dataset['Dayofweek']=testing_dataset['Date'].dt.dayofweek

# set the dates as the index of the dataframe, so that it can be treated as a time-series dataframe
training_dataset.set_index('Date',inplace=True)
testing_dataset.set_index('Date',inplace=True)


#Merge train and feature
merge_df=pd.merge(training_dataset,feature, on=['Store','Date','IsHoliday'], how='inner')
merge_df = pd.merge(merge_df, store, on='Store', how='inner')

def understand_data(stores, train, test, features):
    print("Getting Table info: ")
    print("Stores: ")
    print(stores.info())
    print("Training Dataset info: ")
    print(train.info())
    print("Test: ")
    print(test.info())
    print("Features: ")
    print(features.info())

def check_missing_values(stores,train, test, features):
    print("Printing missing values in the table")
    print("Stores:" + str(stores.isnull().sum()))
    print("Train : " + str(train.isnull().sum()))
    print("Test : " + str(test.isnull().sum()))
    print("Features : " + str(features.isnull().sum()))

def plot_correlation_matrix():
    #Considering only the numneric columns for plotting correlation matrix
    nc=merge_df.select_dtypes(include=[np.number]).columns
    correlationmatrix = merge_df[nc].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlationmatrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()
    #Observation: There is no linear relationship between dependant and non-dependant variables

def plot_monthly_sales_graph():
    m_sales = pd.pivot_table(training_dataset, values = "Weekly_Sales", columns = "Year", index = "Month")
    m = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.plot(m_sales)
    plt.title('Monthly Sales')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.xticks(range(1, 13), m)
    print(plt.show())

def rolling_average(train):
    sales = train.groupby('Date')['Weekly_Sales'].sum()
    months = [4, 6, 8, 12]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    for i, ax in enumerate(axes.flatten()):
        wsize = months[i]
        ax.plot(sales.index, sales, label='Original')
        ax.plot(sales.index, sales.rolling(window=wsize).mean(),
                label=str(wsize)+"-Months Rolling Mean")
        ax.set_title(str(wsize)+"-Months Moving Average")
        ax.set_xlabel("Years")
        ax.set_ylabel("Sales")
        ax.legend(loc='best')

    plt.tight_layout()
    plt.show()

def FeatureEngineering(merge_df):
    merge_df['Prev_Week_Sales'] = merge_df['Weekly_Sales'].shift(1)
    plt.figure(figsize=(8, 6))
    plt.plot(merge_df.index, merge_df['Weekly_Sales'], label='Current Week Sales', color='blue')
    plt.plot(merge_df.index, merge_df['Prev_Week_Sales'], label='Previous Week Sales', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.title('Current vs Lagged Weekly Sales')
    plt.show()

def plot_holiday_nonholiday_graph(merge_df):
    merge_df = merge_df.set_index('Date').sort_index()
    plt.figure(figsize=(8, 6))
    plt.plot(merge_df[merge_df['IsHoliday'] == 1]['Weekly_Sales'], marker='s', linestyle='-',
             color='black',label='Sales on Holidays')
    plt.plot(merge_df[merge_df['IsHoliday'] == 0]['Weekly_Sales'], marker='o', linestyle='--',
             color='cyan',label='Sales on Non-Holidays')

    plt.title('Comparing sales on Holidays vs Non-Holidays')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()


def PredictAndPlot(model, X_test, name):
    print("Making predictions")
    preds = model.predict(X_test)
    pred_df = pd.DataFrame({ 'Date': testing_dataset.index,'predicted_sales': preds })
    group_pred = pred_df.groupby('Date').mean()
    # Plot graph
    plt.figure(figsize=(8, 6))
    plt.plot(group_pred.index, group_pred['predicted_sales'],
             color='red',label='Predictions')
    plt.title(name + " - Sales Prediction")
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.show()

def KNeighboursRegression(Model,test):
    # Make predictions using the trained KNN model
    predications = Model.predict(test)
    predictions_df = pd.DataFrame({'Date': test.index, 'predicted_sales': predications})
    group_predictions = predictions_df.groupby('Date').mean()
    plt.figure(figsize=(8, 6))
    plt.plot(group_predictions.index, group_predictions['predicted_sales'],
             color='red', label='Predictions')
    plt.title("KNN Regression" + " - Sales Prediction")
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.show()

def CompareModels(data):
    features = ['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

    train_set=data[features + ['Weekly_Sales']]
    X_train, X_test, y_train, y_test = train_test_split(train_set[features], train_set['Weekly_Sales'], test_size=0.2,
                                                        random_state=42)

    # KNN Neighbours:
    # Initialize and train KNN regressor
    knn = KNeighborsRegressor(n_neighbors=1, weights='uniform')
    knn.fit(X_train, y_train)
    knn_acc = knn.score(X_test, y_test) * 100
    print("KNeigbhbors Regressor Accuracy - ", knn_acc)
    y_prediction = knn.predict(X_test)
    print("MAE", metrics.mean_absolute_error(y_test, y_prediction))
    print("MSE", metrics.mean_squared_error(y_test, y_prediction))
    print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))
    print("R2", metrics.explained_variance_score(y_test, y_prediction))

    plt.figure(figsize=(20, 8))
    plt.plot(knn.predict(X_test[:200]), label="prediction", linewidth=2.0, color='blue')
    plt.plot(y_test[:200].values, label="real_values", linewidth=2.0, color='lightcoral')
    plt.legend(loc="best")
    plt.title(" Actual vs. Predicted Sales- KNN")
    plt.show()

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # XGBoost
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)


    # Making predictions
    rf_preds = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, rf_preds)
    print("Random Forest - MAE :"+str(mae))
    rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    print("Random Forest - RMSE :"+str(rmse))
    accuracy = 1 - mae / np.mean(y_test)
    print("Random Forest - Accuracy : "+str(+accuracy*100)+"%")

    xgb_preds = xgb_model.predict(X_test)
    mae = mean_absolute_error(y_test, xgb_preds)
    print("XGBoost - MAE :" + str(mae))
    rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    print("XGBoost - RMSE :" + str(rmse))
    accuracy = 1 - mae / np.mean(y_test)
    print("XGBoost - Accuracy : " + str(accuracy*100)+"%")
    # Plotting results
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='Actual Sales', color='black')
    plt.plot(y_test.index, rf_preds, label='Random Forest Predictions', color='blue')
    plt.plot(y_test.index, xgb_preds, label='XG Boost Predictions', color='red')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.title("Sales Prediction Comparison")
    plt.legend()
    plt.show()



# Step 1: Understanding the data:
understand_data(store,training_dataset,testing_dataset,feature)
check_missing_values(store,training_dataset,testing_dataset,feature)

#Plot monthlty sales
plot_monthly_sales_graph()

#Plot Heat map for correlatopn
plot_correlation_matrix()

#Calling Rolling average method
rolling_average(training_dataset)

#Feature Engineering
FeatureEngineering(merge_df)

#Plot holiday and non holiday
plot_holiday_nonholiday_graph(merge_df)

y_train=training_dataset["Weekly_Sales"]
X_train=training_dataset.drop("Weekly_Sales", axis=1)

KN_model= KNeighborsRegressor(n_neighbors=1, weights='uniform')
KN_model.fit(X_train, y_train)
KNeighboursRegression(KN_model,testing_dataset)
# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and plot for Random Forest
PredictAndPlot(rf_model, testing_dataset, 'Random Forest')

# Train XGBoost model
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict and plot for XGBoost
PredictAndPlot(xgb_model, testing_dataset, 'XGBoost')

#Comparing models
print("Comparing models")

CompareModels(merge_df)




