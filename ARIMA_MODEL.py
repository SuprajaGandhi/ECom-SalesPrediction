import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.dates as mdates


def load_data(file_path):
    print("Loading data...")
    try:
    
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print("Error loading the dataset: {str(e)}")
        sys.exit(1)
def load_new_features(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print("Error loading the new features dataset: {str(e)}")
        sys.exit(1)
def load_stores_data(file_path):
    print("Loading stores data...")
    try:
        stores_data = pd.read_csv(file_path)
        return stores_data
    except Exception as e:
        print("Error loading the stores dataset: {str(e)}")
        sys.exit(1)

def preprocess_data(data):
    data.fillna(0, inplace=True)  # You can choose another strategy based on your data
    data['Date'] = pd.to_datetime(data['Date'])
    weekly_sales_mean = data['Weekly_Sales'].mean()
    weekly_sales_std = data['Weekly_Sales'].std()
    lower_bound = weekly_sales_mean - 3 * weekly_sales_std
    upper_bound = weekly_sales_mean + 3 * weekly_sales_std

    data['Weekly_Sales'] = np.where(data['Weekly_Sales'] < lower_bound, weekly_sales_mean, data['Weekly_Sales'])
    data['Weekly_Sales'] = np.where(data['Weekly_Sales'] > upper_bound, weekly_sales_mean, data['Weekly_Sales'])

    return data


def train_test_split_data(data):
    X = data.drop(columns=['Weekly_Sales', 'Date', 'IsHoliday_x', 'IsHoliday_y'])
    y = data['Weekly_Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def determine_arima_order(y_train):
    plot_acf(y_train, lags=20)
    plot_pacf(y_train, lags=20)
    plt.show()
    result = seasonal_decompose(y_train, model='additive', period=52)  # Assuming weekly seasonality
    result.plot()
    plt.show()



def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1, typ='levels')
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test.values, label='Actual')  
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.title('ARIMA Model Evaluation')
    plt.xlabel('Time')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.show()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("\nModel Evaluation:")
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    evaluation_results = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    }, index=y_test.index)


def train_arima_model(y_train , order_to_use ):
    order_to_use = (5, 1, 0)  
    model = ARIMA(y_train, order= order_to_use) 
    fitted_model = model.fit()
    print("ARIMA Model trained and saved.")
    return fitted_model


if __name__ == "__main__":
    training_data_path = "C:\\Users\\sahir\\OneDrive - The University of Colorado Denver\\Desktop\\SEM TWO\\ML\\PROJECT\\train.csv"
    new_features_path = "C:\\Users\\sahir\\OneDrive - The University of Colorado Denver\\Desktop\\SEM TWO\\ML\\PROJECT\\features.csv"  # Replace with the actual path
    print(f"Using training data from: {training_data_path}")
    print(f"Using new features data from: {new_features_path}")
    print("Loading and preprocessing data...")
    data = load_data(training_data_path)
    new_features = load_new_features(new_features_path)
    stores_data = load_stores_data("C:\\Users\\sahir\\OneDrive - The University of Colorado Denver\\Desktop\\SEM TWO\\ML\\PROJECT\\stores.csv")  # Replace with the actual path
    merged_data = pd.merge(data, new_features, on=['Store', 'Date'], how='inner')
    merged_data = pd.merge(merged_data, stores_data, on='Store', how='inner')
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    merged_data['DayOfWeek'] = merged_data['Date'].dt.dayofweek
    merged_data['Month'] = merged_data['Date'].dt.month
    merged_data['Day'] = merged_data['Date'].dt.day
    merged_data.fillna(0, inplace=True)
    merged_data['IsHoliday'] = merged_data['IsHoliday_x'] | merged_data['IsHoliday_y']

    merged_data = preprocess_data(merged_data)

    print("\nUpdated Dataset with Engineered Features:")
    print(merged_data.head())
    
    print("Time Series")

    X_train, X_test, y_train, y_test = train_test_split_data(merged_data)

    determine_arima_order(y_train)
    
    print(f"\nTrying ARIMA Order:")
    trained_model = train_arima_model(y_train, order_to_use =(5, 1, 0))
    evaluate_model(trained_model, X_test, y_test)
    y_pred = trained_model.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1, typ='levels')
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.title(f'ARIMA Model Evaluation ')
    plt.xlabel('Time')
    plt.ylabel('Weekly Sales')
    plt.legend()

    plt.show()
            
    print("Program completed successfully.")


test_data_path = "C:\\Users\\sahir\\OneDrive - The University of Colorado Denver\\Desktop\\SEM TWO\\ML\\PROJECT\\test.csv"
test_data = pd.read_csv(test_data_path)
test_data['Date'] = pd.to_datetime(test_data['Date'])
X_test = test_data[['Store', 'Dept', 'Date', 'IsHoliday']]
y_pred_test = trained_model.predict(start=len(y_train), end=len(y_train) + len(X_test) - 1, typ='levels')
y_pred_test.index = test_data.index  # Align the index
test_data['Predicted_Weekly_Sales'] = y_pred_test
test_data_2013 = test_data[test_data['Date'].dt.year == 2013]
test_data_2013_unique = test_data_2013.drop_duplicates(subset='Predicted_Weekly_Sales')
plt.figure(figsize=(10, 6))
plt.scatter(test_data_2013_unique['Date'], test_data_2013_unique['Predicted_Weekly_Sales'], label='Predicted Weekly Sales', marker='o', color='blue')
plt.title('Test Dataset with Unique Predicted Weekly Sales (Year 2013)')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
plt.gcf().autofmt_xdate()
plt.savefig("unique_predicted_weekly_sales_plot_2013.png")
plt.show()