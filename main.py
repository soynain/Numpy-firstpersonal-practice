import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

# Read the CSV file into a DataFrame
array_nmpy_data = pd.read_csv('C:\\Users\\moise\\Downloads\\archive\\car details v4.csv')

label_encoder=LabelEncoder()
# Remove rows with NaN values from both X and y simultaneously
array_nmpy_data = array_nmpy_data.dropna(subset=['Year', 'Kilometer', 'Price','Length','Width','Height','Seating Capacity','Fuel Tank Capacity'])

# Select the columns of interest for X and y
array_nmpy_data['Fuel Type'] = label_encoder.fit_transform(array_nmpy_data[['Fuel Type']].values.ravel())

array_nmpy_data['Transmission'] = label_encoder.fit_transform(array_nmpy_data[['Transmission']].values.ravel())

array_nmpy_data['Location'] = label_encoder.fit_transform(array_nmpy_data[['Location']].values.ravel())

array_nmpy_data['Color'] = label_encoder.fit_transform(array_nmpy_data[['Color']].values.ravel())

array_nmpy_data['Owner'] = label_encoder.fit_transform(array_nmpy_data[['Owner']].values.ravel())

array_nmpy_data['Seller Type'] = label_encoder.fit_transform(array_nmpy_data[['Seller Type']].values.ravel())

array_nmpy_data['Engine'] = label_encoder.fit_transform(array_nmpy_data[['Engine']].values.ravel())

array_nmpy_data['Max Power'] = label_encoder.fit_transform(array_nmpy_data[['Max Power']].values.ravel())

array_nmpy_data['Max Torque'] = label_encoder.fit_transform(array_nmpy_data[['Max Torque']].values.ravel())

array_nmpy_data['Make'] = label_encoder.fit_transform(array_nmpy_data[['Make']].values.ravel())


X=array_nmpy_data[['Make','Year', 'Kilometer','Length','Width','Height','Seating Capacity','Fuel Tank Capacity','Fuel Type','Transmission','Location','Color','Owner','Seller Type','Engine','Max Power','Max Torque']]

y=array_nmpy_data['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model=LinearRegression().fit(X_train,y_train)


""" 
y_predecir=model.predict(X_test)
mse = mean_squared_error(y_test, y_predecir)
print("\n\n")
print("\n\n")
print(f"MEDIA CUADRADA ERROR: {mse}") """
""" 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=LinearRegression().fit(X_train,y_train)


y_predecir=model.predict(X_test)
mse = mean_squared_error(y_test, y_predecir)
print("\n\n")
print("\n\n")
print(f"MEDIA CUADRADA ERROR: {mse}")
r2 = r2_score(y_test, y_predecir)
print(r2)

 """