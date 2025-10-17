#IMPORTS
#--- Data --- 
import pandas as pd, time
import numpy as np

#--- Machine Learning ---
from sklearn.model_selection import train_test_split
#The model we're using
from sklearn.linear_model import LinearRegression
#To check performace
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#--- Visualisation ---
import matplotlib.pyplot as plt
import seaborn as sns

#--- Import our data ---
data = pd.read_feather("PulseBat.feather")

#--- Data Preprocessing and Aggregation ---
#ATTENTION! SORT HERE (BEFORE DROPPING THE OTHER COLUMNS)
#Only keep Numerical Columns
model_data = data[['Qn', 'Q', 'SOC', 'SOE'] + [f'U{i}' for i in range(1, 22)] + ['SOH']]

#Split the data into input (SOC, SOE, U1-U21) and output (SOH)
X = model_data[[f"U{i}" for i in range(1, 22)] + ["SOC", "SOE"]]
Y = model_data["SOH"]

#Split the data into Train and Test sets (80/20)
#ATTENTION! CURRENT "SORTING METHOD" IS RANDOM: "random_state=42"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#--- Training ---
model = LinearRegression()
model.fit(X_train, Y_train)

#--- Evaluation (test vs prediction) ---
Y_pred = model.predict(X_test)
print("R²:", r2_score(Y_test, Y_pred))
print("MSE:", mean_squared_error(Y_test, Y_pred))
print("MAE:", mean_absolute_error(Y_test, Y_pred))
