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
# To add polynomial features
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline



#--- Visualisation ---
import matplotlib.pyplot as plt
import seaborn as sns

#--- Import our data ---
data = pd.read_feather("PulseBat.feather")

#--- Data Preprocessing and Aggregation ---
#ATTENTION! SORT HERE (BEFORE DROPPING THE OTHER COLUMNS)
data = data.sort_values(by=["SOC","SOE"],ascending=[True,False])



#Only keep Numerical Columns
model_data = data[['Qn', 'Q', 'SOC', 'SOE'] + [f'U{i}' for i in range(1, 22)] + ['SOH']]

#Split the data into input (SOC, SOE, U1-U21) and output (SOH)
X = model_data[[f"U{i}" for i in range(1, 22)] + ["SOC", "SOE"]]
Y = model_data["SOH"]





#Split the data into Train and Test sets (80/20)
#ATTENTION! CURRENT "SORTING METHOD" IS RANDOM: "random_state=42"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,shuffle=False)






#--- Training ---
model = LinearRegression()
model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linreg", LinearRegression())
])
model.fit(X_train, Y_train)

#--- Evaluation (test vs prediction) ---
Y_pred = model.predict(X_test)
print("R²:", r2_score(Y_test, Y_pred))
print("MSE:", mean_squared_error(Y_test, Y_pred))
print("MAE:", mean_absolute_error(Y_test, Y_pred))




# 1) Fit a quick baseline on TRAIN to get residuals
# _baseline = LinearRegression().fit(X_train, Y_train)
# _resid = Y_train - _baseline.predict(X_train)

# 2) Median Absolute Deviation (MAD) threshold
# med = np.median(_resid)
# mad = np.median(np.abs(_resid - med))

# If MAD is zero , keep all points to avoid dropping everything
# if mad == 0:
#     keep_mask = np.ones_like(_resid, dtype=bool)
# else:
#     tol = 3.5 * mad  
#     keep_mask = np.abs(_resid - med) <= tol

# 3) Filter the TRAIN set only 
# X_train_clean = X_train[keep_mask]
# Y_train_clean = Y_train[keep_mask]
# print(f"[Outlier removal] Dropped {len(Y_train) - keep_mask.sum()} train rows, kept {keep_mask.sum()}.")

# ==========================
# Train on cleaned data
# ==========================
# model = LinearRegression()
# model.fit(X_train_clean, Y_train_clean)