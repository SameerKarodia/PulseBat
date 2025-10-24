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
#To add feature scaling
from sklearn.preprocessing import StandardScaler
#cross validation
from sklearn.model_selection import cross_val_score, KFold

#--- Visualisation ---
import matplotlib.pyplot as plt

#--- Import our data ---
data = pd.read_feather("PulseBat.feather")

#--- Data Preprocessing and Aggregation ---
data = data.sort_values(by=["SOC","SOE"],ascending=[True,False])


#Only keep Numerical Columns
model_data = data[['Qn', 'Q', 'SOC', 'SOE'] + [f'U{i}' for i in range(1, 22)] + ['SOH']]

#Split the data into input (SOC, SOE, U1-U21) and output (SOH)
X = model_data[[f"U{i}" for i in range(1, 22)] + ["SOC", "SOE"]]
Y = model_data["SOH"]


#Split the data into Train and Test sets (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,shuffle=False)



#=======================
# outlier removal steps
# 1) Fit a quick baseline on TRAIN to get residuals
_baseline = LinearRegression().fit(X_train, Y_train)
_resid = Y_train - _baseline.predict(X_train)

# 2) Median Absolute Deviation (MAD) threshold
med = np.median(_resid)
mad = np.median(np.abs(_resid - med))

# If MAD is zero , keep all points to avoid dropping everything
if mad == 0:
    keep_mask = np.ones_like(_resid, dtype=bool)
else:
    tol = 3.5 * mad  
    keep_mask = np.abs(_resid - med) <= tol

# 3) Filter the TRAIN set only 
X_train_clean = X_train[keep_mask]
Y_train_clean = Y_train[keep_mask]
print(f"[Outlier removal] Dropped {len(Y_train) - keep_mask.sum()} train rows, kept {keep_mask.sum()}.")
#=======================



model = LinearRegression()

#=======================
#use polynomial factoring to extend num of inputs
model = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linreg", LinearRegression())
])
#=======================

# ==========================
# Cross-Validation Check
cv = KFold(n_splits=5, shuffle=True, random_state=42)  # ensure randomness
cv_scores = cross_val_score(model, X, Y, cv=cv, scoring='r2')

print("\nCross-Validation Results (check for leaks):")
print("Fold R² scores:", np.round(cv_scores, 5))
print("Mean R²:", np.mean(cv_scores))
print("Std Dev:", np.std(cv_scores))
# ==========================

# ==========================
# Train on cleaned and data
model.fit(X_train_clean, Y_train_clean)
# ==========================

#--- Evaluation (test vs prediction) ---
Y_pred = model.predict(X_test)
print("\nTesting Results (how accurate is our model):")
print("R²:", r2_score(Y_test, Y_pred))
print("MSE:", mean_squared_error(Y_test, Y_pred))
print("MAE:", mean_absolute_error(Y_test, Y_pred))

# CROSS-VALIDATION (replaces single train-test split)
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

# Scale features for fair comparison
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define model
model = LinearRegression()

# Define 5-fold cross-validation setup
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define metrics to evaluate
scoring = {
    'r2': 'r2',
    'mse': make_scorer(mean_squared_error),
    'mae': make_scorer(mean_absolute_error)
}

# Perform Cross-Validation
results = cross_validate(model, X_scaled, Y, cv=kfold, scoring=scoring)

# --- Evaluation (average across folds) ---
print("R² per fold:", results['test_r2'])
print("MSE per fold:", results['test_mse'])
print("MAE per fold:", results['test_mae'])
print("\nMean R²:", np.mean(results['test_r2']))
print("Mean MSE:", np.mean(results['test_mse']))
print("Mean MAE:", np.mean(results['test_mae']))

# ====================================
# --- USER INPUT PREDICTION SYSTEM ---
# ====================================

print("\n--- Battery Health Prediction System ---")

# Ask for user-defined SOH threshold
try:
    threshold = float(input("Enter SOH threshold (e.g., 0.8 for 80%): "))
except ValueError:
    print("Invalid input. Defaulting threshold to 0.8")
    threshold = 0.8

# --- User inputs for voltages in one go ---
print("\nPaste 21 cell voltages below (separated by tabs or spaces):")
try:
    row_input = input().strip()
    # split by any whitespace (tabs, spaces, etc.)
    parts = row_input.split()
    U_values = [float(x) for x in parts]
    if len(U_values) != 21:
        raise ValueError(f"Expected 21 values, got {len(U_values)}")
except Exception as e:
    print(f"Input error ({e}). Filling missing values with 0.0.")
    while len(U_values) < 21:
        U_values.append(0.0)

# Get SOC and SOE
try:
    soc = float(input("SOC value: "))
    soe = float(input("SOE value: "))
except ValueError:
    soc, soe = 0.0, 0.0

# Prepare input vector
input_data = np.array(U_values + [soc, soe]).reshape(1, -1)

# Predict SOH
predicted_soh = model.predict(input_data)[0]
print(f"\nPredicted SOH: {predicted_soh:.4f}")

# Compare with threshold
if predicted_soh >= threshold:
    print("Battery Status: Healthy")
else:
    print("Battery Status: Has a Problem")
