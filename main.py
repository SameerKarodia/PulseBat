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
from sklearn.impute import SimpleImputer  



#--- Visualisation ---
import matplotlib.pyplot as plt
#import seaborn as sns

#--- Import our data ---
data = pd.read_feather("PulseBat.feather")

# DEFINE U-COLS & SORTING

U_cols = [f"U{i}" for i in range(1, 22)]

#--- Data Preprocessing and Aggregation ---
#ATTENTION! SORT HERE (BEFORE DROPPING THE OTHER COLUMNS)
data = data.sort_values(by=["SOC","SOE"],ascending=[True,False])

# DROP HIGHLY CORRELATED U-COLS (safer)

# NEW (whole block below is not in second code)
CORR_THRESH = 0.98
corr_matrix = data[U_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESH)]

print("Original U_cols (21):", U_cols)                  # NEW
print(f"Correlation threshold: {CORR_THRESH}")          # NEW
print("Dropping:", to_drop)                             # NEW

if to_drop:
    data = data.drop(columns=to_drop)                   # NEW

U_cols = [c for c in data.columns if c.startswith("U")] # NEW
print("Remaining U_cols:", U_cols)                      # NEW
print("Counts -> before: 21 | after:", len(U_cols))     # NEW

# LIGHT PACK-LEVEL FEATURES

# NEW (these engineered features are not in second code)
data["U_avg"] = data[U_cols].mean(axis=1)              
data["U_std"] = data[U_cols].std(axis=1).fillna(0)     
data["U_range"] = (data[U_cols].max(axis=1) - data[U_cols].min(axis=1))
data["SOE_per_SOC"] = data["SOE"] / (data["SOC"] + 1e-6)
data["U_var_ratio"] = data["U_std"] / (data["U_avg"] + 1e-6)


# Only keep Numerical Columns (now dynamic U_cols + pack features)
model_data = data[['Qn', 'Q', 'SOC', 'SOE'] + U_cols +
                  ['U_avg', 'U_std', 'U_range', 'SOE_per_SOC', 'U_var_ratio', 'SOH']]  # NEW 

# Split the data into input and output
X = model_data[U_cols + ["SOC", "SOE", "U_avg", "U_std", "U_range", "SOE_per_SOC", "U_var_ratio"]]  # NEW 
Y = model_data["SOH"]


# Split the data into Train and Test sets (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
print("X shapes -> X_train:", X_train.shape, "| X_test:", X_test.shape)  # NEW



#=======================
# outlier removal steps
#=======================
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



# MODEL → IMPUTE → SCALE → POLY → LINEAR REGRESSION

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),              # NEW (not in second code)
    ("scaler", StandardScaler(with_mean=True, with_std=True)),  # NEW (not in second code)
    ("poly", PolynomialFeatures(degree=2, include_bias=False)), # (poly exists in second code too)
    ("linreg", LinearRegression())
])

# ==========================
# Train on cleaned and data
# ==========================
model.fit(X_train_clean, Y_train_clean)

#--- Evaluation (test vs prediction) ---
Y_pred = model.predict(X_test)
print("R²:", r2_score(Y_test, Y_pred))
print("MSE:", mean_squared_error(Y_test, Y_pred))
print("MAE:", mean_absolute_error(Y_test, Y_pred))





#outlier removal steps
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
