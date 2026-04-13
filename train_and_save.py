"""
train_and_save.py
-----------------
Trains a Stacking Ensemble (GradientBoosting + ExtraTrees → Ridge)
on the personal carbon footprint dataset and saves:
  - model/stacking_model.pkl
  - model/preprocessor.pkl
  - model/feature_names.pkl
  - model/model_metrics.pkl

Run:  python train_and_save.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ── 1. Load data ────────────────────────────────────────────────────────────
DATA_PATH = "personal_carbon_footprint_behavior.csv"
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")

# ── 2. Feature selection (no leakage — raw features only) ───────────────────
NUMERIC_FEATURES = [
    "distance_km",
    "electricity_kwh",
    "renewable_usage_pct",
    "screen_time_hours",
    "waste_generated_kg",
    "eco_actions",
]
CATEGORICAL_FEATURES = ["day_type", "transport_mode", "food_type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "carbon_footprint_kg"

X = df[ALL_FEATURES]
y = df[TARGET]

# ── 3. Preprocessor ─────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

# ── 4. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_tr = preprocessor.fit_transform(X_train)
X_te = preprocessor.transform(X_test)
print(f"Train: {X_tr.shape}  |  Test: {X_te.shape}")

# ── 5. Stacking Ensemble ─────────────────────────────────────────────────────
print("\nTraining Stacking Ensemble (GB + ExtraTrees → Ridge) …")
estimators = [
    (
        "gb",
        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        ),
    ),
    (
        "et",
        ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    ),
]
model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(),
    cv=3,
    n_jobs=-1,
)
model.fit(X_tr, y_train)
print("Training complete.")

# ── 6. Evaluate ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_te)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
cv_r2 = cross_val_score(model, X_tr, y_train, cv=5, scoring="r2").mean()

metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv_r2}

print(f"\n{'─'*40}")
print(f"  MAE  : {mae:.4f} kg CO₂e")
print(f"  RMSE : {rmse:.4f} kg CO₂e")
print(f"  R²   : {r2:.4f}")
print(f"  CV R²: {cv_r2:.4f}")
print(f"{'─'*40}")

# ── 7. Save artefacts ────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model,        "model/stacking_model.pkl")
joblib.dump(preprocessor, "model/preprocessor.pkl")
joblib.dump(ALL_FEATURES, "model/feature_names.pkl")
joblib.dump(metrics,      "model/model_metrics.pkl")

print("\n✅ Saved to model/")
print("   • stacking_model.pkl")
print("   • preprocessor.pkl")
print("   • feature_names.pkl")
print("   • model_metrics.pkl")