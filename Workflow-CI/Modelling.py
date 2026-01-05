import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# MLFlow
mlflow.set_experiment("SML_Rega Ardana")

# Dataset
X_train = pd.read_csv('Flood_Preprocessing/X_train.csv')
X_test = pd.read_csv('Flood_Preprocessing/X_test.csv')
y_train = pd.read_csv('Flood_Preprocessing/y_train.csv')
y_test = pd.read_csv('Flood_Preprocessing/y_test.csv')

# Meratakan y agar formatnya sesuai standar Scikit-Learn (1D Array)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print("Data berhasil dimuat.")

# HYPERPARAMETER TUNING
rf = RandomForestRegressor(random_state=42)

# Grid parameter
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, 15],
    'min_samples_leaf': [2, 5, 10]
}

# Mencari parameter terbaik
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Tuning Selesai. Parameter Terbaik: {best_params}")

# MANUAL LOGGING
with mlflow.start_run(run_name="Modelling"):
    
    # Log Parameter (Manual)
    print("Logging parameters ke MLflow...")
    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)

    # Prediksi & Hitung Metrik
    y_pred = best_model.predict(X_test)
    
    # Hitung metrik standar regresi
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log Metrik (Manual)
    print("Logging metrics ke MLflow...")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)

    # Log Model (Manual)
    print("Logging model ke MLflow...")
    mlflow.sklearn.log_model(best_model, "model")

    print(f"R2 Score (Test): {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")




