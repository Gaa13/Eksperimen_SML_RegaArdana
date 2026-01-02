import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestRegressor

# Dataset
X_train = pd.read_csv('Flood_Preprocessing/X_train.csv')
X_test = pd.read_csv('Flood_Preprocessing/X_test.csv')
y_train = pd.read_csv('Flood_Preprocessing/y_train.csv')
y_test = pd.read_csv('Flood_Preprocessing/y_test.csv')

# Meratakan y agar formatnya sesuai standar Scikit-Learn (1D Array)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print("Data berhasil dimuat.")

# MLFLOW
mlflow.set_experiment("Flood_Prediction")

# AUTOLOG
mlflow.autolog()

# MODEL
print("Mulai Training Model Basic...")

with mlflow.start_run(run_name="Autolog"):
    model = RandomForestRegressor(random_state=42)
    
    model.fit(X_train, y_train)
    
    # opsional
    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)
    
    print(f"Training Selesai!")
    print(f"Akurasi Training: {acc_train:.4f}")
    print(f"Akurasi Testing: {acc_test:.4f}")
