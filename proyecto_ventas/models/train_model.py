import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from utils import load_csv, basic_cleaning

def main():
    print("Cargando dataset limpio...")

    df = load_csv("data/cleaned_ventas.csv")

    X = df[["region", "category", "discount", "quantity"]]
    y = df["sales"]

    cat_cols = ["region", "category"]

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough"
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Entrenando modelo...")
    pipeline.fit(X_train, y_train)

    # Predicciones para métricas
    y_pred = pipeline.predict(X_test)

    # Calcular métricas
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Guardar métricas
    metrics = {
        "r2_score": r2,
        "mae": mae
    }

    if not os.path.exists("models"):
        os.makedirs("models")

    joblib.dump(pipeline, "models/modelo_ventas.pkl")
    joblib.dump(metrics, "models/metrics.pkl")

    print("Modelo y métricas guardadas.")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

if __name__ == "__main__":
    main()
