import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import joblib
from utils import basic_cleaning

def main():
    print("Cargando dataset limpio...")
    df = pd.read_csv("data/cleaned_ventas.csv")

    print("Realizando limpieza final...")
    df = basic_cleaning(df)

    target = "sales"

    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en el dataset. Verifica el nombre exacto.")

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=['number']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("reg", Ridge())
    ])

    print("Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Entrenando modelo...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    print(f"\nR2 del modelo: {r2:.4f}\n")

    joblib.dump(model, "models/modelo_regresion.joblib")
    print("Modelo guardado en: models/modelo_regresion.joblib")

if __name__ == "__main__":
    main()
