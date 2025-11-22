import pandas as pd

def load_csv(path):
    """Carga un archivo CSV y lo retorna como DataFrame."""
    return pd.read_csv(path)

def basic_cleaning(df):
    """Realiza limpieza completa del dataset."""

    # Eliminar nulos
    df = df.dropna()

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Nueva columna: ventas después del descuento
    df["sales_after_discount"] = df["sales"] * (1 - df["discount"])

    # Ejemplo de uso de .loc (filtrar ventas mayores a 0)
    df = df.loc[df["sales"] > 0]

    # Ejemplo de uso de .iloc (tomar todas las filas y primeras 6 columnas)
    df_subset = df.iloc[:, :6]  # no lo guardamos, solo lo usamos para mostrar

    return df
