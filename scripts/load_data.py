import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils import load_csv, basic_cleaning

def main():
    print("Cargando dataset...")

    df = load_csv("data/ventas.csv")

    print("Limpiando dataset...")
    df = basic_cleaning(df)

    # Guardar dataset limpio
    df.to_csv("data/cleaned_ventas.csv", index=False)

    print("Proceso completado.")
    print("Archivo limpio guardado en: data/cleaned_ventas.csv")

if __name__ == "__main__":
    main()

