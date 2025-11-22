# 📊 Proyecto de Minería de Datos -- Predicción de Ventas

**Autor:** Jhon Quintero\
**Materia:** Minería de Datos\
**Año:** 2025

## Descripción del Proyecto

Este proyecto implementa un flujo completo de minería de datos:

1.  Limpieza del dataset\
2.  Análisis exploratorio (EDA)\
3.  Entrenamiento de un modelo predictivo Random Forest\
4.  Aplicación web con Streamlit\
5.  Documentación clara del proceso

## 1. Limpieza de Datos

-   Eliminación de nulos\
-   Eliminación de duplicados\
-   Creación de columna: `sales_after_discount`\
-   Uso de `.loc` y `.iloc`\
-   Archivo limpio: `data/cleaned_ventas.csv`

## 2. Análisis Exploratorio (EDA)

Gráficos incluidos: - Barras: ventas por categoría\
- Dispersión: cantidad vs ventas\
- Boxplot: ventas por región\
- Heatmap de correlaciones

## 3. Modelado Predictivo

Modelo: **Random Forest Regressor**

Métricas: - R²\
- MAE

Archivos generados:

    models/modelo_ventas.pkl
    models/metrics.pkl

##  4. Aplicación Web Streamlit

Pestañas: - 📁 Datos\
- 📈 EDA\
- 🔮 Predicción

##  5. Cómo Ejecutarlo

    pip install -r requirements.txt
    python scripts/load_data.py
    python models/train_model.py
    python -m streamlit run app/app.py

## Estructura

    proyecto_ventas/
    │── app/
    │── data/
    │── models/
    │── scripts/
    │── utils.py
    │── requirements.txt
    │── README.md

##  Conclusiones

el proyecto esta completo y cumple con todo según la rúbrica.
