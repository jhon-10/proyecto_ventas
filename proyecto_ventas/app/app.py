import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.title("📊 Sistema de Predicción de Ventas")

# Cargar dataset y modelo
df = pd.read_csv("data/cleaned_ventas.csv")
modelo = joblib.load("models/modelo_ventas.pkl")

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["📁 Datos", "📈 Análisis Exploratorio (EDA)", "🔮 Predicción"])

# -------------------- TAB 1: Datos --------------------
with tab1:
    st.header("📁 Vista del Dataset")
    st.dataframe(df)

# ---------------- TAB 2: ANALISIS EDA ----------------
with tab2:
    st.header("📈 Análisis Exploratorio de Datos")

    st.subheader("1. Ventas por categoría")
    fig1, ax1 = plt.subplots()
    sns.barplot(data=df, x="category", y="sales", estimator=sum, ax=ax1)
    st.pyplot(fig1)

    st.subheader("2. Dispersión: Cantidad vs Ventas")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="quantity", y="sales", ax=ax2)
    st.pyplot(fig2)

    st.subheader("3. Boxplot: Ventas por región")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x="region", y="sales", ax=ax3)
    st.pyplot(fig3)

    st.subheader("4. Correlación entre variables numéricas")
    corr = df[["sales", "discount", "quantity", "sales_after_discount"]].corr()
    fig4, ax4 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax4)
    st.pyplot(fig4)

# ------------ TAB 3: Predicción -----------------
# Mostrar métricas del modelo
st.subheader("📊 Métricas del Modelo")

metrics = joblib.load("models/metrics.pkl")

st.write(f"**R² Score:** {metrics['r2_score']:.4f}")
st.write(f"**MAE:** {metrics['mae']:.4f}")

with tab3:
    st.header("🔮 Predicción de Ventas")

    region = st.selectbox("Región:", ["East", "West", "South", "Central"])
    categoria = st.selectbox("Categoría:", ["Furniture", "Technology", "Office Supplies"])
    descuento = st.slider("Descuento:", 0.0, 1.0, 0.1)
    cantidad = st.number_input("Cantidad:", min_value=1, step=1)

    if st.button("Predecir"):
        datos = pd.DataFrame({
            "region": [region],
            "category": [categoria],
            "discount": [descuento],
            "quantity": [cantidad]
        })

        prediccion = modelo.predict(datos)[0]

        st.success(f"📈 Predicción de ventas: **${prediccion:.2f}**")

