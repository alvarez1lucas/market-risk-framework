# 📉 Advanced Market Risk Framework & Hedging Optimizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](TU_URL_DE_STREAMLIT_AQUI)

Este framework integral de Riesgo de Mercado permite el análisis, la predicción y la mitigación de riesgos en portafolios globales. Desarrollado con un enfoque cuantitativo, combina econometría clásica, Machine Learning y algoritmos evolutivos para enfrentar escenarios de alta volatilidad y eventos de "Cisne Negro".

---

## 🚀 Características Principales

### 1. Ingeniería de Datos y Valuación de Bonos
* **Limpieza Nuclear:** Manejo automatizado de outliers y corrección de retornos imposibles (ej. Precios de petróleo negativos).
* **Bond Risk Engine:** Ajuste dinámico de sensibilidad mediante Duración para bonos soberanos (US, UK, GER, JPN), convirtiendo yields en retornos de precio reales.

### 2. Modelado de Riesgo Dinámico (ML & GARCH)
* **DCC-GARCH:** Implementación de modelos de volatilidad condicional para capturar el "volatility clustering".
* **XGBoost Risk Predictor:** Uso de Machine Learning para predecir el VaR 99% basado en variables macroeconómicas como el VIX y el spread de la curva de rendimientos (10Y-2Y).

### 3. Stress Test y Teoría de Valores Extremos (EVT)
* **t-Copula Monte Carlo:** Simulación de 100,000 escenarios capturando dependencias extremas y correlaciones de crisis (*Crash Correlation*).
* **Extreme Value Theory (EVT):** Ajuste de la Distribución de Pareto Generalizada (GPD) para calcular el **Expected Shortfall (ES) al 99.9%**, cumpliendo con los estándares de Basilea III.

### 4. Optimizador de Hedging (Differential Evolution)
* **Evolución Diferencial:** Algoritmo heurístico para encontrar la política de cobertura óptima.
* **Target de Optimización:** Minimización del riesgo de cola (Expected Shortfall) ajustado por costos de fricción y transacción.
* **Implementación Turbo:** Motor vectorizado en NumPy que reduce el tiempo de cómputo en un 95%.

---

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.11+
* **Framework:** Streamlit
* **Análisis de Datos:** Pandas, NumPy (Vectorización)
* **Modelos Cuantitativos:** SciPy (Optimization), Arch (GARCH), Scikit-Learn, XGBoost.
* **Visualización:** Matplotlib, Seaborn.

---

## 📊 Estructura del Proyecto
```text
├── app_market_risk.py    # Aplicación principal de Streamlit
├── csv_final_datos.csv   # Dataset de activos globales (Bonos, Equities, Commodities)
├── outputs.ipynb         # Scripts
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Documentación técnica
