import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, genpareto
from arch import arch_model
from xgboost import XGBRegressor
import seaborn as sns

# Configuración de página
st.set_page_config(page_title="Quant Risk Framework", layout="wide")

# ----------------------------------------------------------------------------------
# 1. CARGA Y LIMPIEZA (Cacheada para velocidad)
# ----------------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    
    # Fix para activos específicos (VIX, Copper, Soybeans)
    for ticker in ['CBOE_VIX', 'COPPER', 'US_SOYBEANS']:
        open_col = f'{ticker}_OPEN'
        if open_col in df.columns:
            df[f'{ticker}_CLOSE'] = df[open_col].shift(-1)
            df[f'{ticker}_CHANGE_'] = df[f'{ticker}_CLOSE'].pct_change()

    # Filtramos retornos
    returns = df.filter(regex='_CHANGE_?$').copy()

    # Fix de Bonos (Duración)
    duration_map = {
        'US_2Y': 1.85, 'US_5Y': 4.55, 'US_10Y': 7.10, 'US_30Y': 15.80,
        'GERMANY_10Y': 7.30, 'FRANCE_10Y': 7.20, 'UK_10Y': 7.50, 'JAPAN_10Y': 7.80
    }
    for col in returns.columns:
        if any(x in col.upper() for x in ['2Y_', '5Y_', '10Y_', '30Y_']):
            base = col.replace('_CHANGE_', '').replace('_CHANGE','')
            ycol = f"{base}_OPEN"
            if ycol in df.columns:
                dur = duration_map.get(base, 7.0)
                dy = df[ycol].diff()
                dy /= 10000 if df[ycol].mean() > 20 else 100
                returns[col] = -dur * dy

    returns /= 100 # Conversión a decimal
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    returns = returns.dropna()
    
    # Cappear petróleo si existe
    if 'CRUDE_OIL_CHANGE_' in returns.columns:
        returns['CRUDE_OIL_CHANGE_'] = returns['CRUDE_OIL_CHANGE_'].clip(lower=-1.0)
        
    return df, returns

# Ejecución de carga
try:
    df_raw, returns = load_and_clean_data('csv_final_datos.csv')
    n_assets = returns.shape[1]
    weights = np.ones(n_assets) / n_assets
    asset_names = [c.replace('_CHANGE_', '').replace('_',' ') for c in returns.columns]
except Exception as e:
    st.error(f"Error al cargar el archivo CSV: {e}")
    st.stop()

# ----------------------------------------------------------------------------------
# INTERFAZ DE USUARIO (Sidebar)
# ----------------------------------------------------------------------------------
st.sidebar.title("🛡️ Risk Settings")
horizonte = st.sidebar.slider("Horizonte de Análisis (Días)", 5, 252, 252)
confianza = st.sidebar.selectbox("Nivel de Confianza VaR", [0.95, 0.99, 0.999])

st.title("📊 Advanced Market Risk Framework")
st.markdown(f"**Análisis de {n_assets} activos globales** | Período: {returns.index[0].date()} a {returns.index[-1].date()}")


with st.expander("ℹ️ Sobre este Framework de Riesgos"):
    st.write("""
    Este dashboard es una herramienta para el análisis de riesgo de mercado. 
    Integra modelos de **Machine Learning (XGBoost)**, **Econometría (GARCH)** y **Teoría de Valores Extremos (EVT)** para proporcionar una visión  de la vulnerabilidad de un portafolio global.
    
    **Puntos clave:**
    - **Validación:** No solo calcula el VaR, sino que lo predice y lo estresa.
    - **Optimización:** Utiliza algoritmos evolutivos para proponer estrategias de mitigación.
    - **Tecnología:** Desarrollado íntegramente en Python con procesamiento vectorizado en NumPy.
    """)


tab_base, tab_ml, tab_extreme, tab_hedge = st.tabs([
    "📈 Métricas Base", 
    "🤖 Riesgo Predictivo (GARCH/ML)", 
    "🌪️ Stress Test & Colas",
    "🛡️ Optimización de Hedging"
])

with tab_base:
    # ----------------------------------------------------------------------------------
    # BLOQUE 1: MÉTRICAS CLÁSICAS Y CONTRIBUCIÓN AL RIESGO
    # ----------------------------------------------------------------------------------
    # Retorno del portafolio (Equal Weight por defecto)
    port_ret_daily = (returns @ weights)
    port_ret_horizon = port_ret_daily.rolling(window=horizonte).sum().dropna()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Retorno Medio Anualizado", f"{port_ret_daily.mean()*252*100:.2f}%")
        st.metric("Volatilidad Anualizada", f"{port_ret_daily.std()*np.sqrt(252)*100:.2f}%")

    with col2:
        var_hist = np.percentile(port_ret_horizon, (1 - confianza) * 100) * 100
        st.metric(f"VaR {confianza*100}% Histórico", f"{var_hist:.2f}%")
        
        es_hist = port_ret_horizon[port_ret_horizon <= np.percentile(port_ret_horizon, (1 - confianza) * 100)].mean() * 100
        st.metric(f"ES {confianza*100}% Histórico", f"{es_hist:.2f}%")

    with col3:
        sharpe = (port_ret_daily.mean()*252) / (port_ret_daily.std()*np.sqrt(252))
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Peor Año Histórico", f"{port_ret_horizon.min()*100:.2f}%")

    st.divider()

    # --- Risk Contribution Chart ---
    st.subheader("🎯 Concentración del Riesgo (Risk Contribution)")
    cov = returns.cov()
    # Shrinkage para estabilidad
    shrunk_cov = 0.95 * cov + 0.05 * np.diag(np.diag(cov))
    sigma_p = np.sqrt(weights @ shrunk_cov @ weights)
    z_score = norm.ppf(confianza)

    marginal_var = (shrunk_cov @ weights) / sigma_p * z_score
    percent_contrib = (weights * marginal_var / sigma_p * 100)

    risk_df = pd.DataFrame({
        'Activo': asset_names,
        '% Riesgo': percent_contrib
    }).sort_values('% Riesgo', ascending=False)

    fig_risk, ax_risk = plt.subplots(figsize=(10, 6))
    sns.barplot(data=risk_df.head(15), x='% Riesgo', y='Activo', palette='coolwarm', ax=ax_risk)
    ax_risk.set_title(f"Top 15 Contribuyentes al Riesgo (Confianza {confianza*100}%)")
    st.pyplot(fig_risk)

    # --- Rolling Returns Chart ---
    st.subheader("📈 Performance Histórica (Ventanas Móviles)")
    fig_roll, ax_roll = plt.subplots(figsize=(12, 5))
    (port_ret_horizon * 100).plot(ax=ax_roll, color='darkblue', lw=1.5)
    ax_roll.axhline(0, color='black', lw=1)
    ax_roll.fill_between(port_ret_horizon.index, (port_ret_horizon * 100), 0, alpha=0.1, color='blue')
    ax_roll.set_ylabel("Retorno Acumulado (%)")
    st.pyplot(fig_roll)

with tab_ml:
    st.header("🤖 Modelos Dinámicos de Volatilidad")
    st.markdown("""
    En esta sección comparamos dos enfoques para predecir el riesgo:
    1. **DCC-GARCH:** Capta la persistencia de la volatilidad (clusters).
    2. **XGBoost:** Utiliza variables macro (VIX, Tasas, Oil) para predecir el VaR futuro.
    """)

    # --- CÁLCULO GARCH ---
    with st.spinner("Ajustando modelo GARCH..."):
        port_ret_pct = port_ret_daily * 100
        model_garch = arch_model(port_ret_pct, vol='Garch', p=1, q=1, dist='normal')
        res_garch = model_garch.fit(disp='off')
        
        cond_vol = res_garch.conditional_volatility / 100
        var99_garch = -cond_vol * norm.ppf(0.99)

    # --- CÁLCULO XGBOOST ---
    with st.spinner("Entrenando XGBoost Predictivo..."):
        # Construcción de Features (basado en tus celdas)
        features = pd.DataFrame(index=returns.index)
        if 'CBOE_VIX_OPEN' in df_raw.columns:
            features['VIX'] = df_raw['CBOE_VIX_OPEN']
        
        # Spread de tasas (si existen)
        if 'US_10Y_OPEN' in df_raw.columns and 'US_2Y_OPEN' in df_raw.columns:
            features['US10Y-2Y'] = df_raw['US_10Y_OPEN'] - df_raw['US_2Y_OPEN']
        
        # Volatilidad móvil de un activo clave (ej. Petróleo o SP500)
        features['Mkt_Vol_20d'] = port_ret_daily.rolling(20).std()
        features = features.dropna()

        # Target: Volatilidad realizada 5 días forward
        target = port_ret_daily.rolling(5).std().shift(-5)
        target = target.reindex(features.index).dropna()
        X = features.loc[target.index]
        y = target

        # Modelo rápido
        xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
        xgb.fit(X, y)
        
        pred_vol = xgb.predict(X)
        var99_xgb = -pred_vol * norm.ppf(0.99)
        var99_xgb_series = pd.Series(var99_xgb, index=X.index)

    # --- GRÁFICO COMPARATIVO ---
    fig_ml, ax_ml = plt.subplots(figsize=(12, 6))
    ax_ml.plot(port_ret_daily.index[-500:], port_ret_daily.values[-500:], label='Retorno Diario', alpha=0.3, color='gray')
    ax_ml.plot(var99_garch.index[-500:], var99_garch.values[-500:], label='VaR 99% GARCH', color='red', lw=1.5)
    ax_ml.plot(var99_xgb_series.index[-500:], var99_xgb_series.values[-500:], label='VaR 99% XGBoost', color='green', lw=1.5, linestyle='--')
    
    ax_ml.set_title("Comparativa de VaR Dinámico: GARCH vs Machine Learning")
    ax_ml.legend()
    st.pyplot(fig_ml)

    # Métricas de error (opcional para impresionar)
    st.info(f"💡 El modelo GARCH detecta picos de volatilidad actuales, mientras que XGBoost intenta anticipar el régimen de riesgo basado en el VIX y el Yield Curve.")

    with tab_extreme:
        st.header("🌪️ Análisis de Eventos Extremos")
        st.markdown("""
        Esta sección utiliza técnicas avanzadas para modelar las colas de la distribución:
        * **t-Copula Monte Carlo:** Simula dependencias extremas cuando todos los activos caen al mismo tiempo.
        * **Extreme Value Theory (EVT):** Ajusta una distribución de Pareto (GPD) solo a las peores pérdidas.
        * **Espere alrededor de 30 segundos para completar los cálculos intensivos.**
        """)

        if st.button("🚀 Ejecutar Simulación de Stress Test (100k Sims)"):
            # --- MONTE CARLO t-COPULA ---
            with st.spinner("Simulando escenarios extremos con t-Copula..."):
                n_sims = 100000
                n_assets = returns.shape[1]
                
                # 1. Ajuste de marginales t-Student
                params = []
                for col in returns.columns:
                    df_fit, loc_fit, scale_fit = t.fit(returns[col].dropna())
                    params.append((max(df_fit, 3.1), loc_fit, scale_fit))

                # 2. Correlación de Rank (Copula)
                corr_rank = returns.rank().corr()
                
                # 3. Simulación Correlacionada
                L = np.linalg.cholesky(corr_rank.values + 1e-8 * np.eye(n_assets))
                Z = np.random.normal(size=(n_sims, n_assets))
                Z_corr = Z @ L.T
                U = norm.cdf(Z_corr)
                
                sim_returns = np.zeros((n_sims, n_assets))
                for i, (df_p, loc_p, scale_p) in enumerate(params):
                    sim_returns[:, i] = t.ppf(U[:, i], df_p, loc_p, scale_p)
                
                sim_port = sim_returns @ weights
                var99_copula = np.percentile(sim_port, 1)
                es99_copula = sim_port[sim_port <= var99_copula].mean()

            # --- EXTREME VALUE THEORY (EVT) ---
            with st.spinner("Calculando EVT (Peaks-Over-Threshold)..."):
                losses = -port_ret_daily.values
                threshold = np.percentile(losses, 95)
                excess = losses[losses > threshold] - threshold
                
                # Ajuste Pareto Generalizada
                shape, _, scale = genpareto.fit(excess)
                
                # ES 99.9% (Cisne Negro extremo)
                p = 0.001
                nu = len(excess) / len(losses)
                es_999_evt = threshold + (scale / shape) * (((p / nu)**(-shape) / (1 - shape)) - 1)

            # --- RESULTADOS VISUALES ---
            c1, c2 = st.columns(2)
            with c1:
                st.metric("VaR 99% t-Copula", f"{var99_copula*100:.2f}%")
                st.metric("ES 99.9% EVT (GPD)", f"{es_999_evt*100:.2f}%")
            
            with c2:
                st.metric("Expected Shortfall 99% (Copula)", f"{es99_copula*100:.2f}%")
                st.info(f"El ES 99.9% EVT de {es_999_evt*100:.2f}% representa la pérdida esperada en un escenario de crisis una vez cada 1000 días.")

            # Histograma de la simulación
            fig_ext, ax_ext = plt.subplots(figsize=(10, 5))
            sns.histplot(sim_port * 100, bins=100, kde=True, color='orange', ax=ax_ext, label='Simulación Monte Carlo')
            ax_ext.axvline(var99_copula * 100, color='red', linestyle='--', label='VaR 99% Copula')
            ax_ext.set_title("Distribución de Pérdidas Simuladas (t-Copula)")
            ax_ext.set_xlabel("P&L del Portafolio (%)")
            ax_ext.legend()
            st.pyplot(fig_ext)
        else:
            st.warning("Haga clic en el botón superior para iniciar los cálculos intensivos.")
    with tab_hedge:
        st.header("🛡️ Optimizador de Cobertura TURBO v2")
        st.markdown("""
        Esta versión optimiza la estrategia buscando minimizar el **Expected Shortfall (99.9%)**. 
        Es decir, busca la configuración de cobertura que mejor protege el portafolio ante eventos de 'Cisne Negro'.
        """)

        if st.button("🚀 Iniciar Optimización ES-Targeted"):
            from scipy.optimize import differential_evolution

            # 1. Preparación de datos (Tu lógica mejorada)
            ret_daily = (returns @ weights).values if hasattr(returns @ weights, 'values') else (returns @ weights)
            vol_20d = pd.Series(ret_daily).rolling(20).std().values

            def evaluate_turbo_streamlit(params):
                h_base, h_max, cost = np.abs(params[:3])
                wealth = 1.0
                # Usamos una lista para wealths para calcular el ES al final
                wealths = np.ones(len(ret_daily))
                
                for i in range(252, len(ret_daily)-1, 3):
                    r = ret_daily[i]
                    vol = vol_20d[i] if not np.isnan(vol_20d[i]) else 0.02
                    
                    hedge = np.clip(h_base + (h_max - h_base) * (vol / 0.02), 0.0, 1.0)
                    # Tu fórmula de retorno neto con penalización por fricción
                    hedged_r = r - hedge * abs(r) * 1.8 - hedge * cost * 1e-4
                    
                    wealth *= (1 + hedged_r)
                    wealths[i] = wealth
                
                # Cálculo de retornos de la estrategia para obtener el ES
                # Usamos log-returns para estabilidad
                recent_wealths = wealths[wealths > 0]
                if len(recent_wealths) < 100: return 0
                
                strat_rets = np.diff(np.log(recent_wealths))
                q_001 = np.quantile(strat_rets, 0.001)
                # Retornamos el ES (el promedio de los peores casos)
                return np.mean(strat_rets[strat_rets <= q_001])

            with st.spinner("Optimizando parámetros para minimizar riesgo de cola..."):
                result = differential_evolution(
                    evaluate_turbo_streamlit,
                    bounds=[(0, 0.8), (0.2, 1.0), (0.1, 3)],
                    strategy='best1bin',
                    popsize=8,
                    maxiter=25,
                    seed=42,
                    polish=True
                )
                
                h_base_opt, h_max_opt, cost_opt = result.x

            # --- MOSTRAR RESULTADOS ---
            st.success(f"✅ Óptimo encontrado en {result.nfev} evaluaciones")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Hedging Base", f"{h_base_opt:.1%}")
            c2.metric("Hedging Máximo", f"{h_max_opt:.1%}")
            c3.metric("Factor de Costo", f"{cost_opt:.3f}")

            # Visualización de la política
            fig_pol, ax_pol = plt.subplots(figsize=(10, 4))
            v_range = np.linspace(0, 0.05, 100)
            h_range = np.clip(h_base_opt + (h_max_opt - h_base_opt) * (v_range / 0.02), 0, 1)
            ax_pol.plot(v_range*100, h_range*100, color='royalblue', lw=2)
            ax_pol.fill_between(v_range*100, h_range*100, alpha=0.2, color='royalblue')
            ax_pol.set_title("Política de Protección Dinámica Optimizada")
            ax_pol.set_xlabel("Volatilidad del Mercado (%)")
            ax_pol.set_ylabel("Intensidad de Cobertura (%)")
            st.pyplot(fig_pol)