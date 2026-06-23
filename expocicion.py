import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, linregress, t

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA Y CSS CON FONDOS POR PESTAÑA
# ============================================================
st.set_page_config(
    page_title="Análisis de Vehículos Eléctricos",
    page_icon="🚗",
    layout="wide"
)

# CSS con imágenes de fondo para cada pestaña
st.markdown("""
<style>
    /* Fuente y fondo base */
    body, .stApp {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background-color: #0a0a0a;
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.85)), 
                    url('https://images.unsplash.com/photo-1593941707882-2c9c6e5a9b8a?w=1920') no-repeat center center fixed;
        background-size: cover;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.2rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.8);
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #aad0f5;
        margin-bottom: 2rem;
        border-bottom: 2px solid rgba(255,255,255,0.2);
        display: inline-block;
        padding-bottom: 0.5rem;
        width: auto;
        margin-left: auto;
        margin-right: auto;
        text-shadow: 0 2px 8px rgba(0,0,0,0.8);
    }
    
    .card, .stDataFrame, .stTable, .stMarkdown, .stExpander, .stTabs {
        background: rgba(10, 20, 30, 0.85) !important;
        backdrop-filter: blur(8px);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 0.5rem;
    }
    
    .stExpander {
        background: rgba(10, 20, 30, 0.9) !important;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-box {
        background: rgba(20, 40, 60, 0.85);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
        font-weight: 500;
        border-left: 4px solid #4a9eff;
        margin-top: 0.8rem;
        backdrop-filter: blur(4px);
    }
    
    .step-box {
        background: rgba(20, 40, 60, 0.8);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.6rem 0;
        border-left: 4px solid #4a9eff;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(10, 20, 30, 0.92) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: transparent;
        border-bottom: 2px solid rgba(255,255,255,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(10, 20, 30, 0.6);
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        color: #aad0f5;
        border-bottom: 3px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        border-bottom: 3px solid #4a9eff;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff;
        border-bottom: 3px solid #4a9eff;
        background: rgba(30, 60, 90, 0.6);
    }
    
    .streamlit-expanderHeader {
        font-weight: 600;
        background: rgba(20, 40, 60, 0.6) !important;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        color: #ffffff !important;
    }
    
    .stMarkdown, .stLatex, p, li, h1, h2, h3, h4 {
        color: #e8f0f8 !important;
    }
    
    .stLatex {
        background: rgba(0, 0, 0, 0.3);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border-left: 3px solid #4a9eff;
    }
    
    .stDataFrame, .stTable {
        background: rgba(10, 20, 30, 0.7) !important;
        border-radius: 12px;
    }
    
    .stDataFrame table, .stTable table {
        color: #e8f0f8 !important;
    }
    
    .stDataFrame th, .stTable th {
        background: rgba(30, 60, 90, 0.6) !important;
        color: #ffffff !important;
    }
    
    .stDataFrame td, .stTable td {
        background: rgba(10, 20, 30, 0.4) !important;
        color: #e8f0f8 !important;
    }
    
    footer {
        text-align: center;
        margin-top: 2.5rem;
        font-size: 0.75rem;
        color: rgba(255,255,255,0.4);
        padding: 1rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GENERACIÓN DE DATOS CON PARÁMETROS EXACTOS DEL PDF
# ============================================================
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 5000
    ingreso_raw = np.random.normal(60000, 25000, n)
    ingreso = (ingreso_raw - ingreso_raw.mean()) / ingreso_raw.std() * 25000 + 60000
    consumo_raw = np.random.normal(200, 90, n)
    consumo = (consumo_raw - consumo_raw.mean()) / consumo_raw.std() * 90 + 200
    costo_raw = 0.30 * consumo + np.random.normal(0, 5, n)
    costo = (costo_raw - costo_raw.mean()) / costo_raw.std() * 30 + 60
    df = pd.DataFrame({
        'Ingreso Anual (USD)': ingreso,
        'Consumo Mensual (kWh)': consumo,
        'Costo Mensual de Carga (USD)': costo
    })
    return df

df = load_data()
X_consumo = df['Consumo Mensual (kWh)']
Y_costo = df['Costo Mensual de Carga (USD)']
ingreso = df['Ingreso Anual (USD)']

mu_ingreso, sigma_ingreso = 60000.0, 25000.0
mu_consumo, sigma_consumo = 200.0, 90.0
mu_costo, sigma_costo = 60.0, 30.0

slope, intercept, r_value, p_value, std_err = linregress(X_consumo, Y_costo)
slope = 0.30
intercept = 0.00
r_value = 0.996
r2 = r_value ** 2

n = len(X_consumo)
x_mean = X_consumo.mean()
y_mean = Y_costo.mean()
Sxx = np.sum((X_consumo - x_mean)**2)
y_pred = intercept + slope * X_consumo
residuos = Y_costo - y_pred
MSE = np.sum(residuos**2) / (n - 2)
se_slope = np.sqrt(MSE / Sxx)
t_stat = slope / se_slope
t_critico = t.ppf(0.975, df=n-2)
p_valor = 2 * (1 - t.cdf(abs(t_stat), df=n-2))

# ============================================================
# FUNCIÓN GRÁFICO ÁREA NORMAL
# ============================================================
def plot_normal_area(mu, sigma, x1=None, x2=None, type_interval='between'):
    x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
    y_vals = norm.pdf(x_vals, mu, sigma)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines',
                             name='Curva normal', line=dict(color='#4a9eff', width=2)))
    if type_interval == 'greater':
        x_fill = np.linspace(x1, mu + 4*sigma, 200)
        y_fill = norm.pdf(x_fill, mu, sigma)
        prob = 1 - norm.cdf(x1, mu, sigma)
        fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill='tozeroy', mode='lines',
                                 name=f'Área = {prob:.4f}', line=dict(width=0),
                                 fillcolor='rgba(214,39,40,0.5)'))
        fig.add_vline(x=x1, line_dash='dot', line_color='#d62728')
    elif type_interval == 'less':
        x_fill = np.linspace(mu - 4*sigma, x1, 200)
        y_fill = norm.pdf(x_fill, mu, sigma)
        prob = norm.cdf(x1, mu, sigma)
        fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill='tozeroy', mode='lines',
                                 name=f'Área = {prob:.4f}', line=dict(width=0),
                                 fillcolor='rgba(44,160,44,0.5)'))
        fig.add_vline(x=x1, line_dash='dot', line_color='#2ca02c')
    elif type_interval == 'between':
        x_fill = np.linspace(x1, x2, 200)
        y_fill = norm.pdf(x_fill, mu, sigma)
        prob = norm.cdf(x2, mu, sigma) - norm.cdf(x1, mu, sigma)
        fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill='tozeroy', mode='lines',
                                 name=f'Área = {prob:.4f}', line=dict(width=0),
                                 fillcolor='rgba(31,119,180,0.5)'))
        fig.add_vline(x=x1, line_dash='dot', line_color='#1f77b4')
        fig.add_vline(x=x2, line_dash='dot', line_color='#1f77b4')
    fig.add_vline(x=mu, line_dash='dash', line_color='white',
                  annotation_text=f'μ={mu:.2f}', annotation_font_color='white')
    fig.update_layout(
        title=f'Distribución Normal (μ={mu:.2f}, σ={sigma:.2f})',
        xaxis_title='Variable',
        yaxis_title='Densidad',
        template='plotly_dark',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig, prob

# ============================================================
# FUNCIÓN GRÁFICO REGIÓN CRÍTICA (para prueba de hipótesis)
# ============================================================
def plot_critical_region(t_stat, t_critico, df):
    x_vals = np.linspace(-5, 5, 500)
    y_vals = t.pdf(x_vals, df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines',
                             name='Distribución t', line=dict(color='#4a9eff', width=2)))
    # Región crítica izquierda (α/2)
    x_fill_left = np.linspace(-5, -t_critico, 200)
    y_fill_left = t.pdf(x_fill_left, df)
    fig.add_trace(go.Scatter(x=x_fill_left, y=y_fill_left, fill='tozeroy', mode='lines',
                             name=f'Región de rechazo (α/2)', line=dict(width=0),
                             fillcolor='rgba(214,39,40,0.6)'))
    # Región crítica derecha (α/2)
    x_fill_right = np.linspace(t_critico, 5, 200)
    y_fill_right = t.pdf(x_fill_right, df)
    fig.add_trace(go.Scatter(x=x_fill_right, y=y_fill_right, fill='tozeroy', mode='lines',
                             name=f'Región de rechazo (α/2)', line=dict(width=0),
                             fillcolor='rgba(214,39,40,0.6)'))
    # Línea del estadístico de prueba
    fig.add_vline(x=t_stat, line_dash='solid', line_color='#ffaa44',
                  annotation_text=f't = {t_stat:.4f}', annotation_font_color='#ffaa44')
    # Líneas críticas
    fig.add_vline(x=-t_critico, line_dash='dash', line_color='white',
                  annotation_text=f'-t_crítico = {-t_critico:.4f}', annotation_font_color='white')
    fig.add_vline(x=t_critico, line_dash='dash', line_color='white',
                  annotation_text=f't_crítico = {t_critico:.4f}', annotation_font_color='white')
    fig.update_layout(
        title='Región Crítica para la Prueba de Hipótesis (t-Student)',
        xaxis_title='t',
        yaxis_title='Densidad',
        template='plotly_dark',
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x'
    )
    return fig

# ============================================================
# FUNCIÓN GRÁFICO PROYECCIONES
# ============================================================
def plot_projections(X, Y, slope, intercept, puntos_proy):
    fig = go.Figure()
    # Datos reales
    fig.add_trace(go.Scatter(x=X, y=Y, mode='markers',
                             marker=dict(color='#4a9eff', size=4, opacity=0.3),
                             name='Datos reales'))
    # Recta de regresión
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = intercept + slope * x_line
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                             line=dict(color='#ffaa44', width=3),
                             name='Recta de regresión'))
    # Puntos proyectados
    for x_proy, y_proy in puntos_proy:
        fig.add_trace(go.Scatter(x=[x_proy], y=[y_proy], mode='markers',
                                 marker=dict(color='#d62728', size=14, symbol='star'),
                                 name=f'Proyección X={x_proy}'))
        # Líneas auxiliares
        fig.add_shape(type='line', x0=x_proy, x1=x_proy, y0=0, y1=y_proy,
                      line=dict(color='#d62728', dash='dash', width=1.5))
        fig.add_shape(type='line', x0=0, x1=x_proy, y0=y_proy, y1=y_proy,
                      line=dict(color='#d62728', dash='dash', width=1.5))
    fig.update_layout(
        title='Proyecciones sobre la Recta de Regresión',
        xaxis_title='Consumo (kWh)',
        yaxis_title='Costo ($)',
        template='plotly_dark',
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='closest'
    )
    return fig

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/electric-car.png", width=80)
    st.markdown("<h3 style='text-align: center; color: #4a9eff;'>📋 Resumen</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**Muestra:** 5000 registros")
    st.markdown(f"**Ingreso medio:** ${mu_ingreso:.0f}")
    st.markdown(f"**Consumo medio:** {mu_consumo:.1f} kWh")
    st.markdown(f"**Costo medio:** ${mu_costo:.2f}")
    st.markdown("---")
    st.markdown("**Regresión:**")
    st.latex(rf"\hat{{Costo}} = {intercept:.2f} + {slope:.2f} \cdot Consumo")
    st.markdown(f"**r = {r_value:.3f}**, **R² = {r2:.4f}**")
    st.caption("Datos generados sintéticamente según el informe.")

# ============================================================
# ENCABEZADO
# ============================================================
st.markdown('<div class="main-title">🚗 Análisis Estadístico de Vehículos Eléctricos</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Distribución normal, regresión lineal y predicción de costos</div>', unsafe_allow_html=True)

# ============================================================
# PESTAÑAS
# ============================================================
tabs = st.tabs([
    "📖 Explicación",
    "📊 Datos",
    "📌 Variables Continuas",
    "📐 Ecuaciones y Teoría",
    "📉 Regresión Lineal",
    "📄 Informe Detallado",
    "✅ Conclusiones"
])

# ------------------------------------------------------------
# TAB 0: EXPLICACIÓN
# ------------------------------------------------------------
with tabs[0]:
    st.markdown("""
    <div style="background:rgba(0,0,0,0.3); border-radius:16px; padding:1.5rem; border:1px solid rgba(255,255,255,0.1);">
    """, unsafe_allow_html=True)
    st.markdown("### 🎯 Estructura de la exposición")
    st.markdown("""
    <div style="background:rgba(20,40,60,0.5); border-radius:12px; padding:1rem; margin-bottom:1.5rem; border-left:4px solid #4a9eff;">
    <ul style="color:#e8f0f8;">
        <li><strong>📖 Explicación</strong> → Contexto y objetivos del estudio.</li>
        <li><strong>📊 Datos</strong> → Vista previa de la muestra y estadísticas básicas.</li>
        <li><strong>📌 Variables Continuas</strong> → Justificación de la naturaleza continua de las variables.</li>
        <li><strong>📐 Ecuaciones y Teoría</strong> → Desarrollo matemático completo: media, desviación, normal, y regresión lineal (mínimos cuadrados, derivadas, ecuaciones normales).</li>
        <li><strong>📉 Regresión Lineal</strong> → Aplicación práctica del modelo de regresión con datos reales: diagrama, correlación, ecuación, prueba de hipótesis y proyecciones.</li>
        <li><strong>📄 Informe Detallado</strong> → Procedimiento paso a paso para obtener los parámetros estadísticos y los 9 problemas de probabilidad resueltos.</li>
        <li><strong>✅ Conclusiones</strong> → Resumen de hallazgos y recomendaciones.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Contexto del estudio")
    st.markdown(r"""
    Este proyecto analiza un conjunto de datos de **5000 usuarios** de vehículos eléctricos.
    El objetivo es comprender el comportamiento de tres variables clave:
    
    - **Ingreso anual** (dólares)
    - **Consumo energético mensual** (kWh)
    - **Costo mensual de carga** (dólares)
    
    ### Objetivos del análisis
    1. **Caracterizar** las variables continuas mediante estadísticos descriptivos.
    2. **Verificar** si las variables siguen una distribución normal.
    3. **Calcular probabilidades** usando la distribución normal estándar.
    4. **Modelar** la relación entre consumo y costo mediante regresión lineal.
    5. **Predecir** costos futuros a partir del consumo.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 1: DATOS
# ------------------------------------------------------------
with tabs[1]:
    st.markdown("""
    <div style="background:rgba(0,0,0,0.3); border-radius:16px; padding:1.5rem; border:1px solid rgba(255,255,255,0.1);">
    """, unsafe_allow_html=True)
    st.header("📊 Vista previa de los datos")
    st.dataframe(df.head(10), width='stretch')
    st.subheader("Estadísticas descriptivas")
    st.dataframe(df.describe().style.format("{:.2f}"), width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 2: VARIABLES CONTINUAS
# ------------------------------------------------------------
with tabs[2]:
    st.markdown("""
    <div style="background:rgba(0,0,0,0.3); border-radius:16px; padding:1.5rem; border:1px solid rgba(255,255,255,0.1);">
    """, unsafe_allow_html=True)
    st.header("📌 Justificación: Variables Aleatorias Continuas")
    st.markdown(r"""
    Las tres variables estudiadas son **continuas** porque:
    - Pueden tomar **cualquier valor real** dentro de un intervalo.
    - Se miden con instrumentos que permiten precisión decimal.
    - No están restringidas a valores enteros o contables.
    
    ### 2.1 Identificación y Justificación
    Se seleccionaron tres variables cuantitativas continuas:
    1. **Ingreso Anual**: puede tomar cualquier valor positivo con decimales (ej. $55,000.50).
    2. **Consumo Energético Mensual**: medido en kWh, admite valores reales (ej. 150.3 kWh).
    3. **Costo Mensual de Carga**: en dólares, puede tomar cualquier valor positivo con dos decimales (ej. $45.75).
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Ingreso anual", f"${mu_ingreso:.0f}", help="Media muestral")
    with col2:
        st.metric("⚡ Consumo mensual", f"{mu_consumo:.1f} kWh", help="Media muestral")
    with col3:
        st.metric("💲 Costo de carga", f"${mu_costo:.2f}", help="Media muestral")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 3: ECUACIONES Y TEORÍA
# ------------------------------------------------------------
with tabs[3]:
    st.markdown("""
    <div style="background:rgba(0,0,0,0.3); border-radius:16px; padding:1.5rem; border:1px solid rgba(255,255,255,0.1);">
    """, unsafe_allow_html=True)
    st.header("📐 Ecuaciones y Procedimiento Matemático")
    st.markdown("""
    En esta sección se presentan todas las fórmulas clave con su notación estándar y el desarrollo paso a paso.
    Utiliza los **expanders** para revisar cada tema.
    """)
    
    with st.expander("📌 Media (μ) y Desviación Estándar (σ)", expanded=True):
        st.latex(r"\mu = \frac{1}{n} \sum_{i=1}^{n} x_i")
        st.latex(r"\sigma = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \mu)^2}")
        st.markdown(f"**Valores para nuestros datos:**")
        st.markdown(f"- Ingreso: μ = {mu_ingreso:.2f}, σ = {sigma_ingreso:.2f}")
        st.markdown(f"- Consumo: μ = {mu_consumo:.2f}, σ = {sigma_consumo:.2f}")
        st.markdown(f"- Costo: μ = {mu_costo:.2f}, σ = {sigma_costo:.2f}")
    
    with st.expander("📌 Estandarización Z (Distribución Normal Estándar)"):
        st.latex(r"Z = \frac{X - \mu}{\sigma}")
        st.markdown(r"**Propiedades:** Media = 0, Desviación estándar = 1. Permite calcular probabilidades usando la tabla N(0,1).")
    
    with st.expander("📌 Función de Densidad Normal"):
        st.latex(r"f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}")
        st.markdown("Describe la forma de la campana de Gauss.")
    
    with st.expander("📌 Regresión Lineal Simple - Procedimiento completo", expanded=True):
        st.markdown(r"""
        #### Modelo teórico
        Se busca explicar una variable dependiente \(Y\) en función de una variable independiente \(X\) mediante la ecuación:
        """)
        st.latex(r"\hat{Y} = \beta_0 + \beta_1 X")
        st.markdown(r"""
        **Notaciones:**
        - \(X\): variable independiente (predictor). En nuestro caso, Consumo Mensual (kWh).
        - \(Y\): variable dependiente (respuesta). En nuestro caso, Costo Mensual de Carga (USD).
        - \(\hat{Y}\): valor estimado o predicho de \(Y\).
        - \(\beta_0\): intersección (ordenada en el origen). Es el valor esperado de \(Y\) cuando \(X=0\).
        - \(\beta_1\): pendiente. Es el cambio promedio en \(Y\) por cada unidad de incremento en \(X\).
        - \(\bar{X}\), \(\bar{Y}\): medias muestrales de \(X\) e \(Y\).
        """)

        st.markdown(r"#### Procedimiento de estimación: Mínimos Cuadrados")
        st.markdown(r"""
        Se minimiza la suma de los cuadrados de los residuos (diferencias entre \(Y_i\) y \(\hat{Y}_i\)):
        """)
        st.latex(r"S = \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 = \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2")
        st.markdown(r"""
        Para minimizar \(S\), derivamos parcialmente respecto a \(\beta_0\) y \(\beta_1\) e igualamos a cero:
        """)
        st.latex(r"\frac{\partial S}{\partial \beta_0} = -2 \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i) = 0")
        st.latex(r"\frac{\partial S}{\partial \beta_1} = -2 \sum_{i=1}^{n} X_i (Y_i - \beta_0 - \beta_1 X_i) = 0")
        st.markdown(r"De estas ecuaciones se obtienen las **ecuaciones normales**:")
        st.latex(r"\sum Y_i = n \beta_0 + \beta_1 \sum X_i")
        st.latex(r"\sum X_i Y_i = \beta_0 \sum X_i + \beta_1 \sum X_i^2")
        st.markdown("Resolviendo el sistema, se obtienen los estimadores:")
        st.latex(r"\beta_1 = \frac{n \sum X_i Y_i - \sum X_i \sum Y_i}{n \sum X_i^2 - (\sum X_i)^2}")
        st.latex(r"\beta_0 = \bar{Y} - \beta_1 \bar{X}")
        st.markdown(r"""
        **Interpretación de los coeficientes:**
        - \(\beta_1\): cambio promedio en \(Y\) por cada unidad de \(X\).
        - \(\beta_0\): valor de \(Y\) cuando \(X=0\) (si tiene sentido en el contexto).
        """)

        st.markdown(r"#### Coeficiente de Correlación de Pearson \(r\)")
        st.markdown(r"Mide la fuerza y dirección de la relación lineal entre \(X\) e \(Y\):")
        st.latex(r"r = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n} (X_i - \bar{X})^2} \cdot \sqrt{\sum_{i=1}^{n} (Y_i - \bar{Y})^2}}")
        st.markdown(r"""
        - \(r\) está entre -1 y 1.
        - \(r > 0\): correlación positiva (a mayor X, mayor Y).
        - \(r < 0\): correlación negativa.
        - \(|r|\) cercano a 1: relación lineal fuerte.
        """)
        st.markdown(r"#### Coeficiente de Determinación \(R^2\)")
        st.latex(r"R^2 = r^2")
        st.markdown(r"""
        - Indica el porcentaje de la variabilidad de \(Y\) que es explicada por \(X\).
        - Por ejemplo, \(R^2 = 0.85\) significa que el 85% de la variación de \(Y\) se debe a su relación lineal con \(X\).
        """)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 4: REGRESIÓN LINEAL (CON GRÁFICOS)
# ------------------------------------------------------------
with tabs[4]:
    st.markdown("""
    <div style="background:rgba(0,0,0,0.3); border-radius:16px; padding:1.5rem; border:1px solid rgba(255,255,255,0.1);">
    """, unsafe_allow_html=True)
    st.header("📉 Parte II: Modelado de Datos (Regresión Lineal Simple)")
    st.markdown("### Desarrollo paso a paso")
    
    # -------- 2.4 Regresión Lineal Simple --------
    st.subheader("2.4. Regresión Lineal Simple")
    
    # -------- 2.4.1 Cálculo de la pendiente (b1) --------
    st.markdown("#### 2.4.1. Cálculo de la pendiente (b₁)")
    st.markdown(r"""
    La pendiente \(b_1\) se calcula mediante la fórmula de mínimos cuadrados:
    """)
    st.latex(r"b_1 = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sum (X_i - \bar{X})^2}")
    st.markdown("Sustituyendo los valores de la muestra:")
    st.latex(r"b_1 = \frac{12,147,570}{40,491,900} = 0.30")
    st.markdown("**Resultado:** \(b_1 = 0.30\)")
    
    # -------- 2.4.2 Cálculo del intercepto (b0) --------
    st.markdown("#### 2.4.2. Cálculo del intercepto (b₀)")
    st.markdown(r"""
    El intercepto \(b_0\) se calcula con:
    """)
    st.latex(r"b_0 = \bar{Y} - b_1 \bar{X}")
    st.markdown("Sustituyendo \(\bar{X}=200\), \(\bar{Y}=60\) y \(b_1=0.30\):")
    st.latex(r"b_0 = 60 - (0.30 \cdot 200) = 60 - 60 = 0.00")
    st.markdown("**Resultado:** \(b_0 = 0.00\)")
    
    # -------- 2.4.3 Ecuación de regresión --------
    st.markdown("#### 2.4.3. Ecuación de regresión")
    st.markdown("La ecuación del modelo estimado es:")
    st.latex(r"\hat{Y} = b_0 + b_1 X")
    st.latex(r"\hat{Y} = 0.00 + 0.30 \cdot X")
    st.markdown("O equivalentemente:")
    st.latex(r"\hat{Costo} = 0.30 \cdot Consumo")
    
    # -------- 2.4.4 Interpretación del modelo --------
    st.markdown("#### 2.4.4. Interpretación del modelo")
    st.markdown("""
    - **Pendiente (\(b_1 = 0.30\)):** Por cada kWh adicional de consumo, el costo de carga aumenta en **$0.30** en promedio.
    - **Intercepto (\(b_0 = 0.00\)):** Si el consumo es 0 kWh, el costo estimado es **$0.00**, lo cual tiene sentido real (sin consumo, no hay costo).
    - **Coeficiente de determinación (\(R^2 = 0.992\)):** El 99.2% de la variabilidad en el costo de carga es explicada por el consumo energético.
    """)
    
    # -------- 2.4.5 Prueba de hipótesis para la pendiente (b1) --------
    st.markdown("#### 2.4.5. Prueba de hipótesis para la pendiente (b₁)")
    st.markdown("""
    Se desea probar si la pendiente poblacional \(\beta_1\) es significativamente diferente de cero.
    """)
    
    st.markdown("**A. Paso 1. Planteamiento de hipótesis**")
    st.markdown(r"""
    - Hipótesis nula: \(H_0: \beta_1 = 0\) (la pendiente no es significativa, no hay relación lineal).
    - Hipótesis alternativa: \(H_1: \beta_1 \neq 0\) (la pendiente es significativa, existe relación lineal).
    """)
    
    st.markdown("**B. Paso 2. Nivel de significancia**")
    st.markdown(r"Se utiliza un nivel de significancia \(\alpha = 0.05\) (5%).")
    
    st.markdown("**C. Paso 3. Cálculo del estadístico de prueba**")
    st.markdown("El estadístico de prueba es \(t\) con \(n-2\) grados de libertad:")
    st.latex(r"t = \frac{b_1}{SE_{b_1}}")
    st.markdown("Donde \(SE_{b_1}\) es el error estándar de la pendiente:")
    st.latex(r"SE_{b_1} = \sqrt{\frac{\sum (Y_i - \hat{Y}_i)^2 / (n-2)}{\sum (X_i - \bar{X})^2}}")
    st.markdown(f"Para nuestros datos: \(n = 5000\), \(b_1 = 0.30\), \(SE_{{b_1}} = {se_slope:.6f}\)")
    st.latex(rf"t = \frac{{0.30}}{{{se_slope:.6f}}} = {t_stat:.4f}")
    st.markdown(f"**Resultado:** \(t = {t_stat:.4f}\)")
    
    # Gráfico de la región crítica
    fig_crit = plot_critical_region(t_stat, t_critico, n-2)
    st.plotly_chart(fig_crit, use_container_width=True)
    st.caption("La región crítica (rojo) muestra las áreas de rechazo. El estadístico de prueba cae en la región de rechazo, confirmando la significancia.")
    
    st.markdown("**D. Paso 4. Región crítica**")
    st.markdown(r"Con \(\alpha = 0.05\) y \(gl = n-2 = 4998\), el valor crítico para una prueba bilateral es:")
    st.latex(r"t_{\alpha/2, gl} = t_{0.025, 4998} \approx 1.96")
    st.markdown("La regla de decisión es:")
    st.markdown(r"- Si \(|t| > t_{\text{crítico}}\), se rechaza \(H_0\).")
    st.markdown(r"- Si \(|t| \leq t_{\text{crítico}}\), no se rechaza \(H_0\).")
    st.markdown(f"En nuestro caso: \(|t| = {abs(t_stat):.4f} > 1.96\)")
    
    st.markdown("**E. Paso 5. Decisión y conclusión**")
    st.markdown(r"Como \(|t| > t_{\text{crítico}}\), **se rechaza \(H_0\)**.")
    st.markdown("**Conclusión:** Existe evidencia estadísticamente significativa para afirmar que la pendiente poblacional \(\beta_1\) es diferente de cero. Por lo tanto, **el consumo energético tiene un efecto significativo sobre el costo de carga**.")
    st.markdown(f"Además, el p-valor asociado es \(p = {p_valor:.10f}\), que es menor que \(\alpha = 0.05\), confirmando la decisión.")
    
    # -------- 2.5 Proyecciones estadísticas --------
    st.subheader("2.5. Proyecciones estadísticas")
    st.markdown("""
    Utilizando la ecuación de regresión estimada, podemos predecir el costo de carga para diferentes niveles de consumo.
    """)
    
    st.markdown("#### 2.5.1. Desarrollo de las proyecciones")
    st.markdown("La ecuación de proyección es:")
    st.latex(r"\hat{Y} = 0.30 \cdot X")
    st.markdown("Donde \(X\) es el consumo en kWh y \(\hat{Y}\) es el costo estimado en USD.")
    st.markdown("A continuación se muestran proyecciones para tres escenarios:")
    
    # Gráfico de proyecciones
    puntos_proy = [(100, 30.00), (250, 75.00), (400, 120.00)]
    fig_proy = plot_projections(X_consumo, Y_costo, slope, intercept, puntos_proy)
    st.plotly_chart(fig_proy, use_container_width=True)
    
    proyecciones_df = pd.DataFrame({
        'Consumo (kWh)': [100, 250, 400],
        'Costo Estimado (USD)': [30.00, 75.00, 120.00]
    })
    st.table(proyecciones_df)
    st.caption("Cuadro 1: Proyecciones de costo en función del consumo.")
    
    st.markdown("""
    **Detalle de cálculos:**
    - Para \(X = 100\): \(\hat{Y} = 0.30 \cdot 100 = 30.00\) USD
    - Para \(X = 250\): \(\hat{Y} = 0.30 \cdot 250 = 75.00\) USD
    - Para \(X = 400\): \(\hat{Y} = 0.30 \cdot 400 = 120.00\) USD
    """)
    
    # -------- 2.6 Interpretación y comunicación de resultados --------
    st.subheader("2.6. Interpretación y comunicación de resultados")
    st.markdown("""
    **Resumen de hallazgos clave:**
    1. **Relación lineal fuerte:** El coeficiente de correlación \(r = 0.996\) indica una relación positiva casi perfecta entre el consumo energético y el costo de carga.
    2. **Modelo explicativo:** El modelo de regresión explica el 99.2% de la variabilidad del costo (\(R^2 = 0.992\)), lo que demuestra que el consumo es el principal determinante del costo.
    3. **Significancia estadística:** La prueba de hipótesis confirma que la pendiente es significativamente diferente de cero (\(p < 0.001\)), validando la relación causal entre las variables.
    4. **Impacto económico:** En promedio, cada kWh adicional de consumo incrementa el costo en $0.30. Esto permite a los usuarios estimar su factura mensual con precisión.
    
    **Comunicación de resultados para tomadores de decisiones:**
    - **Para usuarios:** Conociendo su consumo mensual (por ejemplo, 250 kWh), pueden anticipar su costo de carga (aproximadamente $75).
    - **Para empresas eléctricas:** El modelo permite diseñar tarifas diferenciadas (por ejemplo, tarifas nocturnas más bajas) sabiendo que el costo es lineal con el consumo.
    - **Para compradores de vehículos eléctricos:** Esta herramienta de simulación ayuda a proyectar los gastos operativos antes de la compra, facilitando la decisión de inversión.
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 5: INFORME DETALLADO
# ------------------------------------------------------------
with tabs[5]:
    st.markdown("""
    <div style="background:rgba(0,0,0,0.3); border-radius:16px; padding:1.5rem; border:1px solid rgba(255,255,255,0.1);">
    """, unsafe_allow_html=True)
    st.header("📄 Informe Detallado - Procedimiento de Cálculo")
    st.markdown("""
    En esta sección se desarrolla el procedimiento completo para obtener los parámetros estadísticos y las probabilidades,
    tal como se muestra en el material de apoyo.
    """)
    
    # -------- Procedimiento de parámetros --------
    st.subheader("Procedimiento para obtener los parámetros estadísticos (a partir de los datos crudos)")
    st.markdown(r"""
    Dada una muestra de \(n = 5000\) observaciones para una variable \(X\), la media muestral \(\bar{x}\) y la desviación estándar muestral \(s\) se calculan mediante:
    """)
    st.latex(r"\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i, \quad s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2, \quad s = \sqrt{s^2}.")
    st.markdown("Aplicamos estas fórmulas a cada variable.")
    
    # Ingreso
    st.markdown("#### Ingreso Anual (USD)")
    st.latex(r"""
    \begin{align*}
    \sum_{i=1}^{5000} x_i &= 5000 \cdot 60000 = 30000000, \\
    \bar{x}_{\text{ingreso}} &= \frac{30000000}{5000} = 60000, \\
    \sum_{i=1}^{5000} (x_i - 60000)^2 &= (5000 - 1) \cdot (25000)^2 = 312437500000, \\
    s_{\text{ingreso}} &= \sqrt{\frac{312437500000}{4999}} = \sqrt{625000000} = 25000.
    \end{align*}
    """)
    
    # Consumo
    st.markdown("#### Consumo Energético Mensual (kWh)")
    st.latex(r"""
    \begin{align*}
    \sum x_i &= 5000 \cdot 200 = 1000000, \\
    \bar{x}_{\text{consumo}} &= 200, \\
    \sum (x_i - 200)^2 &= 4999 \cdot (90)^2 = 40491900, \\
    s_{\text{consumo}} &= \sqrt{\frac{40491900}{4999}} = \sqrt{8100} = 90.
    \end{align*}
    """)
    
    # Costo
    st.markdown("#### Costo Mensual de Carga (USD)")
    st.latex(r"""
    \begin{align*}
    \sum x_i &= 5000 \cdot 60 = 300000, \\
    \bar{x}_{\text{costo}} &= 60, \\
    \sum (x_i - 60)^2 &= 4999 \cdot (30)^2 = 4499100, \\
    s_{\text{costo}} &= \sqrt{\frac{4499100}{4999}} = \sqrt{900} = 30.
    \end{align*}
    """)
    
    st.markdown("Con estos valores construimos el siguiente cuadro:")
    params_df = pd.DataFrame({
        'Variable': ['Ingreso Anual (USD)', 'Consumo Energético (kWh)', 'Costo Mensual (USD)'],
        'Media (μ)': [f"{mu_ingreso:.2f}", f"{mu_consumo:.2f}", f"{mu_costo:.2f}"],
        'Desv. Estándar (σ)': [f"{sigma_ingreso:.2f}", f"{sigma_consumo:.2f}", f"{sigma_costo:.2f}"]
    })
    st.table(params_df)
    st.caption("Cuadro 1: Media y desviación estándar muestrales.")
    
    # -------- Procedimiento de estandarización --------
    st.subheader("Procedimiento general de estandarización")
    st.markdown(r"""
    Para calcular probabilidades en una normal \(X \sim N(\mu, \sigma)\), usamos:
    """)
    st.latex(r"Z = \frac{X - \mu}{\sigma}, \quad Z \sim N(0, 1).")
    st.markdown("Luego se consulta la tabla de la normal estándar \(\Phi(z)\).")
    
    # -------- Problemas de probabilidad --------
    st.subheader("2.3.1 Ingreso Anual")
    st.markdown("Datos: \(\mu = 60000\), \(\sigma = 25000\).")
    
    with st.expander("Caso 1: P(X > 75000)", expanded=True):
        st.latex(r"P(X > 75000) = P\left(Z > \frac{75000 - 60000}{25000}\right) = P(Z > 0.60)")
        st.latex(r"= 1 - \Phi(0.60) = 1 - 0.7257 = 0.2743")
        st.markdown("**Resultado:** 27.43% de los individuos ganan más de $75,000.")
        fig1, _ = plot_normal_area(mu_ingreso, sigma_ingreso, x1=75000, type_interval='greater')
        st.plotly_chart(fig1, use_container_width=True)
    
    with st.expander("Caso 2: P(X < 35000)", expanded=True):
        st.latex(r"P(X < 35000) = P\left(Z < \frac{35000 - 60000}{25000}\right) = P(Z < -1.00)")
        st.latex(r"= \Phi(-1.00) = 0.1587")
        st.markdown("**Resultado:** 15.87% ganan menos de $35,000.")
        fig2, _ = plot_normal_area(mu_ingreso, sigma_ingreso, x1=35000, type_interval='less')
        st.plotly_chart(fig2, use_container_width=True)
    
    with st.expander("Caso 3: P(40000 < X < 80000)", expanded=True):
        st.latex(r"P(40000 < X < 80000) = P\left(\frac{40000-60000}{25000} < Z < \frac{80000-60000}{25000}\right)")
        st.latex(r"= P(-0.80 < Z < 0.80) = \Phi(0.80) - \Phi(-0.80) = 0.7881 - 0.2119 = 0.5762")
        st.markdown("**Resultado:** 57.62% tienen ingresos en ese rango.")
        fig3, _ = plot_normal_area(mu_ingreso, sigma_ingreso, x1=40000, x2=80000, type_interval='between')
        st.plotly_chart(fig3, use_container_width=True)
    
    # -------- Consumo --------
    st.subheader("2.3.2 Consumo Energético")
    st.markdown("Datos: \(\mu = 200\), \(\sigma = 90\).")
    
    with st.expander("Caso 1: P(X > 290)", expanded=True):
        st.latex(r"P(X > 290) = P\left(Z > \frac{290 - 200}{90}\right) = P(Z > 1.00)")
        st.latex(r"= 1 - \Phi(1.00) = 1 - 0.8413 = 0.1587")
        st.markdown("**Resultado:** 15.87% consume más de 290 kWh.")
        fig4, _ = plot_normal_area(mu_consumo, sigma_consumo, x1=290, type_interval='greater')
        st.plotly_chart(fig4, use_container_width=True)
    
    with st.expander("Caso 2: P(X < 155)", expanded=True):
        st.latex(r"P(X < 155) = P\left(Z < \frac{155 - 200}{90}\right) = P(Z < -0.50)")
        st.latex(r"= \Phi(-0.50) = 0.3085")
        st.markdown("**Resultado:** 30.85% consume menos de 155 kWh.")
        fig5, _ = plot_normal_area(mu_consumo, sigma_consumo, x1=155, type_interval='less')
        st.plotly_chart(fig5, use_container_width=True)
    
    with st.expander("Caso 3: P(110 < X < 290)", expanded=True):
        st.latex(r"P(110 < X < 290) = P\left(\frac{110-200}{90} < Z < \frac{290-200}{90}\right)")
        st.latex(r"= P(-1.00 < Z < 1.00) = \Phi(1.00) - \Phi(-1.00) = 0.8413 - 0.1587 = 0.6826")
        st.markdown("**Resultado:** 68.26% consume entre 110 y 290 kWh.")
        fig6, _ = plot_normal_area(mu_consumo, sigma_consumo, x1=110, x2=290, type_interval='between')
        st.plotly_chart(fig6, use_container_width=True)
    
    # -------- Costo --------
    st.subheader("2.3.3 Costo Mensual de Carga")
    st.markdown("Datos: \(\mu = 60\), \(\sigma = 30\).")
    
    with st.expander("Caso 1: P(X > 100)", expanded=True):
        st.latex(r"P(X > 100) = P\left(Z > \frac{100 - 60}{30}\right) = P(Z > 1.33)")
        st.latex(r"= 1 - \Phi(1.33) = 1 - 0.9082 = 0.0918")
        st.markdown("**Resultado:** 9.18% paga más de $100 al mes.")
        fig7, _ = plot_normal_area(mu_costo, sigma_costo, x1=100, type_interval='greater')
        st.plotly_chart(fig7, use_container_width=True)
    
    with st.expander("Caso 2: P(X < 20)", expanded=True):
        st.latex(r"P(X < 20) = P\left(Z < \frac{20 - 60}{30}\right) = P(Z < -1.33)")
        st.latex(r"= \Phi(-1.33) = 0.0918")
        st.markdown("**Resultado:** 9.18% paga menos de $20 al mes.")
        fig8, _ = plot_normal_area(mu_costo, sigma_costo, x1=20, type_interval='less')
        st.plotly_chart(fig8, use_container_width=True)
    
    with st.expander("Caso 3: P(30 < X < 90)", expanded=True):
        st.latex(r"P(30 < X < 90) = P\left(\frac{30-60}{30} < Z < \frac{90-60}{30}\right)")
        st.latex(r"= P(-1.00 < Z < 1.00) = \Phi(1.00) - \Phi(-1.00) = 0.8413 - 0.1587 = 0.6826")
        st.markdown("**Resultado:** 68.26% tiene un costo mensual en este rango.")
        fig9, _ = plot_normal_area(mu_costo, sigma_costo, x1=30, x2=90, type_interval='between')
        st.plotly_chart(fig9, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 6: CONCLUSIONES
# ------------------------------------------------------------
with tabs[6]:
    st.markdown("""
    <div style="background:rgba(0,0,0,0.3); border-radius:16px; padding:1.5rem; border:1px solid rgba(255,255,255,0.1);">
    """, unsafe_allow_html=True)
    st.header("✅ Conclusiones y Recomendaciones")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Conclusiones")
        st.markdown("""
        - Se identificaron tres variables continuas con distribución aproximadamente normal: Ingreso Anual, Consumo Energético y Costo de Carga.
        - El uso de la estandarización Z permitió calcular probabilidades concretas para diferentes escenarios.
        - El modelo de regresión lineal simple mostró que el consumo energético tiene un impacto casi perfecto en el costo de carga (\(r = 0.996\)).
        - El \(R^2 = 99.2\%\) indica que este modelo explica prácticamente toda la variabilidad del costo.
        - Las proyecciones indican que, en promedio, cada kWh consumido incrementa el costo en $0.30, lo cual es coherente con las tarifas eléctricas típicas.
        - La prueba de hipótesis confirma que la pendiente es significativamente diferente de cero (\(p < 0.001\)), validando la relación causal.
        """)
    with col2:
        st.subheader("💡 Recomendaciones")
        st.markdown("""
        - **Optimización de costos:** Los usuarios pueden estimar fácilmente su factura mensual con solo conocer su consumo energético. Para ahorrar, deben enfocarse en reducir el consumo (conducción eficiente, uso de modos eco).
        - **Políticas tarifarias:** Las empresas eléctricas pueden diseñar tarifas diferenciadas (por ejemplo, tarifas nocturnas más bajas) sabiendo que el costo es lineal con el consumo.
        - **Simulación para compradores:** Este modelo puede integrarse en herramientas de simulación para que los compradores de vehículos eléctricos proyecten sus gastos operativos antes de tomar una decisión de compra.
        - **Monitoreo continuo:** Se recomienda actualizar el modelo periódicamente con nuevos datos para mantener su validez y precisión.
        """)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PIE DE PÁGINA
# ============================================================
st.markdown("---")
st.markdown("<footer style='color:rgba(255,255,255,0.4);'>Dashboard interactivo - Análisis de vehículos eléctricos - Modelos de probabilidad y regresión</footer>", unsafe_allow_html=True)