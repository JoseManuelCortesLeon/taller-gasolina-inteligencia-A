import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Configuración de la interfaz
st.set_page_config(
    page_title="API & Endpoint Predicción Gasolina",
    page_icon="⛽",
    layout="centered"
)

st.title("⛽ Endpoint de Predicción del Precio de Gasolina")
st.markdown("---")

# 2. Carga segura de componentes (.joblib)
@st.cache_resource
def cargar_endpoint():
    try:
        modelo = joblib.load('modelo_gasolina.joblib')
        scaler = joblib.load('scaler_gasolina.joblib')
        info = joblib.load('info_despliegue.joblib')
        return modelo, scaler, info, True
    except Exception as e:
        return None, None, None, False

modelo, scaler, info, estado_carga = cargar_endpoint()

if not estado_carga:
    st.error("⚠️ Error de conexión con el Endpoint: No se encontraron los archivos del modelo (.joblib).")
    st.info("Asegúrate de haber ejecutado completamente el cuaderno Jupyter antes de iniciar la App.")
    st.stop()

# 3. Formulario de entrada (Input Data Validation)
st.sidebar.header("📥 Parámetros de Entrada (Payload)")

def solicitar_parametros():
    p_petroleo = st.sidebar.number_input("Precio Petróleo (USD/Barril)", min_value=1.0, max_value=300.0, value=75.0, step=0.5)
    t_cambio = st.sidebar.number_input("Tipo de Cambio (USD/COP)", min_value=500.0, max_value=10000.0, value=3900.0, step=10.0)
    demanda = st.sidebar.number_input("Demanda (miles barriles/día)", min_value=100.0, max_value=50000.0, value=9500.0, step=50.0)
    opep = st.sidebar.number_input("Producción OPEP (millones/día)", min_value=1.0, max_value=100.0, value=28.0, step=0.1)
    inflacion = st.sidebar.number_input("Inflación (%)", min_value=-10.0, max_value=100.0, value=3.0, step=0.1)
    pib = st.sidebar.number_input("PIB (%)", min_value=-30.0, max_value=30.0, value=3.0, step=0.1)
    eventos = st.sidebar.selectbox("Eventos Geopolíticos", options=[0, 1], format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")
    impuestos = st.sidebar.number_input("Impuestos Combustibles (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    
    return {
        'Precio_Petroleo_USD_Barril': p_petroleo,
        'Tipo_Cambio_USD_COP': t_cambio,
        'Demanda_miles_barriles_dia': demanda,
        'Produccion_OPEP_millones_dia': opep,
        'Inflacion_Porcentaje': inflacion,
        'PIB_Porcentaje': pib,
        'Eventos_Geopoliticos': eventos,
        'Impuestos_Combustibles_Porcentaje': impuestos
    }

input_data = solicitar_parametros()

# 4. Procesamiento del Endpoint al presionar el botón
if st.button("🚀 Consumir Endpoint (Predecir)", type="primary"):
    try:
        # a. Convertir datos de entrada a DataFrame
        df_input = pd.DataFrame([input_data])
        
        # b. Escalar/Normalizar variables
        df_scaled = scaler.transform(df_input)
        
        # c. Inferencia del modelo
        prediccion_puntual = float(modelo.predict(df_scaled)[0])
        
        # d. Cálculo del Intervalo de Confianza (95%)
        margen = float(info['margen_error_95'])
        limite_inf = max(0.0, prediccion_puntual - margen) # Previene valores negativos
        limite_sup = prediccion_puntual + margen
        
        # e. Visualización mejorada del Endpoint (Métricas / KPIs)
        st.subheader("📤 Respuesta del Endpoint (Output)")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicción Estimada", f"${prediccion_puntual:.2f} USD")
        col2.metric("Límite Inferior (95%)", f"${limite_inf:.2f} USD")
        col3.metric("Límite Superior (95%)", f"${limite_sup:.2f} USD")
        
        st.markdown("---")
        
        # f. Mostrar respuesta formateada como Objeto JSON / Estructurado
        st.subheader("📋 Respuesta Estructurada (JSON Payload)")
        respuesta_json = {
            "status": "200 OK",
            "prediction": {
                "unit": "USD/Galon",
                "estimated_price": round(prediccion_puntual, 4),
                "confidence_interval_95": {
                    "lower_bound": round(limite_inf, 4),
                    "upper_bound": round(limite_sup, 4),
                    "margin_of_error": round(margen, 4)
                }
            },
            "input_parameters": input_data
        }
        st.json(respuesta_json)
        
    except Exception as err:
        st.error(f"❌ Error al procesar la solicitud en el Endpoint: {err}")