import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------
st.set_page_config(
    page_title="Deep Digit AI | MNIST Core",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# ESTILOS CSS: GRADIENTES COHERENTES Y MÚLTIPLES ANIMACIONES
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&family=JetBrains+Mono:wght@500;700&display=swap');

    /* Fondo animado fluido (deep midnight & indigo/cyan mist) */
    .stApp {
        background: linear-gradient(135deg, #070913 0%, #0d122b 50%, #060814 100%);
        background-size: 200% 200%;
        animation: fluidBackground 14s ease infinite;
        color: #e2e8f0;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    @keyframes fluidBackground {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Título principal con gradiente cian-azul-índigo animado */
    .hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        text-align: center;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #3b82f6, #38bdf8);
        background-size: 250% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shineText 5s linear infinite;
        margin-bottom: 0.2rem;
    }

    @keyframes shineText {
        to { background-position: 250% center; }
    }

    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }

    /* Contenedor del lienzo centrado al 80% con aura de pulso */
    .canvas-card {
        background: rgba(13, 18, 43, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 24px;
        padding: 1.5rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        box-shadow: 0 10px 40px -10px rgba(14, 165, 233, 0.2);
        animation: pulseAura 4s ease-in-out infinite alternate;
        transition: transform 0.35s ease, border-color 0.35s ease, box-shadow 0.35s ease;
    }

    @keyframes pulseAura {
        0% {
            border-color: rgba(56, 189, 248, 0.25);
            box-shadow: 0 0 25px rgba(56, 189, 248, 0.15);
        }
        100% {
            border-color: rgba(99, 102, 241, 0.6);
            box-shadow: 0 0 45px rgba(99, 102, 241, 0.3);
        }
    }

    .canvas-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.85);
        box-shadow: 0 15px 50px rgba(56, 189, 248, 0.35);
    }

    /* Ajuste responsivo del canvas */
    div[data-testid="stCanvas"] {
        display: flex;
        justify-content: center;
        border-radius: 16px;
        overflow: hidden;
    }

    div[data-testid="stCanvas"] > canvas {
        border-radius: 16px !important;
        box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.8);
    }

    /* Botones primarios y secundarios con gradiente dinámico */
    div.stButton > button {
        border-radius: 12px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Botón Predecir */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%);
        color: #ffffff;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.4);
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 30px rgba(56, 189, 248, 0.6);
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    }

    /* Botón Limpiar */
    div.stButton > button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.6);
        color: #94a3b8;
    }

    div.stButton > button[kind="secondary"]:hover {
        transform: translateY(-2px);
        background: rgba(51, 65, 85, 0.8);
        color: #f8fafc;
    }

    /* Tarjetas de resultados con animación de entrada */
    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.5s ease backwards;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.5);
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(180deg, #ffffff 30%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# CARGA DE MODELO CACHEADA
# ---------------------------------------------------------
@st.cache_resource
def load_mnist_model():
    model_files = ['mnist_model.keras', 'mnist_model.h5']
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                return keras.models.load_model(model_path), model_path
            except Exception:
                continue
    return None, None

model, loaded_path = load_mnist_model()

# ---------------------------------------------------------
# SIDEBAR INFORMATIVA
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Entorno de Ejecución")
    st.markdown(f"**TensorFlow:** `{tf.__version__}`")
    st.markdown(f"**Keras:** `{keras.__version__}`")
    if loaded_path:
        st.success(f"Archivo: `{loaded_path}`")
    else:
        st.warning("⚠️ Sin modelo en disco. Sube `mnist_model.keras` a la raíz.")

    stroke_width = st.slider("Grosor del Trazo", min_value=12, max_value=36, value=24)

# ---------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------
st.markdown('<div class="hero-title">RECONOCEDOR NEURONAL MNIST</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Traza un dígito del 0 al 9 en el lienzo central de alta precisión</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SECCIÓN PRINCIPAL: LIENZO CENTRADO AL 80%
# ---------------------------------------------------------
# 10% margen izq | 80% centro | 10% margen der
col_left, col_canvas, col_right = st.columns([1, 8, 1])

with col_canvas:
    st.markdown('<div class="canvas-card">', unsafe_allow_html=True)
    
    # Lienzo con ancho panorámico de 840px
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=840,
        height=420,
        drawing_mode="freedraw",
        key="mnist_wide_canvas",
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Barra interactiva de acciones alineada
    st.write("")
    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        predict_btn = st.button("⚡ ANALIZAR DÍGITO", type="primary", use_container_width=True)
    with btn_col2:
        clear_btn = st.button("LIMPIAR", type="secondary", use_container_width=True)

    if clear_btn:
        st.rerun()

# ---------------------------------------------------------
# PREPROCESAMIENTO INTELIGENTE Y PREDICCIÓN
# ---------------------------------------------------------
def preprocess_canvas_image(raw_image_data: np.ndarray) -> tuple[Image.Image, np.ndarray] | tuple[None, None]:
    """
    Convierte trazo en escala de grises, localiza el contorno del dibujo (Bounding Box)
    y lo centra en una caja cuadrada con padding (20x20 contenido en 28x28) como el dataset original.
    """
    img_gray = Image.fromarray(raw_image_data.astype("uint8")).convert("L")
    bbox = img_gray.getbbox()
    
    if not bbox:
        return None, None
    
    # Extraer el trazo exacto dibujado
    cropped = img_gray.crop(bbox)
    
    # Normalizar a caja cuadrada con margen de 4 píxeles (equivalente MNIST 20x20 en 28x28)
    max_side = max(cropped.width, cropped.height)
    square_side = int(max_side * 1.4)
    canvas_square = Image.new("L", (square_side, square_side), color=0)
    
    paste_x = (square_side - cropped.width) // 2
    paste_y = (square_side - cropped.height) // 2
    canvas_square.paste(cropped, (paste_x, paste_y))
    
    # Redimensionar a resolución nativa del modelo (28x28)
    image_28x28 = canvas_square.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(image_28x28, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return image_28x28, img_array


if predict_btn:
    if model is None:
        st.error("❌ No hay un modelo cargado. Coloca `mnist_model.keras` o `.h5` en la raíz del proyecto.")
    elif canvas_result.image_data is None or np.max(canvas_result.image_data) == 0:
        st.warning("⚠️ El lienzo está vacío. Por favor dibuja un dígito.")
    else:
        with col_canvas:
            with st.spinner("Decodificando activaciones..."):
                processed_img, tensor_input = preprocess_canvas_image(canvas_result.image_data)
                
                if tensor_input is None:
                    st.warning("⚠️ No se detectó ningún trazo válido.")
                else:
                    prediction = model.predict(tensor_input, verbose=0)
                    probabilities = prediction[0]
                    digit = int(np.argmax(probabilities))
                    confidence = float(np.max(probabilities) * 100)
                    
                    sorted_indices = np.argsort(probabilities)[::-1]
                    alt_digit = int(sorted_indices[1])
                    alt_confidence = float(probabilities[alt_digit] * 100)
                    
                    # Panel de Métricas animadas
                    st.write("")
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Predicción Principal</div>
                                <div class="metric-value">{digit}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with m_col2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Certeza</div>
                                <div class="metric-value">{confidence:.1f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with m_col3:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Segunda Opción</div>
                                <div class="metric-value">{alt_digit} <span style="font-size:1.1rem; color:#94a3b8;">({alt_confidence:.1f}%)</span></div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Gráfica de Distribución de Probabilidad
                    st.write("")
                    st.markdown("#### 📊 Vector de Probabilidad por Clase")
                    prob_df = pd.DataFrame({
                        "Dígito": [str(i) for i in range(10)],
                        "Probabilidad (%)": probabilities * 100
                    })
                    st.bar_chart(prob_df.set_index("Dígito"), color="#38bdf8")
                    
                    # Comparativa Visual del Preprocesamiento
                    with st.expander("🔬 Inspección de Entrada a la Red (28x28)"):
                        sub_c1, sub_c2 = st.columns([1, 3])
                        with sub_c1:
                            st.image(processed_img, caption="Tensor Redimensionado (28x28)", width=130)
                        with sub_c2:
                            st.markdown(
                                f"""
                                - **Resolución de entrada:** `28 x 28 x 1`
                                - **Rango de activación:** `[{tensor_input.min():.2f}, {tensor_input.max():.2f}]`
                                - **Centrado dinámico:** Aplicado mediante detección de cuadro delimitador.
                                """
                            )
