import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# -----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Reconocedor de Dígitos // Krov Industries",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# ESTILOS VISUALES, LÁMPARA DE LAVA EN CSS (SIN LAG) Y CENTRADO EXACTO
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* 1. Fondo fluido de lámpara de lava acelerado por GPU */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #06070b !important;
        background-image:
            radial-gradient(circle at 15% 25%, rgba(0, 240, 255, 0.16) 0%, transparent 45%),
            radial-gradient(circle at 85% 30%, rgba(168, 85, 247, 0.18) 0%, transparent 50%),
            radial-gradient(circle at 50% 80%, rgba(236, 72, 153, 0.14) 0%, transparent 45%),
            linear-gradient(135deg, #050609 0%, #0d0f17 50%, #06070b 100%) !important;
        background-size: 200% 200% !important;
        animation: lavaMove 14s ease-in-out infinite alternate !important;
        color: #f4f4f5 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        overflow-x: hidden;
    }

    @keyframes lavaMove {
        0% { background-position: 0% 0%; }
        50% { background-position: 100% 100%; }
        100% { background-position: 0% 100%; }
    }

    [data-testid="stHeader"] { background: transparent !important; }

    /* 2. Contenedor centrado: elimina el recuadro oscuro lateral y centra todo */
    .block-container {
        max-width: 640px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        margin: 0 auto !important;
    }

    .brand-signature {
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #71717a;
    }

    .title {
        text-align: center;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #ffffff;
        margin-top: 0.1rem;
        margin-bottom: 0.8rem;
        text-shadow: 0 0 25px rgba(0, 240, 255, 0.35);
    }

    /* 3. Guía interactiva con el número 3 animado */
    .guide-box {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        margin: 0 auto 1.4rem auto;
        padding: 8px 20px;
        background: rgba(14, 16, 24, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        width: fit-content;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }

    .animated-three-path {
        stroke-dasharray: 100;
        stroke-dashoffset: 100;
        animation: traceThree 3s ease-in-out infinite;
    }

    @keyframes traceThree {
        0% { stroke-dashoffset: 100; opacity: 0.2; }
        50% { stroke-dashoffset: 0; opacity: 1; }
        100% { stroke-dashoffset: 0; opacity: 1; }
    }

    /* 4. Estilos y centrado del iframe de dibujo */
    iframe[title*="canvas"] {
        margin: 0 auto !important;
        display: block !important;
        border: 2px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 10px !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.8), 0 0 20px rgba(0, 240, 255, 0.1) !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    }

    iframe[title*="canvas"]:hover {
        border-color: #00f0ff !important;
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.9), 0 0 30px rgba(0, 240, 255, 0.3) !important;
    }

    /* 5. Botones táctiles de alta visibilidad */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00f0ff 0%, #7928ca 100%) !important;
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.75rem 1.4rem !important;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 35px rgba(0, 240, 255, 0.6) !important;
    }

    div.stButton > button[kind="secondary"] {
        background: rgba(18, 20, 30, 0.85) !important;
        color: #e4e4e7 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 6px !important;
        padding: 0.75rem 1.4rem !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        transform: translateY(-2px) !important;
        border-color: #ffffff !important;
        color: #ffffff !important;
    }

    /* 6. Tarjetas de métricas */
    .stat-card {
        background: rgba(14, 16, 24, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
    }
    .stat-label {
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
    }
    .stat-val {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 2px 0;
        line-height: 1;
        text-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# CARGA DEL MODELO NEURONAL
# -----------------------------------------------------------------------------
def locate_weights():
    search_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."]
    matches = []
    for base in set(search_dirs):
        if not os.path.exists(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith((".h5", ".keras")):
                    matches.append(os.path.join(root, f))
    return list(dict.fromkeys(matches))


@st.cache_resource(show_spinner=False)
def load_network():
    files = locate_weights()
    if not files:
        return None, None, "No se encontró ningún archivo de pesos (.h5 o .keras)."
    for fp in files:
        if os.path.getsize(fp) < 2048:
            continue
        try:
            return tf.keras.models.load_model(fp, compile=False), os.path.basename(fp), None
        except Exception:
            try:
                return tf.keras.models.load_model(fp), os.path.basename(fp), None
            except Exception:
                pass
    return None, None, "Error al inicializar la red neuronal."


model, filename, load_err = load_network()

# -----------------------------------------------------------------------------
# ENCABEZADO Y GUÍA VISUAL CON NÚMERO 3 ANIMADO
# -----------------------------------------------------------------------------
st.markdown('<div class="brand-signature">Krov Industries · Reconocimiento Óptico</div>', unsafe_allow_html=True)
st.markdown('<div class="title">Reconocedor de Dígitos</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="guide-box">
        <svg width="30" height="30" viewBox="0 0 40 40" style="display:inline-block; vertical-align:middle;">
            <path class="animated-three-path" d="M 12 10 C 25 10, 25 19, 15 20 C 27 21, 27 32, 11 32" fill="none" stroke="#00f0ff" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span style="font-size: 0.85rem; color: #e4e4e7;">1. Dibuja un dígito centrado (0 al 9)</span>
        <span style="color: #64748b;">→</span>
        <span style="font-size: 0.85rem; color: #00f0ff;">2. Pulsa Analizar Trazo</span>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# LIENZO CENTRADO Y SIMÉTRICO (540x320px)
# -----------------------------------------------------------------------------
canvas_data = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#ffffff",
    background_color="#000000",
    width=540,
    height=320,
    drawing_mode="freedraw",
    key="mnist_canvas_simple",
)

st.write("")
btn_col1, btn_col2 = st.columns([3, 1])
with btn_col1:
    run_btn = st.button("Analizar trazo", type="primary", use_container_width=True)
with btn_col2:
    clear_btn = st.button("Limpiar", type="secondary", use_container_width=True)

if clear_btn:
    st.rerun()

# -----------------------------------------------------------------------------
# PROCESAMIENTO E INFERENCIA
# -----------------------------------------------------------------------------
def prepare_drawing(raw_matrix: np.ndarray):
    img = Image.fromarray(raw_matrix.astype("uint8")).convert("L")
    box = img.getbbox()
    if not box:
        return None, None
    cropped = img.crop(box)
    size = int(max(cropped.width, cropped.height) * 1.4)
    canvas = Image.new("L", (size, size), color=0)
    canvas.paste(cropped, ((size - cropped.width) // 2, (size - cropped.height) // 2))
    digit_img = canvas.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(digit_img, dtype=np.float32) / 255.0
    return digit_img, arr.reshape(1, 28, 28, 1)

if run_btn:
    if model is None:
        st.error(f"No fue posible cargar el modelo: {load_err}")
    elif canvas_data.image_data is None or np.max(canvas_data.image_data) == 0:
        st.warning("Dibuja un número en el lienzo antes de presionar Analizar.")
    else:
        img_28, tensor = prepare_drawing(canvas_data.image_data)
        if tensor is None:
            st.warning("No se detectó ningún trazo en el lienzo.")
        else:
            probs = model.predict(tensor, verbose=0)[0]
            pred = int(np.argmax(probs))
            conf = float(np.max(probs) * 100)
            ranked = np.argsort(probs)[::-1]
            second_pred = int(ranked[1])
            second_conf = float(probs[second_pred] * 100)

            st.write("")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f'<div class="stat-card"><div class="stat-label">Dígito Identificado</div><div class="stat-val">{pred}</div></div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="stat-card"><div class="stat-label">Certeza</div><div class="stat-val">{conf:.0f}%</div></div>', unsafe_allow_html=True)
            with r3:
                st.markdown(f'<div class="stat-card"><div class="stat-label">Segunda Opción</div><div class="stat-val">{second_pred} <span style="font-size:0.85rem; color:#94a3b8;">({second_conf:.0f}%)</span></div></div>', unsafe_allow_html=True)

            st.write("")
            chart_data = pd.DataFrame({
                "Dígito": [str(i) for i in range(10)],
                "Probabilidad (%)": probs * 100,
            }).set_index("Dígito")
            st.bar_chart(chart_data, color="#00f0ff", height=180)
