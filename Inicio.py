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
# LÍNEA BASE VISUAL: GRADIENTES DE CARBONO, FLUIDOS Y TRAZO ANIMADO DEL "3"
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #050507 !important;
        color: #f4f4f5 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        overflow-x: hidden;
    }

    [data-testid="stHeader"] { background: transparent !important; }

    /* 1. Fondo de gradiente continuo de carbono */
    .stApp {
        background: linear-gradient(135deg, #050507 0%, #0d0d12 30%, #15151c 60%, #060608 100%) !important;
        background-size: 300% 300% !important;
        animation: carbonShift 18s ease infinite alternate !important;
    }

    @keyframes carbonShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 2. Esferas orgánicas fluidas de tono platino */
    .fluid-carbon-1 {
        position: fixed;
        top: 15%;
        left: -120px;
        width: 520px;
        height: 520px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.08) 0%, rgba(113, 113, 122, 0.03) 60%, transparent 75%);
        border-radius: 40% 60% 70% 30% / 50% 60% 40% 50%;
        filter: blur(85px);
        animation: carbonDrift1 22s infinite alternate ease-in-out;
        pointer-events: none;
        z-index: 0;
    }

    .fluid-carbon-2 {
        position: fixed;
        bottom: 5%;
        right: -100px;
        width: 560px;
        height: 560px;
        background: radial-gradient(circle, rgba(161, 161, 170, 0.06) 0%, rgba(255, 255, 255, 0.02) 60%, transparent 75%);
        border-radius: 60% 40% 50% 50% / 40% 50% 60% 50%;
        filter: blur(90px);
        animation: carbonDrift2 25s infinite alternate ease-in-out;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes carbonDrift1 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(120px, 70px) scale(1.15); }
        100% { transform: translate(40px, 140px) scale(0.95); }
    }

    @keyframes carbonDrift2 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(-100px, -70px) scale(1.2); }
        100% { transform: translate(-30px, -120px) scale(0.9); }
    }

    .brand-signature {
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #71717a;
        margin-top: 0.2rem;
    }

    .title {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #fafafa;
        margin-top: 0.2rem;
        margin-bottom: 0.8rem;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
    }

    /* RESTAURACIÓN DEL NÚMERO 3 ANIMADO */
    .guide-chip {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1.5rem;
        margin: 0 auto 1.4rem auto;
        padding: 8px 18px;
        background: rgba(13, 13, 16, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        width: fit-content;
        font-size: 0.82rem;
        color: #a1a1aa;
        backdrop-filter: blur(10px);
    }

    .stroke-demo {
        width: 32px;
        height: 32px;
    }

    .animated-digit-3 {
        stroke-dasharray: 100;
        stroke-dashoffset: 100;
        animation: traceThree 3.5s cubic-bezier(0.4, 0, 0.2, 1) infinite;
    }

    @keyframes traceThree {
        0% { stroke-dashoffset: 100; opacity: 0.2; }
        40% { stroke-dashoffset: 0; opacity: 1; }
        80% { stroke-dashoffset: 0; opacity: 1; }
        100% { stroke-dashoffset: 100; opacity: 0.2; }
    }

    /* CENTRADO PERFECTO DEL LIENZO SIN MARCOS ANIDADOS */
    div[data-testid="stCanvas"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        background: transparent !important;
        margin: 0 auto !important;
    }

    div[data-testid="stCanvas"] > div {
        margin: 0 auto !important;
        background: transparent !important;
    }

    div[data-testid="stCanvas"] canvas {
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.95), 0 0 25px rgba(255, 255, 255, 0.06) !important;
        background-color: #000000 !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    }

    div[data-testid="stCanvas"] canvas:hover {
        border-color: rgba(255, 255, 255, 0.6) !important;
        box-shadow: 0 12px 45px rgba(0, 0, 0, 0.95), 0 0 35px rgba(255, 255, 255, 0.12) !important;
    }

    div.stButton > button {
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 0.7rem 1.4rem !important;
        transition: all 0.2s ease !important;
    }

    .stat-card {
        background: rgba(13, 13, 17, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .stat-label {
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #71717a;
    }
    .stat-val {
        font-size: 2.4rem;
        font-weight: 500;
        color: #ffffff;
        margin: 2px 0;
        line-height: 1;
    }
    </style>

    <div class="fluid-carbon-1"></div>
    <div class="fluid-carbon-2"></div>
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
        if not os.path.exists(base): continue
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
        if os.path.getsize(fp) < 2048: continue
        try:
            return tf.keras.models.load_model(fp, compile=False), os.path.basename(fp), None
        except Exception:
            try:
                return tf.keras.models.load_model(fp), os.path.basename(fp), None
            except Exception:
                pass
    return None, None, "Error al cargar la red neuronal."

model, filename, load_err = load_network()

st.markdown('<div class="brand-signature">Krov Industries · Reconocimiento Óptico</div>', unsafe_allow_html=True)
st.markdown('<div class="title">Reconocedor de Dígitos</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# NÚMERO 3 ANIMADO COMO GUÍA VISUAL
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="guide-chip">
        <svg class="stroke-demo" viewBox="0 0 40 40" fill="none">
            <path class="animated-digit-3" d="M 12 10 C 24 10, 24 20, 14 20 C 26 20, 26 32, 10 32" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round"/>
        </svg>
        <span>1. Dibuja un dígito centrado (0 al 9)</span>
        <span style="color:#3f3f46;">→</span>
        <span>2. Pulsa Analizar Trazo</span>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# LIENZO CENTRADO SIN RECORTAR DIMENSIONES (640x340px)
# -----------------------------------------------------------------------------
_, col_canvas, _ = st.columns([1, 4.5, 1])

with col_canvas:
    canvas_data = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#ffffff",
        background_color="#000000",
        width=640,
        height=340,
        drawing_mode="freedraw",
        key="digit_canvas",
    )

    st.write("")
    b_col1, b_col2 = st.columns([3, 1])
    with b_col1:
        run_btn = st.button("Analizar trazo", type="primary", use_container_width=True)
    with b_col2:
        clear_btn = st.button("Limpiar", type="secondary", use_container_width=True)

    if clear_btn:
        st.rerun()

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
        with col_canvas:
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
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.markdown(f'<div class="stat-card"><div class="stat-label">Dígito Identificado</div><div class="stat-val">{pred}</div></div>', unsafe_allow_html=True)
                with res_col2:
                    st.markdown(f'<div class="stat-card"><div class="stat-label">Certeza</div><div class="stat-val">{conf:.0f}%</div></div>', unsafe_allow_html=True)
                with res_col3:
                    st.markdown(f'<div class="stat-card"><div class="stat-label">Segunda Opción</div><div class="stat-val">{second_pred} <span style="font-size:0.9rem; color:#71717a;">({second_conf:.0f}%)</span></div></div>', unsafe_allow_html=True)

                st.write("")
                chart_data = pd.DataFrame({
                    "Dígito": [str(i) for i in range(10)],
                    "Probabilidad (%)": probs * 100,
                }).set_index("Dígito")
                st.bar_chart(chart_data, color="#fafafa", height=180)
