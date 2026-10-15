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
# ESTILOS VISUALES LIMPIOS (SIN RECUADROS SECUNDARIOS DE FONDO)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #050507 !important;
        color: #f4f4f5 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    #MainMenu, footer, header { visibility: hidden; }

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
        font-size: 1.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #fafafa;
        margin-top: 0.2rem;
        margin-bottom: 0.8rem;
    }

    /* Supresión de contenedores secundarios o cuadrados grises detrás del canvas */
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
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 6px !important;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.95), 0 0 15px rgba(255, 255, 255, 0.04) !important;
        background-color: #000000 !important;
        transition: border-color 0.25s ease !important;
    }

    div[data-testid="stCanvas"] canvas:hover {
        border-color: rgba(255, 255, 255, 0.5) !important;
    }

    .guide-chip {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1.5rem;
        margin: 0 auto 1.2rem auto;
        padding: 6px 14px;
        background: #0d0d10;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        width: fit-content;
        font-size: 0.8rem;
        color: #a1a1aa;
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
        background: #0d0d11;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        padding: 1rem;
        text-align: center;
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

st.markdown(
    """
    <div class="guide-chip">
        <span>1. Dibuja un dígito centrado (0 al 9)</span>
        <span style="color:#3f3f46;">→</span>
        <span>2. Pulsa Analizar Trazo</span>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# LIENZO CENTRADO
# -----------------------------------------------------------------------------
_, col_canvas, _ = st.columns([1, 4, 1])

with col_canvas:
    canvas_data = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#ffffff",
        background_color="#000000",
        width=580,
        height=320,
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
