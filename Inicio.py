import os
import io
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DEL SISTEMA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MNIST / Inferencia Matricial",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# MOTOR VISUAL NOIR & ESTELA DE CURSOR INTERACTIVA
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Estructura base */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #08080a !important;
        color: #d4d4d8 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        letter-spacing: -0.01em;
    }

    /* Ocultar elementos nativos innecesarios */
    #MainMenu, footer, header { visibility: hidden; }

    /* Tipografía editorial */
    .noir-header {
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 0.25rem;
    }

    .noir-title {
        font-size: 1.65rem;
        font-weight: 500;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #f4f4f5;
        margin: 0;
    }

    .noir-subtitle {
        font-size: 0.82rem;
        font-weight: 400;
        letter-spacing: 0.08em;
        color: #71717a;
        margin-top: 0.5rem;
        margin-bottom: 2.5rem;
        text-transform: uppercase;
    }

    /* Marco del lienzo: sobrio, angulado, centrado al 80% */
    .canvas-frame {
        position: relative;
        background: #000000;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 4px;
        padding: 8px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.9);
        transition: border-color 0.4s ease, box-shadow 0.4s ease;
    }

    .canvas-frame:hover {
        border-color: rgba(255, 255, 255, 0.35);
        box-shadow: 0 0 35px rgba(255, 255, 255, 0.04);
    }

    div[data-testid="stCanvas"] {
        display: flex;
        justify-content: center;
    }

    div[data-testid="stCanvas"] > canvas {
        border-radius: 2px !important;
    }

    /* Botones sobrios tipo estudio */
    div.stButton > button {
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        padding: 0.75rem 1.4rem !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    div.stButton > button[kind="primary"] {
        background-color: #f4f4f5 !important;
        color: #09090b !important;
        border: 1px solid #f4f4f5 !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(255, 255, 255, 0.2) !important;
    }

    div.stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: #a1a1aa !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        background-color: rgba(255, 255, 255, 0.03) !important;
    }

    /* Métricas estilizadas monolíticas */
    .metric-panel {
        background: #0c0c0e;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        padding: 1.25rem;
        transition: border-color 0.3s ease;
    }

    .metric-panel:hover {
        border-color: rgba(255, 255, 255, 0.2);
    }

    .metric-caption {
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #71717a;
    }

    .metric-figure {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        font-size: 2.8rem;
        color: #ffffff;
        margin-top: 0.25rem;
        line-height: 1;
    }

    .metric-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #a1a1aa;
        margin-top: 0.5rem;
    }

    /* Sidebar sobria */
    section[data-testid="stSidebar"] {
        background-color: #070709 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    </style>

    <!-- Canvas para estela interactiva minimalista -->
    <canvas id="trailCanvas" style="position:fixed; top:0; left:0; width:100vw; height:100vh; pointer-events:none; z-index:99999;"></canvas>
    
    <script>
    (function() {
        const doc = window.parent.document;
        let canvas = doc.getElementById('trailCanvas');
        if (!canvas) {
            canvas = document.createElement('canvas');
            canvas.id = 'trailCanvas';
            canvas.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:99999;';
            doc.body.appendChild(canvas);
        }
        const ctx = canvas.getContext('2d');
        let width = canvas.width = window.parent.innerWidth;
        let height = canvas.height = window.parent.innerHeight;

        window.parent.addEventListener('resize', () => {
            width = canvas.width = window.parent.innerWidth;
            height = canvas.height = window.parent.innerHeight;
        });

        const points = [];
        const maxPoints = 20;

        window.parent.addEventListener('mousemove', (e) => {
            points.push({ x: e.clientX, y: e.clientY, alpha: 1.0 });
        });

        function render() {
            ctx.clearRect(0, 0, width, height);
            for (let i = 0; i < points.length; i++) {
                const pt = points[i];
                pt.alpha *= 0.92;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, (1 - (i / points.length)) * 2.5, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(255, 255, 255, ${pt.alpha * 0.15})`;
                ctx.fill();
            }
            while (points.length > 0 && points[0].alpha < 0.05) {
                points.shift();
            }
            requestAnimationFrame(render);
        }
        render();
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# CARGA DEL MODELO
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    candidates = [
        os.path.join(base_dir, "mnist_model.h5"),
        os.path.join(base_dir, "mnist_model.keras"),
        os.path.join(os.getcwd(), "mnist_model.h5"),
        os.path.join(os.getcwd(), "mnist_model.keras"),
    ]

    for root_dir in [base_dir, os.getcwd()]:
        if os.path.exists(root_dir):
            for file in os.listdir(root_dir):
                if file.endswith((".h5", ".keras")):
                    full = os.path.join(root_dir, file)
                    if full not in candidates:
                        candidates.append(full)

    for path in candidates:
        if os.path.exists(path):
            try:
                return keras.models.load_model(path, compile=False), os.path.basename(path), None
            except Exception as exc:
                try:
                    return keras.models.load_model(path), os.path.basename(path), None
                except Exception:
                    continue

    return None, None, "Binario .h5/.keras no indexado."

model, model_filename, load_err = load_model_pipeline()

# -----------------------------------------------------------------------------
# PANEL LATERAL
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<p style='font-size:0.75rem; color:#71717a; letter-spacing:0.1em; text-transform:uppercase;'>Configuración del Sistema</p>", unsafe_allow_html=True)
    stroke_width = st.slider("Calibre de trazo (px)", 10, 32, 22)
    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.06); margin:1.5rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:0.75rem; color:#71717a; letter-spacing:0.1em; text-transform:uppercase;'>Entorno</p>", unsafe_allow_html=True)
    st.caption(f"Framework: TensorFlow {tf.__version__}")
    if model_filename:
        st.caption(f"Binario activo: {model_filename}")
    else:
        st.caption("Estado: Sin modelo activo")

# -----------------------------------------------------------------------------
# CABECERA PRINCIPAL
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="noir-header">
        <h1 class="noir-title">Inferencia Matricial</h1>
        <div class="noir-subtitle">Clasificación discriminativa de dígitos manuscritos (MNIST)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# ÁREA DE TRAZADO (CENTRADA AL 80%)
# -----------------------------------------------------------------------------
col_l, col_canvas, col_r = st.columns([1, 8, 1])

with col_canvas:
    st.markdown('<div class="canvas-frame">', unsafe_allow_html=True)
    canvas_output = st_canvas(
        fill_color="#000000",
        stroke_width=stroke_width,
        stroke_color="#ffffff",
        background_color="#000000",
        width=880,
        height=400,
        drawing_mode="freedraw",
        key="noir_mnist_canvas",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Fila de acciones
    st.write("")
    c_btn1, c_btn2 = st.columns([3, 1])
    with c_btn1:
        run_inference = st.button("Ejecutar Inferencia", type="primary", use_container_width=True)
    with c_btn2:
        reset_canvas = st.button("Restablecer", type="secondary", use_container_width=True)

    if reset_canvas:
        st.rerun()

# -----------------------------------------------------------------------------
# PREPROCESAMIENTO Y EJECUCIÓN
# -----------------------------------------------------------------------------
def extract_normalised_tensor(raw_matrix: np.ndarray) -> tuple[Image.Image | None, np.ndarray | None]:
    """Aisla la firma matricial, calcula su bounding box y la centra con margen estándar 20x20 en 28x28."""
    img_gray = Image.fromarray(raw_matrix.astype("uint8")).convert("L")
    bbox = img_gray.getbbox()
    if not bbox:
        return None, None

    cropped = img_gray.crop(bbox)
    max_side = max(cropped.width, cropped.height)
    norm_dim = int(max_side * 1.4)
    canvas_container = Image.new("L", (norm_dim, norm_dim), color=0)
    canvas_container.paste(cropped, ((norm_dim - cropped.width) // 2, (norm_dim - cropped.height) // 2))

    digit_28 = canvas_container.resize((28, 28), Image.Resampling.LANCZOS)
    tensor = np.array(digit_28, dtype=np.float32) / 255.0
    return digit_28, tensor.reshape(1, 28, 28, 1)


if run_inference:
    if model is None:
        st.error("No se localizó un modelo válido (.h5 o .keras) en la raíz.")
    elif canvas_output.image_data is None or np.max(canvas_output.image_data) == 0:
        st.info("El plano de captura está vacío.")
    else:
        with col_canvas:
            digit_preview, tensor = extract_normalised_tensor(canvas_output.image_data)
            if tensor is None:
                st.info("Sin señal de trazo identificable.")
            else:
                preds = model.predict(tensor, verbose=0)[0]
                primary_class = int(np.argmax(preds))
                primary_conf = float(np.max(preds) * 100)

                order = np.argsort(preds)[::-1]
                sec_class = int(order[1])
                sec_conf = float(preds[sec_class] * 100)

                # Módulo de métricas de precisión
                st.write("")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(
                        f"""
                        <div class="metric-panel">
                            <div class="metric-caption">Clase Primaria</div>
                            <div class="metric-figure">{primary_class}</div>
                            <div class="metric-sub">Distribución argmax</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f"""
                        <div class="metric-panel">
                            <div class="metric-caption">Confianza</div>
                            <div class="metric-figure">{primary_conf:.1f}<span style="font-size:1.4rem; color:#71717a;">%</span></div>
                            <div class="metric-sub">Softmax saturado</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        f"""
                        <div class="metric-panel">
                            <div class="metric-caption">Alternativa Inmediata</div>
                            <div class="metric-figure">{sec_class}</div>
                            <div class="metric-sub">Margen: {sec_conf:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Vector de densidad de probabilidad monocromático
                st.write("")
                st.markdown("<p style='font-size:0.75rem; color:#71717a; letter-spacing:0.1em; text-transform:uppercase;'>Vector de Probabilidad</p>", unsafe_allow_html=True)
                df_dist = pd.DataFrame({
                    "Clase": [str(i) for i in range(10)],
                    "Densidad (%)": preds * 100,
                }).set_index("Clase")

                st.bar_chart(df_dist, color="#d4d4d8", height=220)

                # Inspección técnica
                with st.expander("Inspección de tensor matricial"):
                    exp_c1, exp_c2 = st.columns([1, 4])
                    with exp_c1:
                        st.image(digit_preview, width=110, caption="Entrada (28x28)")
                    with exp_c2:
                        st.markdown(
                            f"""
                            ```text
                            Forma del tensor: {tensor.shape}
                            Rango dinámico:  [{tensor.min():.3f}, {tensor.max():.3f}]
                            Modo remuestreo: Lanczos Kernel
                            ```
                            """
                        )
