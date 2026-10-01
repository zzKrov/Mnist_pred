import os
import traceback
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Reconocedor de Dígitos",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# ESTILOS NOIR, ANIMACIONES E INSTRUCCIONES VISUALES
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #09090b !important;
        color: #f4f4f5 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    #MainMenu, footer, header { visibility: hidden; }

    /* Encabezado */
    .title {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 500;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #fafafa;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Barra de instrucciones animada */
    .guide-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 2rem;
        margin-bottom: 1.5rem;
        padding: 0.75rem 1.5rem;
        background: #111114;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
    }

    .guide-step {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.82rem;
        color: #a1a1aa;
        letter-spacing: 0.02em;
    }

    /* Animación del trazo continuo */
    .stroke-demo {
        width: 32px;
        height: 32px;
    }

    .animated-digit {
        stroke-dasharray: 100;
        stroke-dashoffset: 100;
        animation: traceDigit 3.5s cubic-bezier(0.4, 0, 0.2, 1) infinite;
    }

    @keyframes traceDigit {
        0% { stroke-dashoffset: 100; opacity: 0.2; }
        40% { stroke-dashoffset: 0; opacity: 1; }
        80% { stroke-dashoffset: 0; opacity: 1; }
        100% { stroke-dashoffset: 100; opacity: 0.2; }
    }

    /* Pulso animado para el botón */
    .pulse-btn-demo {
        width: 14px;
        height: 14px;
        border-radius: 2px;
        background: #fafafa;
        animation: pulseScale 1.8s infinite ease-in-out;
    }

    @keyframes pulseScale {
        0%, 100% { transform: scale(0.85); opacity: 0.4; }
        50% { transform: scale(1.15); opacity: 1; box-shadow: 0 0 10px rgba(255, 255, 255, 0.4); }
    }

    .guide-arrow {
        color: #3f3f46;
        font-size: 0.85rem;
    }

    /* Marco del lienzo (80% centrado) */
    .canvas-frame {
        background: #000000;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 4px;
        padding: 6px;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .canvas-frame:hover {
        border-color: rgba(255, 255, 255, 0.35);
        box-shadow: 0 0 25px rgba(255, 255, 255, 0.03);
    }

    div[data-testid="stCanvas"] {
        display: flex;
        justify-content: center;
    }

    /* Botones */
    div.stButton > button {
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 0.7rem 1.4rem !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button[kind="primary"] {
        background-color: #fafafa !important;
        color: #09090b !important;
        font-weight: 500 !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.15) !important;
    }

    div.stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: #a1a1aa !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        color: #fafafa !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }

    /* Tarjetas de lectura de resultados */
    .stat-card {
        background: #111114;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        padding: 1.25rem;
    }

    .stat-label {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #71717a;
    }

    .stat-val {
        font-size: 2.8rem;
        font-weight: 400;
        color: #fafafa;
        margin-top: 0.25rem;
        line-height: 1;
    }

    .stat-foot {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #a1a1aa;
        margin-top: 0.4rem;
    }
    </style>

    <!-- Estela del cursor sutil -->
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
        window.parent.addEventListener('mousemove', (e) => {
            points.push({ x: e.clientX, y: e.clientY, alpha: 1.0 });
        });

        function render() {
            ctx.clearRect(0, 0, width, height);
            for (let i = 0; i < points.length; i++) {
                const pt = points[i];
                pt.alpha *= 0.93;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, (1 - (i / points.length)) * 2.2, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(255, 255, 255, ${pt.alpha * 0.12})`;
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
# CARGA ROBUSTA DEL MODELO (.h5 o .keras)
# -----------------------------------------------------------------------------
def locate_weights():
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd(),
        os.getcwd(),
    ]
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

    err_log = []
    for file_path in files:
        if os.path.getsize(file_path) < 2048:
            continue
        try:
            return tf.keras.models.load_model(file_path, compile=False), os.path.basename(file_path), None
        except Exception as e1:
            try:
                return tf.keras.models.load_model(file_path), os.path.basename(file_path), None
            except Exception as e2:
                err_log.append(f"{os.path.basename(file_path)}: {str(e1)}")

    return None, None, "\\n".join(err_log)


model, filename, load_err = load_network()
if model is None:
    st.cache_resource.clear()

# -----------------------------------------------------------------------------
# ENCABEZADO E INSTRUCCIONES ANIMADAS
# -----------------------------------------------------------------------------
st.markdown('<div class="title">Reconocedor de Dígitos</div>', unsafe_allow_html=True)

# Guía animada sin saturar con texto
st.markdown(
    """
    <div class="guide-container">
        <div class="guide-step">
            <svg class="stroke-demo" viewBox="0 0 40 40" fill="none">
                <path class="animated-digit" d="M 12 10 C 24 10, 24 20, 14 20 C 26 20, 26 32, 10 32" stroke="#fafafa" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
            <span>Dibuja del 0 al 9</span>
        </div>
        <div class="guide-arrow">→</div>
        <div class="guide-step">
            <div class="pulse-btn-demo"></div>
            <span>Presiona Analizar</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# ÁREA DE DIBUJO (CENTRADA AL 80%)
# -----------------------------------------------------------------------------
_, col_center, _ = st.columns([1, 8, 1])

with col_center:
    st.markdown('<div class="canvas-frame">', unsafe_allow_html=True)
    canvas_data = st_canvas(
        fill_color="#000000",
        stroke_width=22,
        stroke_color="#ffffff",
        background_color="#000000",
        width=880,
        height=380,
        drawing_mode="freedraw",
        key="main_canvas",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    btn_l, btn_r = st.columns([3, 1])
    with btn_l:
        run_btn = st.button("Analizar", type="primary", use_container_width=True)
    with btn_r:
        clear_btn = st.button("Limpiar", type="secondary", use_container_width=True)

    if clear_btn:
        st.rerun()

# -----------------------------------------------------------------------------
# PROCESAMIENTO E IDENTIFICACIÓN
# -----------------------------------------------------------------------------
def prepare_drawing(raw_matrix: np.ndarray):
    """Enfoca el número dibujado, lo centra en un marco cuadrado y lo escala a 28x28."""
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
        st.warning("Dibuja un dígito en el panel antes de presionar Analizar.")
    else:
        with col_center:
            img_28, tensor = prepare_drawing(canvas_data.image_data)
            if tensor is None:
                st.warning("No se detectó ningún trazo.")
            else:
                probs = model.predict(tensor, verbose=0)[0]
                pred = int(np.argmax(probs))
                conf = float(np.max(probs) * 100)

                ranked = np.argsort(probs)[::-1]
                second_pred = int(ranked[1])
                second_conf = float(probs[second_pred] * 100)

                # Lecturas numéricas
                st.write("")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f"""
                        <div class="stat-card">
                            <div class="stat-label">Número detectado</div>
                            <div class="stat-val">{pred}</div>
                            <div class="stat-foot">Resultado principal</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"""
                        <div class="stat-card">
                            <div class="stat-label">Certeza</div>
                            <div class="stat-val">{conf:.0f}%</div>
                            <div class="stat-foot">Nivel de confianza</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f"""
                        <div class="stat-card">
                            <div class="stat-label">Segunda opción</div>
                            <div class="stat-val">{second_pred}</div>
                            <div class="stat-foot">{second_conf:.1f}% de coincidencia</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Gráfica de distribución de probabilidad
                st.write("")
                st.markdown(
                    "<p style='font-size:0.75rem; color:#71717a; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;'>Probabilidades</p>",
                    unsafe_allow_html=True,
                )
                chart_data = pd.DataFrame({
                    "Dígito": [str(i) for i in range(10)],
                    "Porcentaje": probs * 100,
                }).set_index("Dígito")

                st.bar_chart(chart_data, color="#f4f4f5", height=200)
