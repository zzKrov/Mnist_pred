import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
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
# ESTILOS VISUALES, LÁMPARA DE LAVA FLUIDA (GPU) Y CENTRADO EXACTO
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* 1. Transparencia de contenedores nativos para revelar la lámpara de lava */
    html, body, [data-testid="stAppViewContainer"], .stApp, section.main {
        background: transparent !important;
        background-color: transparent !important;
        color: #f4f4f5 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        overflow-x: hidden;
    }

    [data-testid="stHeader"] { background: transparent !important; }

    /* Fondo base continuo */
    body {
        background-color: #050508 !important;
    }

    /* 2. Lámpara de lava fluida con aceleración por GPU (sin consumo de CPU) */
    .lava-lamp-viewport {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        pointer-events: none;
        z-index: 0;
    }

    .lava-orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(75px);
        opacity: 0.28;
        will-change: transform;
        animation-timing-function: ease-in-out;
        animation-iteration-count: infinite;
        animation-direction: alternate;
    }

    .lava-orb-1 {
        width: 480px;
        height: 480px;
        background: radial-gradient(circle, #00f0ff 0%, #3b82f6 50%, transparent 70%);
        top: -8%;
        left: 8%;
        animation: floatOrb1 16s infinite alternate;
    }

    .lava-orb-2 {
        width: 520px;
        height: 520px;
        background: radial-gradient(circle, #a855f7 0%, #ec4899 50%, transparent 70%);
        top: 42%;
        right: -6%;
        animation: floatOrb2 20s infinite alternate;
    }

    .lava-orb-3 {
        width: 460px;
        height: 460px;
        background: radial-gradient(circle, #38bdf8 0%, #6366f1 50%, transparent 70%);
        bottom: -10%;
        left: 18%;
        animation: floatOrb3 18s infinite alternate;
    }

    .lava-orb-4 {
        width: 440px;
        height: 440px;
        background: radial-gradient(circle, #d946ef 0%, #8b5cf6 50%, transparent 70%);
        bottom: 25%;
        right: 25%;
        animation: floatOrb4 22s infinite alternate;
    }

    @keyframes floatOrb1 {
        0% { transform: translate3d(0, 0, 0) scale(1); }
        50% { transform: translate3d(100px, 60px, 0) scale(1.15); }
        100% { transform: translate3d(30px, 120px, 0) scale(0.95); }
    }

    @keyframes floatOrb2 {
        0% { transform: translate3d(0, 0, 0) scale(1); }
        50% { transform: translate3d(-90px, -70px, 0) scale(1.2); }
        100% { transform: translate3d(-20px, -110px, 0) scale(0.92); }
    }

    @keyframes floatOrb3 {
        0% { transform: translate3d(0, 0, 0) scale(1); }
        50% { transform: translate3d(80px, -80px, 0) scale(1.12); }
        100% { transform: translate3d(-40px, -30px, 0) scale(0.98); }
    }

    @keyframes floatOrb4 {
        0% { transform: translate3d(0, 0, 0) scale(1); }
        50% { transform: translate3d(-80px, 50px, 0) scale(1.18); }
        100% { transform: translate3d(40px, -60px, 0) scale(0.9); }
    }

    /* 3. Encabezado y tipografía */
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
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #fafafa;
        margin-top: 0.1rem;
        margin-bottom: 0.8rem;
        text-shadow: 0 0 25px rgba(255, 255, 255, 0.35);
    }

    /* 4. Guía con número 3 animado restaurado */
    .guide-chip {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1.4rem;
        margin: 0 auto 1.4rem auto;
        padding: 8px 24px;
        background: rgba(13, 14, 20, 0.82);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        width: fit-content;
        font-size: 0.85rem;
        color: #a1a1aa;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.6);
        position: relative;
        z-index: 2;
    }

    .stroke-demo {
        width: 36px;
        height: 36px;
        display: inline-block;
        vertical-align: middle;
        filter: drop-shadow(0 0 8px rgba(0, 240, 255, 0.7));
    }

    .animated-digit-3 {
        stroke-dasharray: 120;
        stroke-dashoffset: 120;
        animation: traceDigitMotion 3.2s cubic-bezier(0.4, 0, 0.2, 1) infinite;
    }

    @keyframes traceDigitMotion {
        0% { stroke-dashoffset: 120; opacity: 0.2; }
        45% { stroke-dashoffset: 0; opacity: 1; }
        75% { stroke-dashoffset: 0; opacity: 1; }
        100% { stroke-dashoffset: -120; opacity: 0.2; }
    }

    /* 5. Centrado estricto y eliminación del recuadro lateral */
    div[data-testid="stCustomComponentV1"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        background: transparent !important;
    }

    /* El iframe mide exactamente 600px, coincidiendo con el canvas sin espacio sobrante */
    iframe[title*="st_canvas"], iframe[title*="drawable_canvas"] {
        width: 600px !important;
        height: 340px !important;
        margin: 0 auto !important;
        display: block !important;
        background: transparent !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.95), 0 0 30px rgba(0, 240, 255, 0.12) !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }

    iframe[title*="st_canvas"]:hover, iframe[title*="drawable_canvas"]:hover {
        border-color: #00f0ff !important;
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.95), 0 0 45px rgba(0, 240, 255, 0.3) !important;
    }

    /* 6. Botones con estilos táctiles y de alto contraste */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00f0ff 0%, #3b82f6 50%, #8b5cf6 100%) !important;
        color: #030712 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.75rem 1.6rem !important;
        box-shadow: 0 0 25px rgba(0, 240, 255, 0.45) !important;
        transition: all 0.25s ease !important;
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 40px rgba(0, 240, 255, 0.75) !important;
        color: #ffffff !important;
    }

    div.stButton > button[kind="secondary"] {
        background: rgba(15, 17, 26, 0.85) !important;
        color: #e4e4e7 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 6px !important;
        padding: 0.75rem 1.6rem !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.25s ease !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        transform: translateY(-2px) !important;
        border-color: #ffffff !important;
        color: #ffffff !important;
        box-shadow: 0 0 25px rgba(255, 255, 255, 0.2) !important;
    }

    /* Tarjetas de resultados */
    .stat-card {
        background: rgba(13, 14, 22, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1.1rem;
        text-align: center;
        backdrop-filter: blur(14px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        position: relative;
        z-index: 2;
    }
    .stat-label {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
    }
    .stat-val {
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        margin: 2px 0;
        line-height: 1;
        text-shadow: 0 0 20px rgba(0, 240, 255, 0.5);
    }
    </style>

    <div class="lava-lamp-viewport">
        <div class="lava-orb lava-orb-1"></div>
        <div class="lava-orb lava-orb-2"></div>
        <div class="lava-orb lava-orb-3"></div>
        <div class="lava-orb lava-orb-4"></div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# ESTELA DE PARTÍCULAS EFICIENTE (SIN DUPLICADOS NI FUGAS DE MEMORIA)
# -----------------------------------------------------------------------------
components.html(
    """
    <script>
    (function() {
        const pWin = window.parent;
        const pDoc = pWin.document;

        // Evita re-inicializaciones en recargas de Streamlit
        if (pWin.__KROV_DIGIT_INITIALIZED__) return;
        pWin.__KROV_DIGIT_INITIALIZED__ = true;

        // Canvas de estela ligero
        let canvas = pDoc.getElementById('krov-cursor-trail');
        if (!canvas) {
            canvas = pDoc.createElement('canvas');
            canvas.id = 'krov-cursor-trail';
            canvas.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:1;opacity:0.8;';
            pDoc.body.prepend(canvas);
        }

        const ctx = canvas.getContext('2d');
        let w = canvas.width = pWin.innerWidth;
        let h = canvas.height = pWin.innerHeight;

        pWin.addEventListener('resize', () => {
            w = canvas.width = pWin.innerWidth;
            h = canvas.height = pWin.innerHeight;
        });

        const particles = [];
        let mouse = { x: -100, y: -100 };

        pWin.addEventListener('mousemove', (e) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
            if (particles.length < 25) {
                particles.push({
                    x: e.clientX,
                    y: e.clientY,
                    r: Math.random() * 2 + 1,
                    vx: (Math.random() - 0.5) * 1.2,
                    vy: (Math.random() - 0.5) * 1.2,
                    alpha: 1.0
                });
            }
        });

        function renderTrail() {
            ctx.clearRect(0, 0, w, h);
            for (let i = particles.length - 1; i >= 0; i--) {
                const p = particles[i];
                p.x += p.vx;
                p.y += p.vy;
                p.alpha -= 0.035;

                if (p.alpha <= 0) {
                    particles.splice(i, 1);
                    continue;
                }

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0, 240, 255, ${p.alpha})`;
                ctx.shadowBlur = 8;
                ctx.shadowColor = '#00f0ff';
                ctx.fill();
            }
            requestAnimationFrame(renderTrail);
        }
        renderTrail();

        // Elimina fondos residuales dentro de los iframes mediante MutationObserver
        function stripIframeBg() {
            const iframes = pDoc.querySelectorAll('iframe');
            iframes.forEach(f => {
                try {
                    const idoc = f.contentDocument || f.contentWindow.document;
                    if (idoc && idoc.querySelector('.canvas-container')) {
                        idoc.body.style.background = 'transparent';
                        idoc.documentElement.style.background = 'transparent';
                        const cc = idoc.querySelector('.canvas-container');
                        if (cc) cc.style.margin = '0 auto';
                    }
                } catch(e) {}
            });
        }

        const observer = new MutationObserver(stripIframeBg);
        observer.observe(pDoc.body, { childList: true, subtree: true });
        stripIframeBg();
    })();
    </script>
    """,
    height=0,
    width=0
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
    return None, None, "Error al inicializar la red neuronal."

model, filename, load_err = load_network()

st.markdown('<div class="brand-signature">Krov Industries · Reconocimiento Óptico</div>', unsafe_allow_html=True)
st.markdown('<div class="title">Reconocedor de Dígitos</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# GUÍA VISUAL CON NÚMERO 3 ANIMADO (SVG ROBUSTO)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="guide-chip">
        <svg class="stroke-demo" viewBox="0 0 40 40">
            <path class="animated-digit-3" d="M 12 10 C 25 10, 25 19, 15 20 C 27 21, 27 32, 11 32" fill="none" stroke="#00f0ff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span>1. Dibuja un dígito centrado (0 al 9)</span>
        <span style="color:#52525b;">→</span>
        <span>2. Pulsa Analizar Trazo</span>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# ÁREA DE TRABAJO CENTRADA CON ALINEACIÓN SIMÉTRICA (600px)
# -----------------------------------------------------------------------------
_, col_canvas, _ = st.columns([1.2, 2.3, 1.2])

with col_canvas:
    canvas_data = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#ffffff",
        background_color="#000000",
        width=600,
        height=340,
        drawing_mode="freedraw",
        key="digit_canvas_final",
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
                    st.markdown(f'<div class="stat-card"><div class="stat-label">Segunda Opción</div><div class="stat-val">{second_pred} <span style="font-size:0.9rem; color:#94a3b8;">({second_conf:.0f}%)</span></div></div>', unsafe_allow_html=True)

                st.write("")
                chart_data = pd.DataFrame({
                    "Dígito": [str(i) for i in range(10)],
                    "Probabilidad (%)": probs * 100,
                }).set_index("Dígito")
                st.bar_chart(chart_data, color="#00f0ff", height=180)
