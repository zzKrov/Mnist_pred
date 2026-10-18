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
# ATMÓSFERA VISUAL: TRANSPARENCIA Y ELIMINACIÓN DEL RECUADRO LATERAL
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], .stApp, section.main {
        background: transparent !important;
        background-color: transparent !important;
        color: #f4f4f5 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        overflow-x: hidden;
    }

    [data-testid="stHeader"] { background: transparent !important; }

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
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #fafafa;
        margin-top: 0.2rem;
        margin-bottom: 0.8rem;
        text-shadow: 0 0 25px rgba(255, 255, 255, 0.3);
    }

    /* GUÍA VISUAL CON NÚMERO 3 ANIMADO RESTAURADO */
    .guide-chip {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1.4rem;
        margin: 0 auto 1.4rem auto;
        padding: 8px 22px;
        background: rgba(13, 13, 18, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        width: fit-content;
        font-size: 0.85rem;
        color: #a1a1aa;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.6);
    }

    .stroke-demo {
        width: 34px;
        height: 34px;
        display: inline-block;
        vertical-align: middle;
    }

    .animated-digit-3 {
        stroke-dasharray: 100;
        stroke-dashoffset: 100;
        animation: traceThree 3.2s cubic-bezier(0.4, 0, 0.2, 1) infinite;
    }

    @keyframes traceThree {
        0% { stroke-dashoffset: 100; opacity: 0.2; }
        40% { stroke-dashoffset: 0; opacity: 1; }
        80% { stroke-dashoffset: 0; opacity: 1; }
        100% { stroke-dashoffset: 100; opacity: 0.2; }
    }

    /* ELIMINACIÓN DE CONTENEDORES SECUNDARIOS Y CENTRADO EXACTO */
    div[data-testid="stCustomComponentV1"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        background: transparent !important;
    }

    /* El iframe ahora mide exactamente 600px igual que el canvas (sin espacios laterales) */
    iframe[title*="st_canvas"], iframe[title*="drawable_canvas"] {
        width: 600px !important;
        height: 340px !important;
        margin: 0 auto !important;
        display: block !important;
        background: transparent !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.95), 0 0 30px rgba(255, 255, 255, 0.08) !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }

    iframe[title*="st_canvas"]:hover, iframe[title*="drawable_canvas"]:hover {
        border-color: #ffffff !important;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.95), 0 0 45px rgba(255, 255, 255, 0.2) !important;
    }

    div.stButton > button {
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 0.75rem 1.6rem !important;
        transition: all 0.25s ease !important;
    }

    .stat-card {
        background: rgba(13, 13, 18, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1.1rem;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.5);
    }
    .stat-label {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #71717a;
    }
    .stat-val {
        font-size: 2.5rem;
        font-weight: 600;
        color: #ffffff;
        margin: 2px 0;
        line-height: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# LÁMPARA DE LAVA FLUIDA INTERACTIVA Y SUPRESIÓN DEL FONDO DEL IFRAME
# -----------------------------------------------------------------------------
components.html(
    """
    <script>
    (function() {
        const doc = window.parent.document;

        // 1. Inyección de la simulación de Lámpara de Lava en Canvas
        let canvas = doc.getElementById('digit-lava-canvas');
        if (!canvas) {
            canvas = doc.createElement('canvas');
            canvas.id = 'digit-lava-canvas';
            canvas.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:0;background:#050508;';
            doc.body.prepend(canvas);
        }

        const ctx = canvas.getContext('2d');
        let w = canvas.width = window.parent.innerWidth;
        let h = canvas.height = window.parent.innerHeight;

        window.parent.addEventListener('resize', () => {
            w = canvas.width = window.parent.innerWidth;
            h = canvas.height = window.parent.innerHeight;
        });

        // Esferas de lámpara de lava con atracción y deformación fluida
        const blobs = [
            { x: w * 0.25, y: h * 0.3, r: 260, vx: 0.5, vy: 0.4, color: 'rgba(255, 255, 255, 0.08)' },
            { x: w * 0.75, y: h * 0.35, r: 280, vx: -0.4, vy: 0.5, color: 'rgba(113, 113, 122, 0.07)' },
            { x: w * 0.4, y: h * 0.75, r: 270, vx: 0.4, vy: -0.5, color: 'rgba(212, 212, 216, 0.07)' },
            { x: w * 0.85, y: h * 0.8, r: 230, vx: -0.5, vy: -0.4, color: 'rgba(161, 161, 170, 0.06)' }
        ];

        let mouse = { x: -1000, y: -1000 };
        const sparkles = [];

        window.parent.addEventListener('mousemove', (e) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
            for (let i = 0; i < 2; i++) {
                sparkles.push({
                    x: e.clientX,
                    y: e.clientY,
                    r: Math.random() * 2 + 0.8,
                    vx: (Math.random() - 0.5) * 1.5,
                    vy: (Math.random() - 0.5) * 1.5,
                    alpha: 1.0
                });
            }
        });

        function animate() {
            ctx.clearRect(0, 0, w, h);
            ctx.fillStyle = '#050508';
            ctx.fillRect(0, 0, w, h);
            ctx.globalCompositeOperation = 'screen';

            for (let b of blobs) {
                b.x += b.vx;
                b.y += b.vy;

                if (b.x < -100 || b.x > w + 100) b.vx *= -1;
                if (b.y < -100 || b.y > h + 100) b.vy *= -1;

                // Deformación suave hacia el mouse
                const dm = Math.hypot(b.x - mouse.x, b.y - mouse.y);
                if (dm < 280) {
                    b.x += (mouse.x - b.x) * 0.008;
                    b.y += (mouse.y - b.y) * 0.008;
                }

                const grad = ctx.createRadialGradient(b.x, b.y, b.r * 0.05, b.x, b.y, b.r);
                grad.addColorStop(0, b.color);
                grad.addColorStop(1, 'transparent');

                ctx.beginPath();
                ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
                ctx.fillStyle = grad;
                ctx.fill();
            }

            // Estela de partículas
            for (let i = sparkles.length - 1; i >= 0; i--) {
                const s = sparkles[i];
                s.x += s.vx;
                s.y += s.vy;
                s.alpha -= 0.025;
                if (s.alpha <= 0) {
                    sparkles.splice(i, 1);
                    continue;
                }
                ctx.beginPath();
                ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(255, 255, 255, ${s.alpha})`;
                ctx.shadowBlur = 8;
                ctx.shadowColor = '#ffffff';
                ctx.fill();
            }

            ctx.globalCompositeOperation = 'source-over';
            requestAnimationFrame(animate);
        }
        animate();

        // 2. Eliminación garantizada del fondo del iframe de Fabric.js
        setInterval(() => {
            const iframes = doc.querySelectorAll('iframe');
            iframes.forEach(f => {
                try {
                    const idoc = f.contentDocument || f.contentWindow.document;
                    if (idoc && idoc.querySelector('.canvas-container')) {
                        idoc.body.style.background = 'transparent';
                        idoc.documentElement.style.background = 'transparent';
                    }
                } catch(e) {}
            });
        }, 150);
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
# GUÍA CON EL NÚMERO 3 ANIMADO RESTAURADO
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="guide-chip">
        <svg class="stroke-demo" viewBox="0 0 40 40">
            <path class="animated-digit-3" d="M 12 10 C 24 10, 24 20, 14 20 C 26 20, 26 32, 10 32" fill="none" stroke="#ffffff" stroke-width="2.6" stroke-linecap="round"/>
        </svg>
        <span>1. Dibuja un dígito centrado (0 al 9)</span>
        <span style="color:#52525b;">→</span>
        <span>2. Pulsa Analizar Trazo</span>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# LIENZO CENTRADO ALINEADO EXACTAMENTE CON SUS BOTONES (600x340px)
# -----------------------------------------------------------------------------
_, col_canvas, _ = st.columns([1, 4.2, 1])

with col_canvas:
    canvas_data = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#ffffff",
        background_color="#000000",
        width=600,
        height=340,
        drawing_mode="freedraw",
        key="digit_canvas_v4",
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
