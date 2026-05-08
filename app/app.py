"""
SmartVision AI - Main Streamlit Application Entry Point
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="SmartVision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp { background: #0a0e1a; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid #1e2d45;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* Hero gradient text */
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 50%, #ff6b6b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.5rem;
    animation: shimmer 3s ease-in-out infinite;
}
@keyframes shimmer {
    0%, 100% { filter: brightness(1); }
    50% { filter: brightness(1.2); }
}

.hero-subtitle {
    text-align: center;
    color: #8b949e;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #161b27 0%, #1e2d45 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 212, 255, 0.15);
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    color: #8b949e;
    font-size: 0.85rem;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Feature card */
.feature-card {
    background: linear-gradient(135deg, #0d1117 0%, #1a1f2e 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 1.8rem;
    height: 100%;
    transition: border-color 0.3s, transform 0.2s;
}
.feature-card:hover {
    border-color: #00d4ff;
    transform: translateY(-2px);
}
.feature-icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
.feature-title {
    color: #e6edf3;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.feature-desc { color: #8b949e; font-size: 0.9rem; line-height: 1.6; }

/* Prediction card */
.pred-card {
    background: linear-gradient(135deg, #0d1117 0%, #1a1f2e 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.6rem;
}
.pred-rank { color: #8b949e; font-size: 0.8rem; }
.pred-label { color: #e6edf3; font-weight: 600; font-size: 1.05rem; }
.pred-conf { color: #00d4ff; font-weight: 700; }

/* Section headers */
.section-header {
    color: #e6edf3;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #1e2d45;
}

/* Model badge */
.model-badge {
    display: inline-block;
    background: linear-gradient(135deg, #00d4ff22, #7b2ff722);
    border: 1px solid #00d4ff44;
    color: #00d4ff;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Nav pills in sidebar */
.nav-pill {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.7rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.3rem;
    color: #c9d1d9;
    cursor: pointer;
    transition: all 0.2s;
}
.nav-pill:hover, .nav-pill.active {
    background: linear-gradient(135deg, #00d4ff22, #7b2ff722);
    border-left: 3px solid #00d4ff;
    color: #00d4ff;
}

/* Divider */
.fancy-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #7b2ff7, transparent);
    margin: 2rem 0;
    border: none;
}

/* Step badges */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: 50%;
    background: linear-gradient(135deg, #00d4ff, #7b2ff7);
    color: white;
    font-weight: 700;
    font-size: 0.9rem;
    margin-right: 0.7rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0;'>
        <div style='font-size:2.5rem'>👁️</div>
        <div style='font-size:1.3rem; font-weight:700; color:#00d4ff;'>SmartVision AI</div>
        <div style='font-size:0.75rem; color:#8b949e; margin-top:0.3rem;'>v2.0 · Powered by TensorFlow</div>
    </div>
    <hr style='border-color:#1e2d45; margin-bottom:1.5rem;'/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home",
         "🖼️  Image Classification",
         "🔍  Object Detection",
         "📊  Model Performance",
         "📷  Live Webcam",
         "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("""<hr style='border-color:#1e2d45; margin:1.5rem 0;'/>""", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#8b949e; text-align:center;'>
        SmartVision AI © 2026<br/>
        Built with TensorFlow & Streamlit
    </div>
    """, unsafe_allow_html=True)


# ── Page Router ──────────────────────────────────────────────────────────────
if "🏠" in page:
    from app_pages.page_home import show
    show()
elif "🖼️" in page:
    from app_pages.page_classification import show
    show()
elif "🔍" in page:
    from app_pages.page_detection import show
    show()
elif "📊" in page:
    from app_pages.page_performance import show
    show()
elif "📷" in page:
    from app_pages.page_webcam import show
    show()
elif "ℹ️" in page:
    from app_pages.page_about import show
    show()
