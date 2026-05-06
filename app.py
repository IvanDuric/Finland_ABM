"""
GROCERYsim — Streamlit Web App v2.0
=====================================
Scientific ABM dashboard for consumer behaviour and supply chain
stress-testing in grocery retail.

Tabs
----
  🏠 Data & Population  — upload Firebase + product files, preview demographics
  🎮 Interactive Demo   — single-run visual simulation with live chart updates
  🔬 Scientific Analysis — multi-run Monte Carlo workflow with AI optimisation
  ♻️  Food Waste          — waste dashboard by product / category / reason
  📦 Per-Product         — deep-dive stock / revenue / waste per SKU
  🏛️ Policy Analysis     — Baseline vs Policy comparison (fat tax, subsidy, supply shock, labelling)
  📥 Export              — download all simulation data as CSV

Deployment (online access)
--------------------------
  Simplest (free):  push repo to GitHub → deploy on Streamlit Community Cloud
                    https://streamlit.io/cloud
  Google ecosystem: Google Cloud Run (containerised) → see README for Dockerfile
"""

import base64
import io
import json
import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF

from data_processor import run_pipeline_from_data, ARCHETYPE_LABELS
from model import SupermarketModel, ProductAgent

plt.switch_backend("Agg")

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

def _logo_uri(filename: str) -> str:
    """Read a file from static/ and return a base64 data URI, or '' if missing."""
    path = os.path.join(_STATIC_DIR, filename)
    try:
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "svg": "image/svg+xml", "webp": "image/webp",
                "pdf": "application/pdf"}.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            return f"data:{mime};base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""

# ===========================================================================
# 0. PAGE CONFIG & THEME
# ===========================================================================

st.set_page_config(
    page_title="GROCERYsim ABM",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def apply_theme():
    """Theme — custom component style overrides with dark-mode support."""
    st.markdown("""
    <style>
        /* ── Custom components — light defaults with explicit text colour ── */
        .step-card {
            background: #f0f7ff;
            padding: 14px 18px;
            border-radius: 4px;
            border-left: 4px solid #44A1A0;
            margin-bottom: 14px;
            color: #1a2035;
        }
        .step-card h3, .step-card p { color: #1a2035; }

        .metric-card {
            background: #f8f9fa;
            padding: 14px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            text-align: center;
            color: #1a2035;
        }
        .archetype-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
            margin: 2px;
        }
        .footer {
            text-align: center;
            padding: 20px;
            margin-top: 50px;
            border-top: 1px solid #e5e5e5;
            color: #888;
            font-size: 0.82rem;
        }
        .eu-text { font-style: italic; color: #44A1A0; }

        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            padding: 0 4px;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 2px solid #44A1A0 !important;
        }

        /* ── Dark-mode overrides ── */
        @media (prefers-color-scheme: dark) {
            .step-card {
                background: #1a2a3e !important;
                border-left-color: #44A1A0 !important;
                color: #e4eaf4 !important;
            }
            .step-card h3, .step-card p { color: #e4eaf4 !important; }

            .metric-card {
                background: #1e2d3d !important;
                border-color: #2d4a6e !important;
                color: #e4eaf4 !important;
            }

            .footer {
                border-top-color: #2d4a6e !important;
                color: #8899aa !important;
            }
        }

        [data-theme="dark"] .step-card,
        .stApp[data-theme="dark"] .step-card {
            background: #1a2a3e !important;
            border-left-color: #44A1A0 !important;
            color: #e4eaf4 !important;
        }
        [data-theme="dark"] .step-card h3,
        [data-theme="dark"] .step-card p {
            color: #e4eaf4 !important;
        }
        [data-theme="dark"] .metric-card {
            background: #1e2d3d !important;
            border-color: #2d4a6e !important;
            color: #e4eaf4 !important;
        }
        [data-theme="dark"] .footer {
            border-top-color: #2d4a6e !important;
            color: #8899aa !important;
        }

        /* ── Title row: stack on mobile ── */
        @media (max-width: 640px) {
            div[data-testid="stColumns"] > div[data-testid="stColumn"] {
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }
            div[data-testid="stDownloadButton"] > button {
                font-size: 12px !important;
                padding: 8px 10px !important;
            }
        }

        /* ── Mobile / tablet responsiveness ── */
        @media (max-width: 768px) {
            .stTabs [data-baseweb="tab-list"] {
                overflow-x: auto !important;
                flex-wrap: nowrap !important;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: none;
            }
            .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
            .stTabs [data-baseweb="tab"] {
                white-space: nowrap;
                flex-shrink: 0;
            }

            .stButton > button {
                min-height: 44px !important;
                font-size: 14px !important;
            }

            [data-testid="stSidebar"] {
                min-width: 240px !important;
                max-width: 280px !important;
            }

            .js-plotly-plot, .plotly { width: 100% !important; }

            [data-testid="stMetric"] { min-width: 120px; }
        }

        @media (max-width: 480px) {
            .main .block-container { padding-left: 12px !important; padding-right: 12px !important; }

            .stButton > button { width: 100% !important; min-height: 48px !important; }

            [data-testid="stDataFrame"] { overflow-x: auto !important; }
            [data-testid="stDataFrame"] > div { overflow-x: auto !important; }
        }
    </style>
    """, unsafe_allow_html=True)

apply_theme()

ARCHETYPE_COLORS = {
    "price_champion":   "#DBA159",   # amber
    "green_buyer":      "#BCDC8B",   # green
    "health_optimizer": "#44A1A0",   # teal
    "habitual_buyer":   "#92DDDB",   # light teal
}
ARCHETYPE_EMOJI = {
    "price_champion":   "💸",
    "green_buyer":      "🌿",
    "health_optimizer": "💪",
    "habitual_buyer":   "🔁",
}

# ---------------------------------------------------------------------------
# Global Plotly config — responsive + mobile-friendly toolbar
# ---------------------------------------------------------------------------
_PLOTLY_CFG = {
    "responsive": True,
    "displayModeBar": "hover",
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "scrollZoom": False,
}

# ---------------------------------------------------------------------------
# Onboarding tour steps
# ---------------------------------------------------------------------------
_TOUR_STEPS = [
    {
        "title": "👋 Welcome to GROCERYsim ABM v2.0",
        "body": (
            "This guided tour walks you through the key features in about 2 minutes. "
            "You can skip at any time using the button on the left."
        ),
    },
    {
        "title": "⚙️ Simulation Parameters — Sidebar",
        "body": (
            "The left sidebar controls every aspect of the simulation: duration, "
            "number of consumers, reorder points and lead times. "
            "The model auto-calibrates shelf capacity and stock levels to your store size."
        ),
    },
    {
        "title": "🏠 Tab 1 — Data & Population",
        "body": (
            "Start here. Load your participant cohort from Firebase or upload CSV/JSON files. "
            "The simulation matches real consumer baskets against your product catalogue "
            "before each run."
        ),
    },
    {
        "title": "🎮 Tab 2 — Interactive Demo",
        "body": (
            "Run a single simulation with live chart updates. "
            "Perfect for quickly exploring how parameter changes affect day-by-day revenue, "
            "stock levels, and consumer behaviour archetypes."
        ),
    },
    {
        "title": "🔬 Tab 3 — Scientific Analysis",
        "body": (
            "Run multiple simulations (Monte Carlo) for statistically robust results "
            "with percentile confidence bands (p10–p90). "
            "Uses AI to recommend optimal parameters and compares baseline vs. crisis scenarios."
        ),
    },
    {
        "title": "🏛️ Tab 6 — Policy Analysis",
        "body": (
            "Test policy interventions: fat taxes, subsidies, purchase caps, labelling. "
            "See how each policy shifts consumer behaviour and revenue "
            "across the four consumer archetypes."
        ),
    },
    {
        "title": "🌿 SecureFood Scenario Simulator",
        "body": (
            "Dedicated tool for the Horizon Europe SecureFood project. "
            "Simulate climate disruption in Finnish dairy supply chains from the perspective "
            "of a Supply Chain Actor or Policy Maker."
        ),
    },
    {
        "title": "✅ You're all set!",
        "body": (
            "Explore the remaining tabs: ♻️ Food Waste, 📦 Per-Product deep-dives, "
            "👔 Stakeholder View, 🎚️ Sensitivity Analysis, and 🧪 Behavioural Theory. "
            "Use the 📥 Export tab to download all results as CSV. "
            "Click the '🎓 Tour' button in the sidebar to replay this tour any time."
        ),
    },
]


def render_onboarding_tour():
    """Render the guided onboarding tour banner. Skippable at every step."""
    step = st.session_state.get("tour_step", 0)
    if step <= 0:
        return

    idx = min(step - 1, len(_TOUR_STEPS) - 1)
    current = _TOUR_STEPS[idx]
    total = len(_TOUR_STEPS)
    pct = int(step / total * 100)

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#042026 0%,#073B4C 100%);
                    color:#F4EFE6; padding:20px 26px 16px 26px; border-radius:10px;
                    border-left:5px solid #DBA159; margin-bottom:18px;
                    box-shadow:0 4px 24px rgba(4,32,38,0.18);">
          <div style="font-size:11px;letter-spacing:0.14em;text-transform:uppercase;
                      color:#DBA159;font-weight:700;margin-bottom:6px;">
            Guided Tour &nbsp;·&nbsp; Step {step} of {total}
          </div>
          <div style="font-size:17px;font-weight:700;margin-bottom:8px;
                      color:#F4EFE6;">{current['title']}</div>
          <div style="font-size:14px;line-height:1.65;color:rgba(244,239,230,0.85);
                      max-width:680px;">{current['body']}</div>
          <div style="margin-top:14px;background:rgba(255,255,255,0.12);
                      border-radius:3px;height:3px;">
            <div style="background:#DBA159;width:{pct}%;height:3px;
                        border-radius:3px;transition:width .3s;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_skip, _, c_prev, c_next = st.columns([2, 4, 1, 1])
    with c_skip:
        if st.button("✕  Skip tour", key="tour_skip"):
            st.session_state["tour_step"] = 0
            st.rerun()
    with c_prev:
        if step > 1:
            if st.button("← Back", key="tour_prev"):
                st.session_state["tour_step"] = step - 1
                st.rerun()
    with c_next:
        label = "✓ Done" if step == total else "Next →"
        btn_type = "secondary" if step == total else "primary"
        if st.button(label, key="tour_next", type=btn_type):
            if step >= total:
                st.session_state["tour_step"] = 0
            else:
                st.session_state["tour_step"] = step + 1
            st.rerun()


# ===========================================================================
# 0b. LANDING PAGE
# ===========================================================================

_LANDING_CSS = """
/* GROCERYsim ABM — landing page styles */
* { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg-dark: #042026;
  --bg-mid: #073B4C;
  --teal: #44A1A0;
  --teal-light: #92DDDB;
  --amber: #DBA159;
  --amber-light: #FCC995;
  --amber-dark: #895833;
  --green: #BCDC8B;
  --cream: #F4EFE6;
  --cream-dim: rgba(244, 239, 230, 0.68);
  --cream-mute: rgba(244, 239, 230, 0.42);
  --hairline: rgba(146, 221, 219, 0.16);
  --hairline-strong: rgba(146, 221, 219, 0.28);
  --max-width: 1440px;
}

html, body {
  background: var(--bg-dark);
  color: var(--cream);
  font-family: 'Figtree', system-ui, -apple-system, sans-serif;
  font-size: 16px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
  overflow-x: hidden;
}

.mono {
  font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace;
  font-size: 0.78em;
  letter-spacing: 0.04em;
}

a { color: inherit; text-decoration: none; }

/* ── Background ── */
/* position:absolute avoids the iOS-Safari iframe bug where position:fixed
   creates a new stacking context that renders over z-index:1 content */
.bg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; pointer-events: none; overflow: hidden; }
.bg-gradient {
  background:
    radial-gradient(ellipse at 80% -10%, rgba(68,161,160,0.22), transparent 55%),
    radial-gradient(ellipse at -10% 80%, rgba(219,161,89,0.16), transparent 60%),
    linear-gradient(180deg, var(--bg-dark) 0%, var(--bg-mid) 100%);
}
/* filter:blur removed — causes GPU compositing layer bugs in iOS Safari inside iframes */
.aurora { position: absolute; border-radius: 50%; opacity: 0.35; animation: drift 24s ease-in-out infinite alternate; }
.aurora.a1 { width: 700px; height: 700px; left: -10%; top: 5%; background: radial-gradient(circle, rgba(68,161,160,0.45), transparent 60%); animation-duration: 28s; }
.aurora.a2 { width: 600px; height: 600px; right: -8%; top: 30%; background: radial-gradient(circle, rgba(146,221,219,0.22), transparent 60%); animation-duration: 32s; animation-delay: -8s; }
.aurora.a3 { width: 800px; height: 800px; left: 30%; bottom: 0%; background: radial-gradient(circle, rgba(219,161,89,0.22), transparent 60%); animation-duration: 36s; animation-delay: -14s; }
@keyframes drift { 0% { transform: translate(0,0) scale(1); } 50% { transform: translate(40px,-30px) scale(1.08); } 100% { transform: translate(-30px,30px) scale(0.95); } }
.grain { position: absolute; inset: 0; opacity: 0.05; background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='240' height='240'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>"); }

/* ── Layout ── */
.page { position: relative; z-index: 1; }
main { max-width: var(--max-width); margin: 0 auto; padding: 0 clamp(24px, 5vw, 80px); }

/* ── Header ── */
.site-header {
  position: relative; z-index: 10;
  display: flex; align-items: center; justify-content: flex-start;
  padding: 22px clamp(24px, 5vw, 80px);
  max-width: var(--max-width); margin: 0 auto;
  border-bottom: 1px solid var(--hairline);
}
.logo-slot { position: relative; width: 200px; height: 56px; display: flex; align-items: center; justify-content: center; }
.logo-slot-bg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.logo-slot-label { position: relative; z-index: 1; font-size: 10px; letter-spacing: 0.14em; color: var(--cream-dim); opacity: 0.7; }
.brand { display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 16px; letter-spacing: -0.005em; }
.brand-dim { color: var(--cream-dim); font-weight: 500; letter-spacing: 0.04em; }

/* ── Hero ── */
.hero {
  display: grid; grid-template-columns: 1.1fr 1fr;
  gap: clamp(32px, 5vw, 80px);
  padding: clamp(56px, 9vh, 110px) 0 clamp(32px, 5vh, 56px);
  align-items: center;
}
.eyebrow { display: flex; align-items: center; gap: 14px; font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px; letter-spacing: 0.14em; color: var(--teal-light); margin-bottom: 28px; text-transform: uppercase; }
.eyebrow-line { width: 28px; height: 1px; background: var(--teal); }
.hero-title { font-size: clamp(48px, 7vw, 96px); font-weight: 800; line-height: 1; letter-spacing: -0.03em; color: #F4EFE6 !important; margin-bottom: 18px; text-shadow: 0 2px 24px rgba(0,0,0,0.4); display: flex; flex-direction: column; align-items: flex-start; gap: 20px; }
.hero-title .title-name { display: block; line-height: 0.95; }
.hero-title .title-accent { font-style: italic; font-weight: 300; color: var(--amber); letter-spacing: -0.02em; }
.hero-title .title-tag { display: inline-flex; align-items: center; font-size: 0.28em; font-weight: 600; letter-spacing: 0.18em; color: var(--teal-light); border: 1px solid var(--teal); padding: 8px 16px; border-radius: 2px; vertical-align: unset; margin-top: 0; }
.hero-lede { font-size: clamp(17px, 1.4vw, 22px); font-weight: 500; line-height: 1.35; color: var(--cream); max-width: 520px; margin-bottom: 20px; text-wrap: balance; }
.hero-sub { font-size: clamp(14px, 1vw, 16px); line-height: 1.6; color: var(--cream-dim); max-width: 480px; margin-bottom: 36px; }
.hero-actions { display: flex; gap: 14px; flex-wrap: wrap; }

/* Hero logo strip */
.hero-logos { margin-top: 36px; padding-top: 24px; border-top: 1px solid var(--hairline); display: flex; flex-direction: column; gap: 12px; max-width: 560px; }
.hero-logos-label { color: var(--cream-mute); font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase; }
.hero-logos-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.hero-logo-slot { position: relative; aspect-ratio: 5 / 2; border: none; border-radius: 3px; background: rgba(7, 59, 76, 0.25); display: flex; align-items: center; justify-content: center; overflow: hidden; transition: background 0.2s ease; }
.hero-logo-slot.has-logo { background: #ffffff; }
.hero-logo-slot.has-logo:hover { background: #f5f5f5; }
.hero-logo-slot:not(.has-logo):hover { border: 1px dashed var(--teal); background: rgba(7, 59, 76, 0.45); }
.hero-logo-slot svg { position: absolute; inset: 0; opacity: 0.55; }
.hero-logo-tag { position: relative; z-index: 1; color: var(--cream-mute); font-size: 9px; letter-spacing: 0.1em; background: rgba(4, 32, 38, 0.6); padding: 3px 8px; border-radius: 2px; }

/* Buttons */
.btn { display: inline-flex; align-items: center; gap: 8px; padding: 14px 22px; font-size: 14px; font-weight: 600; letter-spacing: 0.01em; border-radius: 2px; transition: all 0.2s ease; cursor: pointer; font-family: inherit; border: none; }
.btn-primary { background: var(--amber); color: var(--bg-dark); border: 1px solid var(--amber); }
.btn-primary:hover { background: var(--amber-light); border-color: var(--amber-light); transform: translateY(-1px); box-shadow: 0 8px 24px rgba(219,161,89,0.25); }
.btn-ghost { background: transparent; color: var(--cream); border: 1px solid var(--hairline-strong); }
.btn-ghost:hover { border-color: var(--teal-light); color: var(--teal-light); }

/* Hero right (sim) */
.hero-right { display: flex; flex-direction: column; gap: 14px; }
.sim-frame { position: relative; aspect-ratio: 16 / 11; border-radius: 4px; border: 1px solid var(--hairline-strong); background: rgba(4, 32, 38, 0.6); overflow: hidden; box-shadow: 0 0 0 1px rgba(146,221,219,0.04), 0 30px 80px rgba(0,0,0,0.4), inset 0 0 80px rgba(68,161,160,0.06); }
.sim-caption { display: flex; align-items: center; gap: 12px; font-size: 11px; letter-spacing: 0.06em; color: var(--cream-mute); font-family: 'JetBrains Mono', ui-monospace, monospace; text-transform: uppercase; }

/* Simulation overlay */
.sim-wrap { position: absolute; inset: 0; }
.sim-wrap canvas { display: block; width: 100%; height: 100%; }
.sim-overlay { position: absolute; inset: 0; pointer-events: none; font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 10px; letter-spacing: 0.1em; color: var(--cream-dim); }
.sim-corner { position: absolute; display: flex; align-items: center; gap: 6px; padding: 12px 14px; text-transform: uppercase; }
.sim-corner.tl { top: 0; left: 0; }
.sim-corner.tr { top: 0; right: 0; }
.sim-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 1.6s ease-in-out infinite; }
.sim-legend { position: absolute; bottom: 14px; left: 14px; right: 14px; display: flex; flex-wrap: wrap; gap: 10px 16px; }
.sim-legend-item { display: flex; align-items: center; gap: 6px; font-size: 9px; }
.sim-legend-dot { width: 6px; height: 6px; border-radius: 50%; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ── Sections ── */
section { padding: clamp(80px, 14vh, 160px) 0; }
.section-head { display: flex; align-items: center; gap: 16px; margin-bottom: 36px; font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--teal-light); }
.section-num { display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; border: 1px solid var(--teal); color: var(--teal); border-radius: 50%; }
.section-title { font-size: clamp(36px, 5vw, 68px); font-weight: 700; line-height: 1.05; letter-spacing: -0.03em; margin-bottom: 64px; max-width: 22ch; text-wrap: balance; color: var(--cream); }
.section-title em { font-style: italic; font-weight: 300; color: var(--amber); }

/* ── Overview ── */
.overview-grid { display: grid; grid-template-columns: 1.4fr 1fr; gap: clamp(32px, 5vw, 80px); align-items: start; }
.overview-lead p { font-size: clamp(16px, 1.3vw, 19px); line-height: 1.6; margin-bottom: 20px; color: var(--cream); }
.overview-lead p.muted { color: var(--cream-dim); font-size: 15px; }
.overview-stats { display: flex; flex-direction: column; gap: 24px; border-left: 1px solid var(--hairline); padding-left: clamp(24px, 3vw, 48px); }
.stat { display: flex; flex-direction: column; gap: 4px; }
.stat-num { font-size: clamp(40px, 4.4vw, 64px); font-weight: 800; line-height: 1; letter-spacing: -0.03em; color: var(--cream); }
.stat-label { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--cream-mute); }

/* ── Features ── */
.feature-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
.feature-card { padding: 32px 28px; background: #073B4C; border: 1px solid rgba(146,221,219,0.28); border-radius: 3px; transition: border-color 0.25s ease, transform 0.25s ease; display: flex; flex-direction: column; gap: 14px; min-width: 0; overflow: hidden; }
.feature-card:hover { border-color: var(--teal); background: #0a4d63; transform: translateY(-2px); }
.feature-head { display: flex; justify-content: space-between; align-items: center; }
.feature-n { color: var(--amber); font-size: 11px; }
.feature-tag { color: var(--teal-light); font-size: 9px; padding: 3px 8px; border: 1px solid var(--hairline-strong); border-radius: 2px; }
.feature-card h3 { font-size: 22px; font-weight: 600; letter-spacing: -0.015em; color: #F4EFE6; line-height: 1.15; }
.feature-card p { font-size: 14px; line-height: 1.6; color: rgba(244,239,230,0.85); }
section.features { padding: 20px 0 40px; }
section.features .section-head { display: none; }
section.features .section-title { display: none; }

/* ── Partners ── */
section.partners { padding-top: 40px; }
.partner-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--hairline); border: 1px solid var(--hairline); }
.partner-card { display: flex; align-items: center; gap: 16px; padding: 22px 20px; background: rgba(4, 32, 38, 0.6); transition: background 0.2s; }
.partner-card:hover { background: rgba(7, 59, 76, 0.6); }
.partner-mark { flex-shrink: 0; width: 40px; height: 40px; border-radius: 2px; overflow: hidden; }
.partner-name { font-size: 13.5px; font-weight: 600; letter-spacing: -0.005em; color: var(--cream); margin-bottom: 2px; }
.partner-kind { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 10px; letter-spacing: 0.08em; color: var(--cream-mute); text-transform: uppercase; }
.partner-foot { margin-top: 32px; padding-top: 24px; border-top: 1px solid var(--hairline); }
.eu-flag { display: flex; align-items: center; gap: 12px; font-size: 13px; color: var(--cream-dim); }
.eu-flag svg { flex-shrink: 0; }

/* ── Footer ── */
.site-footer { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 24px; max-width: var(--max-width); margin: 60px auto 0; padding: 40px clamp(24px, 5vw, 80px) 32px; border-top: 1px solid var(--hairline); }
.foot-left p { font-size: 13px; color: var(--cream-dim); margin-top: 8px; max-width: 380px; }
.foot-right { color: var(--cream-mute); }

/* ── Responsive ── */
/* ── Responsive — tablet (≤860px) ── */
@media (max-width: 860px) {
  .hero { grid-template-columns: 1fr; }
  .sim-frame { aspect-ratio: 16 / 11; max-height: 50vh; }
  .overview-grid { grid-template-columns: 1fr; }
  .overview-stats { border-left: none; border-top: 1px solid var(--hairline); padding-left: 0; padding-top: 24px; flex-direction: row; gap: 32px; flex-wrap: wrap; }
  .feature-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .feature-card h3 { font-size: 16px; }
  .feature-card { padding: 20px 16px; gap: 10px; }
  .partner-grid { grid-template-columns: repeat(2, 1fr); }
  .hero-logos-row { grid-template-columns: repeat(4, 1fr); }

  main { padding: 0 clamp(16px, 4vw, 40px); }
  .site-header { padding: 0 clamp(16px, 4vw, 40px); }
  .hero { padding: 48px 0 20px; }
  .hero-title { font-size: clamp(52px, 10vw, 96px); }
  .hero-actions { gap: 10px; }
  .btn { min-height: 44px; padding: 11px 22px; font-size: 14px; }
  .overview-stat-val { font-size: clamp(28px, 5vw, 48px); }
  .section-title { font-size: clamp(26px, 4vw, 44px); }
}

/* ── Responsive — phone (≤560px) ── */
@media (max-width: 560px) {
  .partner-grid { grid-template-columns: 1fr; }
  .overview-stats { flex-direction: column; }

  main { padding: 0 16px; }
  .site-header { padding: 0 16px; height: 52px; }
  .logo-slot { max-width: 80px; }
  .logo-slot img { max-height: 60px !important; max-width: 90px !important; }
  .logo-slot-label { font-size: 9px; }

  .hero { padding: 32px 0 16px; }
  .feature-grid { grid-template-columns: 1fr; }
  .hero-title { font-size: clamp(44px, 14vw, 72px); }
  .hero-lede { font-size: 15px; }
  .hero-sub { font-size: 13px; }
  .hero-actions { flex-direction: column; align-items: flex-start; gap: 10px; }
  .btn { width: 100%; min-height: 48px; justify-content: center; font-size: 14px; padding: 12px 20px; }
  .hero-logos-row { grid-template-columns: repeat(2, 1fr); gap: 8px; }

  .overview-stat-val { font-size: clamp(24px, 8vw, 40px); }
  .section-title { font-size: clamp(22px, 6vw, 36px); }
  .feature-card { padding: 20px; }
  .partner-card { padding: 12px; }
  .partner-mark { width: 36px; height: 36px; }

  .site-footer { flex-direction: column; gap: 12px; align-items: flex-start; padding: 20px 16px; }
  .lang-btn { min-height: 36px; padding: 6px 10px; }
}

/* ── Language switcher ── */
.lang-switcher { display: flex; align-items: center; gap: 2px; margin-left: auto; background: rgba(7,59,76,0.5); border: 1px solid var(--hairline-strong); border-radius: 3px; padding: 3px; }
.lang-btn { background: transparent; border: none; color: var(--cream-mute); font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; padding: 5px 10px; border-radius: 2px; cursor: pointer; transition: all 0.15s ease; }
.lang-btn:hover { color: var(--teal-light); background: rgba(68,161,160,0.12); }
.lang-btn.active { color: var(--amber); background: rgba(219,161,89,0.15); }

/* ── Case studies page ── */
.cs-hero { padding: 12px 0 20px; text-align: center; }
.cs-hero-back { display: inline-flex; align-items: center; gap: 8px; color: var(--cream-mute); font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; cursor: pointer; transition: color 0.2s; margin-bottom: 12px; border: none; background: transparent; }
.cs-hero-back:hover { color: var(--teal-light); }
.cs-hero-back svg { transition: transform 0.2s; }
.cs-hero-back:hover svg { transform: translateX(-3px); }
.cs-title { font-size: clamp(28px, 3.5vw, 52px); font-weight: 800; line-height: 1; letter-spacing: -0.03em; color: var(--cream); margin-bottom: 8px; }
.cs-subtitle { font-size: clamp(13px, 1vw, 15px); color: var(--cream-dim); max-width: 500px; margin: 0 auto; }
.cs-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; padding-bottom: 32px; }
@media (max-width: 1100px) { .cs-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px)  { .cs-grid { grid-template-columns: 1fr; } }

.cs-card { position: relative; display: flex; flex-direction: column; background: rgba(7,59,76,0.45); border: 1px solid var(--hairline); border-radius: 4px; overflow: hidden; transition: all 0.25s ease; }
.cs-card.active:hover { border-color: var(--teal); transform: translateY(-3px); box-shadow: 0 16px 48px rgba(0,0,0,0.35), 0 0 0 1px rgba(68,161,160,0.2); }
.cs-card.inactive { opacity: 0.55; pointer-events: none; }
.cs-photo { position: relative; aspect-ratio: 16/9; overflow: hidden; }
.cs-photo-inner { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; }
.cs-badge { position: absolute; top: 10px; right: 10px; font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 10px; font-weight: 600; letter-spacing: 0.12em; padding: 3px 8px; border-radius: 2px; text-transform: uppercase; }
.cs-badge.active   { background: rgba(188,220,139,0.2); color: var(--green); border: 1px solid rgba(188,220,139,0.4); }
.cs-badge.soon     { background: rgba(219,161,89,0.15); color: var(--amber); border: 1px solid rgba(219,161,89,0.3); }
.cs-country { position: absolute; top: 10px; left: 10px; font-size: 18px; }
.cs-body { padding: 14px 16px 18px; display: flex; flex-direction: column; gap: 8px; flex: 1; }
.cs-card-title { font-size: 17px; font-weight: 700; letter-spacing: -0.01em; color: var(--cream); line-height: 1.2; }
.cs-card-desc  { font-size: 13px; line-height: 1.6; color: var(--cream-dim); flex: 1; }
.cs-card-tag   { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 10px; letter-spacing: 0.1em; color: var(--teal-light); text-transform: uppercase; }
.cs-card-btn   { margin-top: 6px; display: inline-flex; align-items: center; gap: 8px; padding: 11px 18px; font-size: 13px; font-weight: 600; border-radius: 2px; cursor: pointer; font-family: inherit; border: none; transition: all 0.2s; }
.cs-card-btn.launch { background: var(--amber); color: var(--bg-dark); }
.cs-card-btn.launch:hover { background: var(--amber-light); transform: translateY(-1px); box-shadow: 0 6px 20px rgba(219,161,89,0.3); }
.cs-card-btn.disabled { background: rgba(146,221,219,0.1); color: var(--cream-mute); border: 1px solid var(--hairline); cursor: not-allowed; }
"""

_GROCERY_SIM_JSX = """
const GroceryStore = ({ palette }) => {
  const canvasRef = React.useRef(null);
  const wrapRef = React.useRef(null);
  const stateRef = React.useRef(null);
  const [hud, setHud] = React.useState({ shoppers: 12, stockouts: 0, restock: 0 });

  React.useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext('2d');
    let raf;
    const W = 320, H = 220;
    const SHELF_X0 = 40, SHELF_DX = 38, SHELF_W = 30, SHELF_H = 14, SHELF_COUNT = 6;
    const AISLE_Y = [60, 110, 160];
    const CORRIDOR_LEFT_X = 22;
    const CORRIDOR_RIGHT_X = SHELF_X0 + SHELF_COUNT * SHELF_DX - 6;
    const AISLE_BELOW_Y = AISLE_Y.map(y => y + SHELF_H + 8);
    const TOP_LANE_Y = 30;
    const makeShelves = () => {
      const s = [];
      for (let row = 0; row < 3; row++)
        for (let i = 0; i < SHELF_COUNT; i++)
          s.push({ x: SHELF_X0 + i*SHELF_DX, y: AISLE_Y[row], w: SHELF_W, h: SHELF_H, stock: 0.6+Math.random()*0.4, cat: i%3, row });
      return s;
    };
    const archetypes = [
      { label:'Calm',         color: palette.lightTeal,  panic:0.0,  hoard:0.2,  speed:0.55 },
      { label:'Price-sens.',  color: palette.green,      panic:0.1,  hoard:0.4,  speed:0.60 },
      { label:'Hoarder',      color: palette.amber,      panic:0.4,  hoard:0.85, speed:0.70 },
      { label:'Panic',        color: palette.amberLight, panic:0.85, hoard:0.6,  speed:0.85 },
    ];
    const pathToShelf = (x, y, shelf) => {
      const cx = shelf.x + shelf.w/2;
      const ay = AISLE_BELOW_Y[shelf.row];
      const useLeft = Math.abs(x - CORRIDOR_LEFT_X) <= Math.abs(x - CORRIDOR_RIGHT_X);
      const corX = useLeft ? CORRIDOR_LEFT_X : CORRIDOR_RIGHT_X;
      return [{x:corX,y},{x:corX,y:ay},{x:cx,y:ay}];
    };
    const pathToExit = (x, y, shelf) => {
      const ay = AISLE_BELOW_Y[shelf.row];
      const useLeft = Math.abs(x - CORRIDOR_LEFT_X) <= Math.abs(x - CORRIDOR_RIGHT_X);
      const corX = useLeft ? CORRIDOR_LEFT_X : CORRIDOR_RIGHT_X;
      return [{x:corX,y:ay},{x:corX,y:TOP_LANE_Y},{x:CORRIDOR_LEFT_X,y:TOP_LANE_Y},{x:CORRIDOR_LEFT_X,y:20}];
    };
    const makeShopper = (shelves) => {
      const a = archetypes[Math.floor(Math.random()*archetypes.length)];
      const target = shelves[Math.floor(Math.random()*shelves.length)];
      const x = 16 + Math.random()*8, y = 20 + Math.random()*12;
      return { x, y, targetShelf: target, archetype: a, state:'walking', stateT:0, cart:0, path: pathToShelf(x,y,target), wp:0 };
    };
    const init = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = wrap.getBoundingClientRect();
      canvas.width = rect.width*dpr; canvas.height = rect.height*dpr;
      canvas.style.width = rect.width+'px'; canvas.style.height = rect.height+'px';
      ctx.setTransform(rect.width*dpr/W, 0, 0, rect.height*dpr/H, 0, 0);
      const shelves = makeShelves();
      const shoppers = [];
      for (let i=0;i<12;i++) shoppers.push(makeShopper(shelves));
      stateRef.current = { W, H, shelves, shoppers, t:0, restocks:0, stockouts:0 };
    };
    init();
    const ro = new ResizeObserver(init); ro.observe(wrap);
    let lastHudT = 0;
    const stepAlongPath = (sh) => {
      if (sh.wp >= sh.path.length) return true;
      const wp = sh.path[sh.wp];
      const dx = wp.x-sh.x, dy = wp.y-sh.y;
      const d = Math.sqrt(dx*dx+dy*dy);
      if (d < 1.2) { sh.wp++; return sh.wp >= sh.path.length; }
      sh.x += (dx/d)*sh.archetype.speed;
      sh.y += (dy/d)*sh.archetype.speed;
      return false;
    };
    const tick = () => {
      const s = stateRef.current;
      if (!s) { raf = requestAnimationFrame(tick); return; }
      s.t++;
      if (s.t % 600 === 0) { for (const sh of s.shelves) if (sh.stock<0.5) sh.stock=Math.min(1,sh.stock+0.5); s.restocks++; }
      while (s.shoppers.length < 12) s.shoppers.push(makeShopper(s.shelves));
      for (const sh of s.shoppers) {
        sh.stateT++;
        if (sh.state==='walking') { if (stepAlongPath(sh)) { sh.state='picking'; sh.stateT=0; } }
        else if (sh.state==='picking') {
          if (sh.stateT>50) {
            const want=0.05+sh.archetype.hoard*0.18;
            const took=Math.min(want,sh.targetShelf.stock);
            sh.targetShelf.stock-=took;
            if (sh.targetShelf.stock<=0.001) { sh.targetShelf.stock=0; s.stockouts++; }
            sh.cart+=took;
            if (sh.archetype.panic>0.5 && Math.random()<0.6 && sh.cart<0.6) {
              sh.targetShelf=s.shelves[Math.floor(Math.random()*s.shelves.length)];
              sh.path=pathToShelf(sh.x,sh.y,sh.targetShelf); sh.wp=0; sh.state='walking'; sh.stateT=0;
            } else { sh.path=pathToExit(sh.x,sh.y,sh.targetShelf); sh.wp=0; sh.state='leaving'; sh.stateT=0; }
          }
        } else if (sh.state==='leaving') {
          if (stepAlongPath(sh)) { const idx=s.shoppers.indexOf(sh); if(idx>=0) s.shoppers.splice(idx,1); }
        }
      }
      // Render
      ctx.fillStyle=palette.bgDark; ctx.fillRect(0,0,W,H);
      ctx.strokeStyle='rgba(146,221,219,0.05)'; ctx.lineWidth=0.3;
      for(let x=0;x<W;x+=16){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
      for(let y=0;y<H;y+=16){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
      ctx.strokeStyle='rgba(146,221,219,0.25)'; ctx.lineWidth=0.6; ctx.strokeRect(8,8,W-16,H-16);
      ctx.fillStyle='rgba(146,221,219,0.5)'; ctx.font='5px "JetBrains Mono",monospace';
      ctx.fillText('ENTRANCE',12,18); ctx.fillText('CHECKOUT',12,H-6); ctx.fillText('STOCKROOM',W-50,H-6);
      ctx.fillStyle='rgba(68,161,160,0.15)'; ctx.fillRect(W-16,H-30,8,15);
      for (const sh of s.shelves) {
        const sc = sh.stock>0.5 ? palette.lightTeal : sh.stock>0.15 ? palette.amberLight : 'rgba(255,90,90,0.9)';
        ctx.fillStyle='rgba(7,59,76,0.85)'; ctx.fillRect(sh.x,sh.y,sh.w,sh.h);
        ctx.strokeStyle='rgba(146,221,219,0.35)'; ctx.lineWidth=0.4; ctx.strokeRect(sh.x,sh.y,sh.w,sh.h);
        ctx.fillStyle=sc; ctx.fillRect(sh.x+2,sh.y+2,(sh.w-4)*sh.stock,sh.h-4);
        if (sh.stock<0.05) {
          const a=0.3+0.3*Math.sin(s.t*0.15);
          ctx.strokeStyle=`rgba(255,90,90,${a})`; ctx.lineWidth=0.8; ctx.strokeRect(sh.x-1,sh.y-1,sh.w+2,sh.h+2);
        }
      }
      ctx.fillStyle='rgba(146,221,219,0.4)'; ctx.font='4px "JetBrains Mono",monospace';
      ctx.fillText('AISLE 01',12,70); ctx.fillText('AISLE 02',12,120); ctx.fillText('AISLE 03',12,170);
      for (const sh of s.shoppers) {
        ctx.fillStyle=sh.archetype.color; ctx.beginPath(); ctx.arc(sh.x,sh.y,2.2,0,Math.PI*2); ctx.fill();
        ctx.strokeStyle=sh.archetype.color+'60'; ctx.lineWidth=0.4; ctx.beginPath(); ctx.arc(sh.x,sh.y,3.6,0,Math.PI*2); ctx.stroke();
        if (sh.state==='walking' && sh.wp<sh.path.length) {
          ctx.strokeStyle=sh.archetype.color+'22'; ctx.lineWidth=0.3; ctx.beginPath(); ctx.moveTo(sh.x,sh.y);
          for(let i=sh.wp;i<sh.path.length;i++) ctx.lineTo(sh.path[i].x,sh.path[i].y);
          ctx.stroke();
        }
      }
      if (s.t-lastHudT>30) { lastHudT=s.t; setHud({shoppers:s.shoppers.length,stockouts:s.stockouts,restocks:s.restocks}); }
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, [palette]);

  return (
    <div ref={wrapRef} className="sim-wrap">
      <canvas ref={canvasRef} />
      <div className="sim-overlay">
        <div className="sim-corner tl"><span className="sim-dot"/><span>STORE LIVE · {hud.shoppers} SHOPPERS</span></div>
        <div className="sim-corner tr"><span>STOCKOUTS · {String(hud.stockouts).padStart(3,'0')}</span></div>
        <div className="sim-legend">
          {[['CALM',palette.lightTeal],['PRICE SENS.',palette.green],['HOARDER',palette.amber],['PANIC',palette.amberLight]].map(([l,c])=>(
            <div key={l} className="sim-legend-item">
              <span className="sim-legend-dot" style={{background:c}}/>
              <span>{l}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
"""

_LANDING_APP_JSX = """
const { useState, useEffect } = React;

// =============================================================================
// LOGO CONFIGURATION
// All PNG files live in the static/ folder next to app.py.
// They are served at /app/static/<filename>.png
//
// SIZE ADJUSTMENTS:
//   Header logo   -> search for "HEADER LOGO SIZE" (~60 lines below)
//                    change maxHeight / maxWidth on the <img> tag
//   Hero strip    -> search for "HERO LOGO SIZE" (~320 lines below)
//                    change maxHeight on the <img> style
//   Partner marks -> CSS class .partner-mark in _LANDING_CSS (~line 322)
//                    change width / height on that rule
// =============================================================================

// LOGO_URIS is injected by Python as base64 data URIs (see render_landing_page).
// SIZE ADJUSTMENTS:
//   Header logo   -> search for "HEADER LOGO SIZE" (~60 lines below)
//                    change maxHeight / maxWidth on the <img> tag
//   Hero strip    -> search for "HERO LOGO SIZE" (~320 lines below)
//                    change maxHeight on the <img> style
//   Partner marks -> CSS class .partner-mark in _LANDING_CSS (~line 322)
//                    change width / height on that rule

const LOGO_CONFIG = {
  // Populated from LOGO_URIS injected by Python; falls back to '' (shows placeholder)
  header:   (typeof LOGO_URIS !== 'undefined' && LOGO_URIS.header)    ? LOGO_URIS.header    : '',
  hero:     (typeof LOGO_URIS !== 'undefined' && LOGO_URIS.hero)      ? LOGO_URIS.hero      : ['','','',''],
  partners: (typeof LOGO_URIS !== 'undefined' && LOGO_URIS.partners)  ? LOGO_URIS.partners  : ['','','','','','','',''],
};

// Helper: renders a real <img> if url is set, otherwise the striped placeholder
const LogoImg = ({ url, alt, className, style }) => url
  ? <img src={url} alt={alt || ''} className={className} style={{objectFit:'contain', ...style}} />
  : null;

const PALETTE = {
  bgDark: '#042026', bgMid: '#073B4C', teal: '#44A1A0', lightTeal: '#92DDDB',
  amber: '#DBA159', amberLight: '#FCC995', amberDark: '#895833',
  green: '#BCDC8B', cream: '#F4EFE6',
};

// ── Translations ─────────────────────────────────────────────────────────────
const TRANSLATIONS = {
  en: {
    eyebrow: 'SECUREFOOD',
    heroLede: 'Agent-Based Model for Consumer Behavior & Supply Chain Stress-Testing.',
    heroSub: 'Simulate consumer behavior and supply-chain dynamics in a retail environment. Stress-test the resilience of food supply chains under crisis scenarios — from panic buying to logistics disruption.',
    heroBtn: 'Explore case studies',
    heroDocsBtn: 'Read documentation',
    figCaption: 'Live store — cognitive shoppers, shelf inventory, periodic restock.',
    overviewLabel: 'WHAT IS GROCERYSIM',
    overviewTitle: 'A web application that lets stakeholders stress-test the resilience of food supply chains.',
    overviewP1: 'The model represents a retail environment as autonomous agents: consumers with individual cognitive traits, shelves with finite inventory, and logistics with realistic lead times. From their interactions, system-level behaviour emerges — resilience, fragility, and adaptation under stress.',
    overviewP2: 'Researchers calibrate scenarios; policy makers explore interventions; retailers validate contingency plans before they are needed.',
    stat1: 'Agent-runs / day', stat2: 'Crisis scenarios', stat3: 'EU markets modeled',
    featuresLabel: 'KEY FEATURES',
    featuresTitle: 'Three pillars. One model.',
    f1t: 'Cognitive Agents', f1d: 'Consumers with individual traits — panic, hoarding, price sensitivity. Each shopper makes decisions under uncertainty, producing realistic behavioural diversity.', f1tag: 'BEHAVIOUR',
    f2t: 'Logistics Simulation', f2d: 'Realistic inventory management with lead times, capacity limits, and delivery cycles. Stockouts cascade through the network just as they do in the real world.', f2tag: 'SUPPLY CHAIN',
    f3t: 'Scientific Optimization', f3d: 'AI-driven recommendations balance waste against stockouts. Run thousands of policy permutations and surface the strategies that hold under shock.', f3tag: 'OPTIMIZATION',
    partnersLabel: 'CONSORTIUM',
    partnersTitle: 'Built by 23 institutions across 14 countries.',
    euText: 'Funded by the European Union under Horizon Europe · SecureFood Consortium',
    footerTagline: 'An open agent-based model for stress-testing the resilience of grocery supply chains.',
    footerCopy: '© 2026 · SecureFood Consortium',
  },
  fi: {
    eyebrow: 'SECUREFOOD',
    heroLede: 'Agenttipohjainen malli kuluttajakäyttäytymiselle ja toimitusketjun stressitestaukselle.',
    heroSub: 'Simuloi kuluttajakäyttäytymistä ja elintarviketoimitusketjun dynamiikkaa kriisitilanteissa — paniikkiostoksista logistiikkahäiriöihin.',
    heroBtn: 'Tutustu tapaustutkimuksiin',
    heroDocsBtn: 'Lue dokumentaatio',
    figCaption: 'Reaaliaikainen kauppa — kognitiiviset ostajat, hyllyvarasto, jaksottainen täydennys.',
    overviewLabel: 'MIKÄ ON GROCERYSIM',
    overviewTitle: 'Verkkosovellus, jolla sidosryhmät voivat testata elintarviketoimitusketjujen kestävyyttä.',
    overviewP1: 'Malli kuvaa vähittäismyyntiympäristöä autonomisina agentteina: kuluttajat yksilöllisillä kognitiivisilla ominaisuuksilla, hyllyt rajallisella varastolla ja logistiikka realistisilla toimitusajoilla. Niiden vuorovaikutuksesta syntyy järjestelmätason käyttäytyminen.',
    overviewP2: 'Tutkijat kalibroivat skenaarioita; päättäjät tutkivat interventioita; vähittäiskauppiaat validoivat valmiussuunnitelmia ennen kuin niitä tarvitaan.',
    stat1: 'Agenttikierrosta / pv', stat2: 'Kriisiskenaariot', stat3: 'EU-markkinat mallinnettuna',
    featuresLabel: 'OMINAISUUDET',
    featuresTitle: 'Kolme pilaria. Yksi malli.',
    f1t: 'Kognitiiviset agentit', f1d: 'Kuluttajat yksilöllisillä ominaisuuksilla — paniikki, hamstraus, hintaherkkyys. Jokainen ostaja tekee päätöksiä epävarmuudessa tuottaen realistista käyttäytymistä.', f1tag: 'KÄYTTÄYTYMINEN',
    f2t: 'Logistiikkasimulaatio', f2d: 'Realistinen varastonhallinta toimitusajoilla, kapasiteettirajoituksilla ja toimitussykleillä. Varastopuutteet leviävät verkon läpi kuten todellisuudessa.', f2tag: 'TOIMITUSKETJU',
    f3t: 'Tieteellinen optimointi', f3d: 'Tekoälypohjainen optimointi tasapainottaa hävikin varastopuutteiden välillä. Testaa tuhansia politiikkapermutaatioita ja löydä vakaat strategiat.', f3tag: 'OPTIMOINTI',
    partnersLabel: 'KONSORTIO',
    partnersTitle: 'Rakennettu 23 instituution toimesta 14 maassa.',
    euText: 'Rahoittaa Euroopan unioni Horisontti Eurooppa ‑ohjelman kautta · SecureFood-konsortio',
    footerTagline: 'Avoin agenttipohjainen malli elintarviketoimitusketjujen resilienssitestaukseen.',
    footerCopy: '© 2026 · SecureFood-konsortio',
  },
  el: {
    eyebrow: 'SECUREFOOD',
    heroLede: 'Μοντέλο Πολλαπλών Παραγόντων για τη Συμπεριφορά Καταναλωτών και τον Έλεγχο Ανθεκτικότητας της Εφοδιαστικής Αλυσίδας.',
    heroSub: 'Προσομοιώστε τη συμπεριφορά καταναλωτών και τη δυναμική της εφοδιαστικής αλυσίδας τροφίμων σε σενάρια κρίσης — από πανικόβλητες αγορές έως διαταραχές εφοδιασμού.',
    heroBtn: 'Εξερευνήστε μελέτες περίπτωσης',
    heroDocsBtn: 'Διαβάστε την τεκμηρίωση',
    figCaption: 'Ζωντανό κατάστημα — γνωστικοί αγοραστές, αποθέματα ραφιών, περιοδικός εφοδιασμός.',
    overviewLabel: 'ΤΙ ΕΙΝΑΙ ΤΟ GROCERYSIM',
    overviewTitle: 'Μια εφαρμογή που επιτρέπει στους φορείς να δοκιμάζουν την ανθεκτικότητα των αλυσίδων τροφίμων.',
    overviewP1: 'Το μοντέλο αναπαριστά ένα λιανικό περιβάλλον ως αυτόνομους παράγοντες: καταναλωτές με ατομικά γνωστικά χαρακτηριστικά, ράφια με πεπερασμένα αποθέματα και εφοδιαστική με ρεαλιστικούς χρόνους παράδοσης.',
    overviewP2: 'Ερευνητές βαθμονομούν σενάρια· υπεύθυνοι χάραξης πολιτικής εξερευνούν παρεμβάσεις· λιανοπωλητές επικυρώνουν σχέδια έκτακτης ανάγκης.',
    stat1: 'Εκτελέσεις πρακτόρων / ημ.', stat2: 'Σενάρια κρίσης', stat3: 'Αγορές ΕΕ σε μοντέλο',
    featuresLabel: 'ΒΑΣΙΚΑ ΧΑΡΑΚΤΗΡΙΣΤΙΚΑ',
    featuresTitle: 'Τρεις πυλώνες. Ένα μοντέλο.',
    f1t: 'Γνωστικοί Πράκτορες', f1d: 'Καταναλωτές με ατομικά χαρακτηριστικά — πανικός, συσσώρευση, ευαισθησία τιμών. Κάθε αγοραστής λαμβάνει αποφάσεις υπό αβεβαιότητα.', f1tag: 'ΣΥΜΠΕΡΙΦΟΡΑ',
    f2t: 'Προσομοίωση Logistics', f2d: 'Ρεαλιστική διαχείριση αποθεμάτων με χρόνους παράδοσης και περιορισμούς χωρητικότητας. Οι ελλείψεις εξαπλώνονται στο δίκτυο.', f2tag: 'ΕΦΟΔΙΑΣΤΙΚΗ',
    f3t: 'Επιστημονική Βελτιστοποίηση', f3d: 'Συστάσεις ΑΙ που εξισορροπούν τις απώλειες με τις ελλείψεις αποθεμάτων. Εκτελέστε χιλιάδες παραμετροποιήσεις πολιτικής.', f3tag: 'ΒΕΛΤΙΣΤΟΠΟΙΗΣΗ',
    partnersLabel: 'ΚΟΝΣΟΡΤΣΙΟΥΜ',
    partnersTitle: 'Κατασκευάστηκε από 23 ιδρύματα σε 14 χώρες.',
    euText: 'Χρηματοδοτείται από την Ευρωπαϊκή Ένωση στο πλαίσιο του Horizon Europe · Κοινοπραξία SecureFood',
    footerTagline: 'Ένα ανοιχτό μοντέλο πολλαπλών παραγόντων για τον έλεγχο ανθεκτικότητας.',
    footerCopy: '© 2026 · Κοινοπραξία SecureFood',
  },
  pt: {
    eyebrow: 'SECUREFOOD',
    heroLede: 'Modelo Baseado em Agentes para Comportamento do Consumidor e Resiliência da Cadeia de Abastecimento.',
    heroSub: 'Simule o comportamento do consumidor e a dinâmica da cadeia de abastecimento alimentar em cenários de crise — do pânico nas compras às perturbações logísticas.',
    heroBtn: 'Explorar estudos de caso',
    heroDocsBtn: 'Ler documentação',
    figCaption: 'Loja ao vivo — compradores cognitivos, inventário de prateleiras, reabastecimento periódico.',
    overviewLabel: 'O QUE É O GROCERYSIM',
    overviewTitle: 'Uma aplicação web que permite às partes interessadas testar a resiliência das cadeias de abastecimento alimentar.',
    overviewP1: 'O modelo representa um ambiente de retalho como agentes autónomos: consumidores com traços cognitivos individuais, prateleiras com inventário finito e logística com prazos de entrega realistas.',
    overviewP2: 'Investigadores calibram cenários; decisores exploram intervenções; retalhistas validam planos de contingência antes de serem necessários.',
    stat1: 'Execuções de agentes / dia', stat2: 'Cenários de crise', stat3: 'Mercados UE modelados',
    featuresLabel: 'FUNCIONALIDADES',
    featuresTitle: 'Três pilares. Um modelo.',
    f1t: 'Agentes Cognitivos', f1d: 'Consumidores com traços individuais — pânico, acumulação, sensibilidade ao preço. Cada comprador toma decisões sob incerteza.', f1tag: 'COMPORTAMENTO',
    f2t: 'Simulação Logística', f2d: 'Gestão de inventário realista com prazos de entrega, limites de capacidade e ciclos de entrega. As ruturas propagam-se pela rede.', f2tag: 'CADEIA DE ABAST.',
    f3t: 'Otimização Científica', f3d: 'Recomendações de IA que equilibram desperdício e ruturas de stock. Execute milhares de permutações de políticas e identifique estratégias robustas.', f3tag: 'OTIMIZAÇÃO',
    partnersLabel: 'CONSÓRCIO',
    partnersTitle: 'Construído por 23 instituições em 14 países.',
    euText: 'Financiado pela União Europeia ao abrigo do Horizonte Europa · Consórcio SecureFood',
    footerTagline: 'Um modelo aberto baseado em agentes para testar a resiliência das cadeias de abastecimento alimentar.',
    footerCopy: '© 2026 · Consórcio SecureFood',
  },
};

// ── Background ──────────────────────────────────────────────────────────────
const Background = () => (
  <div className="bg bg-gradient">
    <div className="aurora a1" />
    <div className="aurora a2" />
    <div className="aurora a3" />
    <div className="grain" />
  </div>
);

// ── Language switcher ─────────────────────────────────────────────────────────
const LangSwitcher = ({ lang, setLang }) => {
  const langs = [['en','EN'],['fi','FI'],['el','EL'],['pt','PT']];
  return (
    <div className="lang-switcher" title="Select language">
      {langs.map(([code, label]) => (
        <button key={code}
          className={'lang-btn' + (lang === code ? ' active' : '')}
          onClick={() => setLang(code)}>
          {label}
        </button>
      ))}
    </div>
  );
};

// ── Header ───────────────────────────────────────────────────────────────────
const Header = ({ lang, setLang }) => (
  <header className="site-header">
    <div className="logo-slot" aria-label="Header logo">
      {LOGO_CONFIG.header
        ? (/* HEADER LOGO SIZE — change maxHeight (px) and maxWidth (px) here */
           <img src={LOGO_CONFIG.header} alt="Logo"
                style={{maxHeight:'200px', maxWidth:'300px', objectFit:'contain', display:'block'}} />)
        : (<>
            <svg className="logo-slot-bg" viewBox="0 0 200 56" preserveAspectRatio="none" aria-hidden="true">
              <defs>
                <pattern id="logo-stripe" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <line x1="0" y1="0" x2="0" y2="8" stroke="rgba(146,221,219,0.18)" strokeWidth="2"/>
                </pattern>
              </defs>
              <rect x="0.5" y="0.5" width="199" height="55" fill="url(#logo-stripe)" stroke="rgba(146,221,219,0.28)" strokeDasharray="3 3"/>
            </svg>
            <span className="logo-slot-label mono">LOGO&nbsp;·&nbsp;200×56</span>
          </>)
      }
    </div>
    <LangSwitcher lang={lang} setLang={setLang} />
  </header>
);

// ── Hero ─────────────────────────────────────────────────────────────────────
const Hero = ({ t }) => (
  <section className="hero">
    <div className="hero-left" style={{position: 'relative', zIndex: 5}}>
      <div className="eyebrow">
        <span className="eyebrow-line" />
        <span>{t.eyebrow}</span>
      </div>
      <h1 className="hero-title" style={{color: '#F4EFE6'}}>
        <span className="title-name">
          <span style={{color: '#F4EFE6', fontWeight: 800}}>GROCERY</span><span style={{color: '#DBA159', fontStyle: 'italic', fontWeight: 300}}>sim</span>
        </span>
        <span className="title-tag">ABM</span>
      </h1>
      <p className="hero-lede" style={{color: '#F4EFE6'}}>{t.heroLede}</p>
      <p className="hero-sub">{t.heroSub}</p>
      <div className="hero-actions">
        <button className="btn btn-primary" onClick={() => {
          try { window.parent.postMessage({type:'launch_case_studies'}, '*'); } catch(e) {}
        }}>
          {t.heroBtn}
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M3 7h8m0 0L7.5 3.5M11 7l-3.5 3.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
        <a className="btn btn-ghost" href="#"
           onClick={(e) => {
             e.preventDefault();
             const uri = (typeof PDF_URI !== 'undefined' && PDF_URI) ? PDF_URI : null;
             if (uri) {
               const a = document.createElement('a');
               a.href = uri;
               a.download = 'GROCERYsim_User_Manual.pdf';
               document.body.appendChild(a);
               a.click();
               document.body.removeChild(a);
             }
           }}>{t.heroDocsBtn}</a>
      </div>
      <div className="hero-logos">
        <div className="hero-logos-row">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className={'hero-logo-slot' + (LOGO_CONFIG.hero[i] ? ' has-logo' : '')}
                 aria-label={'Partner logo ' + (i+1)}>
              {LOGO_CONFIG.hero[i]
                ? (/* HERO LOGO SIZE
                      Slot 0 (EU.png)  — change the first  maxHeight / padding values below
                      Slots 1-3        — change the second maxHeight / padding values below */
                   <img src={LOGO_CONFIG.hero[i]} alt={'Partner ' + (i+1)}
                        style={{maxWidth:'100%',
                                maxHeight: i === 0 ? '52px' : '40px',
                                objectFit:'contain', display:'block', margin:'auto',
                                padding:  i === 0 ? '3px 6px' : '5px 8px'}} />)
                : (<>
                    <svg viewBox="0 0 120 48" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
                      <defs>
                        <pattern id={'hlogo-stripe-' + i} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                          <line x1="0" y1="0" x2="0" y2="6" stroke="rgba(146,221,219,0.18)" strokeWidth="2"/>
                        </pattern>
                      </defs>
                      <rect width="120" height="48" fill={'url(#hlogo-stripe-' + i + ')'} />
                    </svg>
                    <span className="hero-logo-tag mono">{'LOGO ' + String(i+1).padStart(2,'0')}</span>
                  </>)
              }
            </div>
          ))}
        </div>
      </div>
    </div>
    <div className="hero-right">
      <div className="sim-frame">
        <GroceryStore palette={PALETTE} />
      </div>
      <div className="sim-caption">
        <span className="mono">FIG&nbsp;01</span>
        <span>{t.figCaption}</span>
      </div>
    </div>
  </section>
);

// ── Overview ─────────────────────────────────────────────────────────────────
const Overview = ({ t }) => (
  <section className="overview" id="overview">
    <div className="section-head">
      <span className="section-num">01</span>
      <span className="section-label">{t.overviewLabel}</span>
    </div>
    <h2 className="section-title">
      {t.overviewTitle.replace('stress-test', '')}
      <em>stress-test</em>
      {t.overviewTitle.split('stress-test')[1] || ''}
    </h2>
    <div className="overview-grid">
      <div className="overview-lead">
        <p>{t.overviewP1}</p>
        <p className="muted">{t.overviewP2}</p>
      </div>
      <div className="overview-stats">
        <div className="stat"><span className="stat-num">142k</span><span className="stat-label">{t.stat1}</span></div>
        <div className="stat"><span className="stat-num">9</span><span className="stat-label">{t.stat2}</span></div>
        <div className="stat"><span className="stat-num">27</span><span className="stat-label">{t.stat3}</span></div>
      </div>
    </div>
  </section>
);

// ── Features ─────────────────────────────────────────────────────────────────
const Features = ({ t }) => {
  const features = [
    { n: '01', title: t.f1t, desc: t.f1d, tag: t.f1tag },
    { n: '02', title: t.f2t, desc: t.f2d, tag: t.f2tag },
    { n: '03', title: t.f3t, desc: t.f3d, tag: t.f3tag },
  ];
  return (
    <section className="features" id="features">
      <div className="section-head">
        <span className="section-num">02</span>
        <span className="section-label">{t.featuresLabel}</span>
      </div>
      <h2 className="section-title">{t.featuresTitle.split('. ')[0]}. <em>{t.featuresTitle.split('. ')[1]}</em></h2>
      <div className="feature-grid">
        {features.map((f, i) => (
          <article key={i} className="feature-card">
            <div className="feature-head">
              <span className="mono feature-n">{f.n}</span>
              <span className="mono feature-tag">{f.tag}</span>
            </div>
            <h3>{f.title}</h3>
            <p>{f.desc}</p>
          </article>
        ))}
      </div>
    </section>
  );
};

// ── Partners ─────────────────────────────────────────────────────────────────
const Partners = ({ t }) => {
  const partners = [
    { name: 'Wageningen University', kind: 'Research lead' },
    { name: 'CNR — Italy', kind: 'Modelling' },
    { name: 'INRAE', kind: 'Agricultural data' },
    { name: 'JRC Ispra', kind: 'EU coordination' },
    { name: 'University of Bonn', kind: 'Economics' },
    { name: 'TU Delft', kind: 'Systems engineering' },
    { name: 'Aarhus University', kind: 'Behavioural science' },
    { name: 'EFSA', kind: 'Food safety' },
  ];
  return (
    <section className="partners" id="partners">
      <div className="section-head">
        <span className="section-num">03</span>
        <span className="section-label">{t.partnersLabel}</span>
      </div>
      <h2 className="section-title">{t.partnersTitle.replace('14 countries', '')} <em>14 {t.partnersTitle.includes('countries') ? 'countries' : t.partnersTitle.includes('maassa') ? 'maassa' : t.partnersTitle.includes('χώρες') ? 'χώρες' : 'países'}.</em></h2>
      <div className="partner-grid">
        {partners.map((p, i) => (
          <div key={i} className="partner-card">
            <div className="partner-mark">
              {LOGO_CONFIG.partners[i]
                ? <img src={LOGO_CONFIG.partners[i]} alt={p.name} style={{width:'100%',height:'100%',objectFit:'contain',borderRadius:'2px'}} />
                : (<svg viewBox="0 0 56 56" width="100%" height="100%">
                    <defs>
                      <pattern id={'stripe-' + i} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                        <line x1="0" y1="0" x2="0" y2="6" stroke="rgba(146,221,219,0.25)" strokeWidth="2"/>
                      </pattern>
                    </defs>
                    <rect width="56" height="56" fill={'url(#stripe-' + i + ')'} stroke="rgba(146,221,219,0.18)" />
                  </svg>)
              }
            </div>
            <div className="partner-text">
              <div className="partner-name">{p.name}</div>
              <div className="partner-kind">{p.kind}</div>
            </div>
          </div>
        ))}
      </div>
      <div className="partner-foot">
        <div className="eu-flag">
          <svg viewBox="0 0 24 24" width="22" height="22">
            <circle cx="12" cy="12" r="11" fill="none" stroke="#DBA159" strokeWidth="0.8"/>
            {Array.from({length: 12}).map((_, k) => {
              const a = (k / 12) * Math.PI * 2 - Math.PI/2;
              return <circle key={k} cx={12 + Math.cos(a)*7.5} cy={12 + Math.sin(a)*7.5} r="0.9" fill="#DBA159"/>;
            })}
          </svg>
          <span>{t.euText}</span>
        </div>
      </div>
    </section>
  );
};

// ── Footer ───────────────────────────────────────────────────────────────────
const Footer = ({ t }) => (
  <footer className="site-footer">
    <div className="foot-left">
      <div className="brand">
        <span style={{color: '#F4EFE6'}}>GROCERY<span style={{color: '#DBA159'}}>sim</span><span className="brand-dim">&nbsp;ABM</span></span>
      </div>
      <p>{t.footerTagline}</p>
    </div>
    <div className="foot-right">
      <span className="mono">{t.footerCopy}</span>
    </div>
  </footer>
);

// ── App ───────────────────────────────────────────────────────────────────────
const App = () => {
  const _storedLang = (() => { try { return window.parent.sessionStorage.getItem('grocerysim_lang'); } catch(e) { return null; } })();
  const initLang = (_storedLang && TRANSLATIONS[_storedLang]) ? _storedLang : ((typeof INITIAL_LANG !== 'undefined' && TRANSLATIONS[INITIAL_LANG]) ? INITIAL_LANG : 'en');
  const [lang, setLangState] = useState(initLang);
  const setLang = (code) => { setLangState(code); try { window.parent.sessionStorage.setItem('grocerysim_lang', code); } catch(e) {} };
  const t = TRANSLATIONS[lang];
  return (
    <div className="page">
      <Background />
      <Header lang={lang} setLang={setLang} />
      <main>
        <Hero t={t} />
        <Features t={t} />
      </main>
      <Footer t={t} />
    </div>
  );
};

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
"""

# ---------------------------------------------------------------------------
# 0c. CASE STUDIES PAGE JSX
# ---------------------------------------------------------------------------

_CASE_STUDIES_JSX = """
const { useState } = React;

// ── Logo config (mirror of landing page LOGO_CONFIG.header) ─────────────────
// CS_HEADER_URI is injected by Python as a base64 data URI (see render_case_studies_page).
// SIZE: change maxHeight/maxWidth on the <img> inside CSHeader (~15 lines below)
const CS_HEADER_LOGO = (typeof CS_HEADER_URI !== 'undefined') ? CS_HEADER_URI : '';

// ── Case-Studies Translations ────────────────────────────────────────────────
const CS_TRANS = {
  en: {
    eyebrow: 'SECUREFOOD',
    pageTitle: 'Case Studies',
    pageSub: 'Select a market context to explore.',
    backBtn: 'Back to overview',
    launchBtn: 'Launch simulation',
    comingSoon: 'Coming soon',
    cards: [
      { title: 'Finland — Dairy Supply Chain',
        desc: 'Simulate dairy product availability, panic-buying dynamics and supply disruptions in the Finnish grocery market.',
        tag: 'DAIRY · NORTHERN EU', status: 'active' },
      { title: 'Greece — Dairy Supply Chain',
        desc: 'Simulate dairy product availability, panic-buying dynamics and supply disruptions in the Greek grocery market.',
        tag: 'DAIRY · SOUTH EU', status: 'soon' },
      { title: 'Portugal — Fruits',
        desc: 'Simulate fruit product availability, panic-buying dynamics and supply disruptions in the Portuguese grocery market.',
        tag: 'FRUITS · WEST EU', status: 'soon' },
      { title: 'Greece — Fish Supply Chain',
        desc: 'Simulate fish product availability, panic-buying dynamics and supply disruptions in the Greek grocery market.',
        tag: 'FISH · SOUTH EU', status: 'soon' },
    ],
  },
  fi: {
    eyebrow: 'SECUREFOOD',
    pageTitle: 'Tapaustutkimukset',
    pageSub: 'Valitse markkinalähtökohta tutkittavaksi.',
    backBtn: 'Takaisin etusivulle',
    launchBtn: 'Käynnistä simulaatio',
    comingSoon: 'Tulossa pian',
    cards: [
      { title: 'Suomi — Maitotuoteketju',
        desc: 'Simuloi maitotuotteiden saatavuutta, paniikkiostoksia ja toimitushäiriöitä suomalaisessa ruokakaupassa.',
        tag: 'MAITOTUOTTEET · POHJ. EU', status: 'active' },
      { title: 'Kreikka — Maitotuoteketju',
        desc: 'Simuloi maitotuotteiden saatavuutta, paniikkiostoksia ja toimitushäiriöitä kreikkalaisessa ruokakaupassa.',
        tag: 'MAITOTUOTTEET · ET. EU', status: 'soon' },
      { title: 'Portugali — Hedelmät',
        desc: 'Simuloi hedelmätuotteiden saatavuutta, paniikkiostoksia ja toimitushäiriöitä portugalilaisessa ruokakaupassa.',
        tag: 'HEDELMÄT · LÄNSI-EU', status: 'soon' },
      { title: 'Kreikka — Kalan toimitusketju',
        desc: 'Simuloi kalatuotteiden saatavuutta, paniikkiostoksia ja toimitushäiriöitä kreikkalaisessa ruokakaupassa.',
        tag: 'KALA · ET. EU', status: 'soon' },
    ],
  },
  el: {
    eyebrow: 'SECUREFOOD',
    pageTitle: 'Μελέτες Περίπτωσης',
    pageSub: 'Επιλέξτε πλαίσιο αγοράς για εξερεύνηση.',
    backBtn: 'Πίσω στην επισκόπηση',
    launchBtn: 'Εκκίνηση προσομοίωσης',
    comingSoon: 'Σύντομα',
    cards: [
      { title: 'Φινλανδία — Αλυσίδα Γαλακτοκομικών',
        desc: 'Προσομοιώστε διαθεσιμότητα γαλακτοκομικών, αγορές πανικού και διαταραχές στη φινλανδική αγορά.',
        tag: 'ΓΑΛΑΚΤΟΚΟΜΙΚΑ · ΒΟΡΕΙΑ ΕΕ', status: 'active' },
      { title: 'Ελλάδα — Αλυσίδα Γαλακτοκομικών',
        desc: 'Προσομοιώστε τη διαθεσιμότητα γαλακτοκομικών, αγορές πανικού και διαταραχές στην ελληνική αγορά.',
        tag: 'ΓΑΛΑΚΤΟΚΟΜΙΚΑ · ΝΟΤΙΑ ΕΕ', status: 'soon' },
      { title: 'Πορτογαλία — Φρούτα',
        desc: 'Προσομοιώστε τη διαθεσιμότητα φρούτων, αγορές πανικού και διαταραχές στην πορτογαλική αγορά.',
        tag: 'ΦΡΟΥΤΑ · ΔΥΤΙΚΗ ΕΕ', status: 'soon' },
      { title: 'Ελλάδα — Αλυσίδα Ψαριού',
        desc: 'Προσομοιώστε τη διαθεσιμότητα ψαριών, αγορές πανικού και διαταραχές στην ελληνική αγορά.',
        tag: 'ΨΑΡΙ · ΝΟΤΙΑ ΕΕ', status: 'soon' },
    ],
  },
  pt: {
    eyebrow: 'SECUREFOOD',
    pageTitle: 'Estudos de Caso',
    pageSub: 'Selecione um contexto de mercado para explorar.',
    backBtn: 'Voltar à visão geral',
    launchBtn: 'Iniciar simulação',
    comingSoon: 'Em breve',
    cards: [
      { title: 'Finlândia — Cadeia de Laticínios',
        desc: 'Simule disponibilidade de laticínios, compras em pânico e perturbações logísticas no mercado finlandês.',
        tag: 'LATICÍNIOS · NORTE UE', status: 'active' },
      { title: 'Grécia — Cadeia de Laticínios',
        desc: 'Simule disponibilidade de laticínios, compras em pânico e perturbações logísticas no mercado grego.',
        tag: 'LATICÍNIOS · SUL UE', status: 'soon' },
      { title: 'Portugal — Frutas',
        desc: 'Simule disponibilidade de frutas, compras em pânico e perturbações logísticas no mercado português.',
        tag: 'FRUTAS · OESTE UE', status: 'soon' },
      { title: 'Grécia — Cadeia de Peixe',
        desc: 'Simule disponibilidade de peixe, compras em pânico e perturbações logísticas no mercado grego.',
        tag: 'PEIXE · SUL UE', status: 'soon' },
    ],
  },
};

const CARD_FLAGS = ['🇫🇮', '🇬🇷', '🇵🇹', '🇬🇷'];

// ── Icon illustrations ────────────────────────────────────────────────────────

const PhotoFI = () => (
  <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" style={{display:'block'}}>
    <defs>
      <linearGradient id="fi-g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stopColor="#042026"/>
        <stop offset="100%" stopColor="#0a4a5e"/>
      </linearGradient>
    </defs>
    <rect width="400" height="300" fill="url(#fi-g)"/>
    <ellipse cx="200" cy="148" rx="72" ry="72" fill="rgba(146,221,219,0.05)"/>
    <rect x="184" y="66" width="32" height="14" rx="6" fill="rgba(188,220,139,0.25)" stroke="rgba(188,220,139,0.65)" strokeWidth="1.5"/>
    <rect x="181" y="79" width="38" height="18" rx="3" fill="rgba(146,221,219,0.08)" stroke="rgba(146,221,219,0.5)" strokeWidth="1.5"/>
    <path d="M181,97 C168,104 162,118 162,132 L162,208 Q162,222 176,222 L224,222 Q238,222 238,208 L238,132 C238,118 232,104 219,97 Z" fill="rgba(146,221,219,0.08)" stroke="rgba(146,221,219,0.55)" strokeWidth="1.8"/>
    <path d="M164,175 Q182,168 200,172 Q218,176 236,170" fill="none" stroke="rgba(188,220,139,0.45)" strokeWidth="1.5" strokeLinecap="round"/>
    <text x="200" y="200" textAnchor="middle" fill="rgba(188,220,139,0.7)" fontSize="9" fontFamily="monospace" letterSpacing="3" fontWeight="600">DAIRY</text>
    <text x="200" y="260" textAnchor="middle" fill="rgba(146,221,219,0.22)" fontSize="10" fontFamily="monospace" letterSpacing="4">FINLAND · FI</text>
  </svg>
);

const PhotoGR = () => (
  <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" style={{display:'block'}}>
    <defs>
      <linearGradient id="gr-g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stopColor="#1a0a00"/>
        <stop offset="100%" stopColor="#2a1500"/>
      </linearGradient>
    </defs>
    <rect width="400" height="300" fill="url(#gr-g)"/>
    <ellipse cx="200" cy="148" rx="72" ry="72" fill="rgba(219,161,89,0.05)"/>
    <rect x="184" y="66" width="32" height="14" rx="6" fill="rgba(219,161,89,0.22)" stroke="rgba(219,161,89,0.65)" strokeWidth="1.5"/>
    <rect x="181" y="79" width="38" height="18" rx="3" fill="rgba(219,161,89,0.07)" stroke="rgba(219,161,89,0.5)" strokeWidth="1.5"/>
    <path d="M181,97 C168,104 162,118 162,132 L162,208 Q162,222 176,222 L224,222 Q238,222 238,208 L238,132 C238,118 232,104 219,97 Z" fill="rgba(219,161,89,0.07)" stroke="rgba(219,161,89,0.55)" strokeWidth="1.8"/>
    <path d="M164,175 Q182,168 200,172 Q218,176 236,170" fill="none" stroke="rgba(252,201,149,0.45)" strokeWidth="1.5" strokeLinecap="round"/>
    <text x="200" y="200" textAnchor="middle" fill="rgba(252,201,149,0.7)" fontSize="9" fontFamily="monospace" letterSpacing="3" fontWeight="600">DAIRY</text>
    <text x="200" y="260" textAnchor="middle" fill="rgba(219,161,89,0.28)" fontSize="10" fontFamily="monospace" letterSpacing="4">GREECE · GR</text>
  </svg>
);

const PhotoPT = () => (
  <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" style={{display:'block'}}>
    <defs>
      <linearGradient id="pt-g" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#021520"/>
        <stop offset="100%" stopColor="#042a3a"/>
      </linearGradient>
    </defs>
    <rect width="400" height="300" fill="url(#pt-g)"/>
    <ellipse cx="200" cy="148" rx="72" ry="72" fill="rgba(68,161,160,0.05)"/>
    <circle cx="200" cy="150" r="58" fill="rgba(68,161,160,0.09)" stroke="rgba(146,221,219,0.55)" strokeWidth="1.8"/>
    <line x1="200" y1="92" x2="200" y2="208" stroke="rgba(146,221,219,0.18)" strokeWidth="1"/>
    <line x1="150" y1="121" x2="250" y2="179" stroke="rgba(146,221,219,0.18)" strokeWidth="1"/>
    <line x1="150" y1="179" x2="250" y2="121" stroke="rgba(146,221,219,0.18)" strokeWidth="1"/>
    <circle cx="200" cy="94" r="6" fill="rgba(68,161,160,0.15)" stroke="rgba(146,221,219,0.5)" strokeWidth="1.5"/>
    <path d="M200,88 L200,76" stroke="rgba(146,221,219,0.5)" strokeWidth="2" strokeLinecap="round"/>
    <path d="M200,80 C208,71 222,72 226,81 C219,85 208,81 200,80 Z" fill="rgba(188,220,139,0.18)" stroke="rgba(188,220,139,0.55)" strokeWidth="1.4"/>
    <text x="200" y="230" textAnchor="middle" fill="rgba(146,221,219,0.7)" fontSize="9" fontFamily="monospace" letterSpacing="3" fontWeight="600">FRUITS</text>
    <text x="200" y="260" textAnchor="middle" fill="rgba(68,161,160,0.32)" fontSize="10" fontFamily="monospace" letterSpacing="4">PORTUGAL · PT</text>
  </svg>
);

const PhotoEU = () => (
  <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" style={{display:'block'}}>
    <defs>
      <radialGradient id="eu-g" cx="50%" cy="47%">
        <stop offset="0%" stopColor="#073B4C"/>
        <stop offset="100%" stopColor="#042026"/>
      </radialGradient>
    </defs>
    <rect width="400" height="300" fill="url(#eu-g)"/>
    <ellipse cx="200" cy="140" rx="90" ry="70" fill="rgba(219,161,89,0.04)"/>
    <path d="M118,140 C130,106 168,98 200,106 C232,98 268,108 278,140 C268,172 232,182 200,174 C168,182 130,174 118,140 Z" fill="rgba(219,161,89,0.08)" stroke="rgba(219,161,89,0.55)" strokeWidth="1.8"/>
    <path d="M278,140 L312,114 L300,140 L312,166 Z" fill="rgba(219,161,89,0.12)" stroke="rgba(219,161,89,0.55)" strokeWidth="1.5" strokeLinejoin="round"/>
    <path d="M162,108 C178,88 210,86 222,106" fill="rgba(219,161,89,0.06)" stroke="rgba(219,161,89,0.45)" strokeWidth="1.5" strokeLinecap="round"/>
    <path d="M158,152 C164,164 175,168 182,162" fill="none" stroke="rgba(219,161,89,0.38)" strokeWidth="1.4" strokeLinecap="round"/>
    <circle cx="150" cy="134" r="10" fill="rgba(219,161,89,0.1)" stroke="rgba(219,161,89,0.55)" strokeWidth="1.5"/>
    <circle cx="150" cy="134" r="4" fill="rgba(219,161,89,0.4)"/>
    <path d="M180,112 C186,124 186,156 180,168" fill="none" stroke="rgba(219,161,89,0.2)" strokeWidth="1" strokeLinecap="round"/>
    <path d="M204,108 C210,122 210,158 204,172" fill="none" stroke="rgba(219,161,89,0.2)" strokeWidth="1" strokeLinecap="round"/>
    <path d="M228,112 C233,126 233,154 228,168" fill="none" stroke="rgba(219,161,89,0.2)" strokeWidth="1" strokeLinecap="round"/>
    <text x="195" y="222" textAnchor="middle" fill="rgba(252,201,149,0.7)" fontSize="9" fontFamily="monospace" letterSpacing="3" fontWeight="600">FISH</text>
    <text x="195" y="260" textAnchor="middle" fill="rgba(219,161,89,0.28)" fontSize="10" fontFamily="monospace" letterSpacing="4">GREECE · GR</text>
  </svg>
);

const PHOTOS = [PhotoFI, PhotoGR, PhotoPT, PhotoEU];

// ── Shared layout components ──────────────────────────────────────────────────
const CSBackground = () => (
  <div className="bg bg-gradient">
    <div className="aurora a1" />
    <div className="aurora a2" />
    <div className="aurora a3" />
    <div className="grain" />
  </div>
);

const CSLangSwitcher = ({ lang, setLang }) => {
  const langs = [['en','EN'],['fi','FI'],['el','EL'],['pt','PT']];
  return (
    <div className="lang-switcher" title="Select language">
      {langs.map(([code, label]) => (
        <button key={code}
          className={'lang-btn' + (lang === code ? ' active' : '')}
          onClick={() => setLang(code)}>
          {label}
        </button>
      ))}
    </div>
  );
};

const CSHeader = ({ lang, setLang }) => (
  <header className="site-header">
    <div className="logo-slot" aria-label="Header logo">
      {CS_HEADER_LOGO
        ? (/* HEADER LOGO SIZE (case studies page) — change maxHeight / maxWidth here */
           <img src={CS_HEADER_LOGO} alt="Logo" style={{maxHeight:'200px', maxWidth:'300px', objectFit:'contain', display:'block'}} />)
        : (<>
            <svg className="logo-slot-bg" viewBox="0 0 200 56" preserveAspectRatio="none" aria-hidden="true">
              <defs>
                <pattern id="cs-logo-stripe" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <line x1="0" y1="0" x2="0" y2="8" stroke="rgba(146,221,219,0.18)" strokeWidth="2"/>
                </pattern>
              </defs>
              <rect x="0.5" y="0.5" width="199" height="55" fill="url(#cs-logo-stripe)" stroke="rgba(146,221,219,0.28)" strokeDasharray="3 3"/>
            </svg>
            <span className="logo-slot-label mono">LOGO&nbsp;·&nbsp;200×56</span>
          </>)
      }
    </div>
    <CSLangSwitcher lang={lang} setLang={setLang} />
  </header>
);

const CSFooter = () => (
  <footer className="site-footer">
    <div className="foot-left">
      <div className="brand">
        <span style={{color:'#F4EFE6'}}>GROCERY<span style={{color:'#DBA159'}}>sim</span><span style={{color:'rgba(244,239,230,0.6)',fontWeight:500}}>&nbsp;ABM</span></span>
      </div>
      <p style={{fontSize:'13px',color:'rgba(244,239,230,0.6)',marginTop:'8px'}}>© 2026 · SecureFood Consortium · Horizon Europe</p>
    </div>
    <div className="foot-right">
      <span className="mono" style={{color:'rgba(244,239,230,0.35)',fontSize:'11px'}}>SECUREFOOD</span>
    </div>
  </footer>
);

// ── Single case-study card ────────────────────────────────────────────────────
const CaseStudyCard = ({ card, flag, Photo, launchBtn, comingSoon }) => {
  const isActive = card.status === 'active';
  const handleLaunch = () => {
    if (!isActive) return;
    try { window.parent.postMessage({type:'launch_grocerysim'}, '*'); } catch(e) {}
  };
  return (
    <div className={'cs-card ' + (isActive ? 'active' : 'inactive')}>
      <div className="cs-photo">
        <div className="cs-photo-inner"><Photo /></div>
        <div className={'cs-badge ' + (isActive ? 'active' : 'soon')}>
          {isActive ? 'LIVE' : 'SOON'}
        </div>
        <div className="cs-country">{flag}</div>
      </div>
      <div className="cs-body">
        <div className="cs-card-tag mono">{card.tag}</div>
        <div className="cs-card-title">{card.title}</div>
        <div className="cs-card-desc">{card.desc}</div>
        <button
          className={'cs-card-btn ' + (isActive ? 'launch' : 'disabled')}
          onClick={handleLaunch}
          disabled={!isActive}
        >
          {isActive ? (
            <>
              {launchBtn}
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M2 6h8m0 0L7 3M10 6L7 9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </>
          ) : comingSoon}
        </button>
      </div>
    </div>
  );
};

// ── Case Studies Page ─────────────────────────────────────────────────────────
const CaseStudiesPage = ({ t, lang, setLang }) => {
  const handleBack = () => {
    try { window.parent.postMessage({type:'launch_back'}, '*'); } catch(e) {}
  };
  return (
    <div className="page">
      <CSBackground />
      <CSHeader lang={lang} setLang={setLang} />
      <main>
        <div className="cs-hero">
          <button className="cs-hero-back" onClick={handleBack}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            {t.backBtn}
          </button>
          <h1 className="cs-title">{t.pageTitle}</h1>
          <p className="cs-subtitle">{t.pageSub}</p>
        </div>
        <div className="cs-grid">
          {t.cards.map((card, i) => (
            <CaseStudyCard
              key={i}
              card={card}
              flag={CARD_FLAGS[i]}
              Photo={PHOTOS[i]}
              launchBtn={t.launchBtn}
              comingSoon={t.comingSoon}
            />
          ))}
        </div>
      </main>
      <CSFooter />
    </div>
  );
};

// ── App root ──────────────────────────────────────────────────────────────────
const CaseStudiesApp = () => {
  const _storedLang = (() => { try { return window.parent.sessionStorage.getItem('grocerysim_lang'); } catch(e) { return null; } })();
  const initLang = (_storedLang && CS_TRANS[_storedLang]) ? _storedLang : ((typeof INITIAL_LANG !== 'undefined' && CS_TRANS[INITIAL_LANG]) ? INITIAL_LANG : 'en');
  const [lang, setLangState] = useState(initLang);
  const setLang = (code) => { setLangState(code); try { window.parent.sessionStorage.setItem('grocerysim_lang', code); } catch(e) {} };
  const t = CS_TRANS[lang];
  return <CaseStudiesPage t={t} lang={lang} setLang={setLang} />;
};

ReactDOM.createRoot(document.getElementById('root')).render(<CaseStudiesApp />);
"""


def render_landing_page():
    """Display the full GROCERYsim landing page."""
    st.markdown("""
    <style>
        /* ── Hide ALL Streamlit chrome ── */
        section[data-testid="stSidebar"],
        div[data-testid="stSidebarCollapsedControl"],
        div[data-testid="collapsedControl"],
        button[data-testid="baseButton-headerNoPadding"],
        [data-testid*="Sidebar"] { display: none !important; visibility: hidden !important; }
        header[data-testid="stHeader"] { display: none !important; }
        #MainMenu, footer { display: none !important; }

        /* ── Dark background ── */
        .stApp,
        .stApp > div,
        section.main,
        div.block-container { background-color: #042026 !important; }
        .main .block-container {
            background-color: #042026 !important;
            padding: 0 !important;
            max-width: 100% !important;
        }
        div[data-testid="stVerticalBlock"] { gap: 0 !important; background: #042026 !important; }

        /* ── Strip component iframe border ── */
        iframe { border: none !important; outline: none !important;
                 display: block !important; }
        .element-container { padding: 0 !important; margin: 0 !important;
                              background: transparent !important; }

        /* ── Hide hidden nav trigger buttons ── */
        div[data-testid="stButton"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    _lu = {
        "header":   _logo_uri("GROCERYsim.png"),
        "hero0":    _logo_uri("EU.png"),
        "hero1":    _logo_uri("SecureFood.png"),
        "hero2":    _logo_uri("IAMO.png"),
        "hero3":    _logo_uri("Logo_lab.png"),
    }
    _pdf_uri = _logo_uri("GROCERYsim_User_Manual.pdf")  # reuses same base64 helper

    landing_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Figtree:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,300&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>{_LANDING_CSS}</style>
</head>
<body>
  <div id="root"></div>
  <script>
    var LOGO_URIS = {{
      header:   "{_lu['header']}",
      hero:     ["{_lu['hero0']}", "{_lu['hero1']}", "{_lu['hero2']}", "{_lu['hero3']}"],
      partners: ['','','','','','','','']
    }};
    var PDF_URI = "{_pdf_uri}";
  </script>
  <script src="https://unpkg.com/react@18.3.1/umd/react.development.js" crossorigin="anonymous"></script>
  <script src="https://unpkg.com/react-dom@18.3.1/umd/react-dom.development.js" crossorigin="anonymous"></script>
  <script src="https://unpkg.com/@babel/standalone@7.29.0/babel.min.js" crossorigin="anonymous"></script>
  <script type="text/babel">
    {_GROCERY_SIM_JSX}
    {_LANDING_APP_JSX}
  </script>
  <script>
  (function(){{
    function sendH(){{
      var h = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
      window.parent.postMessage({{type:'grocerysim_resize', height:h}}, '*');
    }}
    window.addEventListener('load', function(){{ sendH(); setTimeout(sendH,600); setTimeout(sendH,1800); }});
    window.addEventListener('resize', sendH);
    if(typeof ResizeObserver!=='undefined') new ResizeObserver(sendH).observe(document.body);
  }})();
  </script>
</body>
</html>"""

    # Hidden nav trigger — clicked programmatically by the JS bridge below
    if st.button("→cases", key="cases_nav_btn"):
        st.session_state["page"] = "case_studies"
        st.rerun()

    components.html(landing_html, height=1200)

    # JS bridge: listen for postMessage from React iframe, then click the hidden button
    components.html("""<script>
(function(){
  var NAV = {'launch_case_studies': '→cases'};
  var obs = new MutationObserver(function(){
    window.parent.document.querySelectorAll('[data-testid="stButton"]').forEach(function(c){
      var lbl = (c.querySelector('button p, button') || {}).textContent || '';
      if(Object.values(NAV).some(function(v){ return lbl.trim() === v; }))
        c.style.display = 'none';
    });
  });
  try { obs.observe(window.parent.document.body, {childList:true, subtree:true}); } catch(e){}
  window.parent.addEventListener('message', function(e){
    if(!e.data) return;
    if(NAV[e.data.type]){
      var target = NAV[e.data.type];
      window.parent.document.querySelectorAll('button').forEach(function(b){
        if((b.textContent || '').trim() === target) b.click();
      });
    }
    if(e.data.type === 'grocerysim_resize' && e.data.height){
      window.parent.document.querySelectorAll('iframe').forEach(function(fr){
        try { if(fr.contentWindow === e.source){ fr.style.height = e.data.height+'px'; fr.style.minHeight = e.data.height+'px'; } } catch(ex){}
      });
    }
  });
})();
</script>""", height=0)

def render_case_studies_page():
    """Display the Case Studies hub — 4 scenario cards, back navigation, multi-language."""
    st.markdown("""
    <style>
        /* ── Hide ALL Streamlit chrome ── */
        section[data-testid="stSidebar"],
        div[data-testid="stSidebarCollapsedControl"],
        div[data-testid="collapsedControl"],
        [data-testid*="Sidebar"] { display: none !important; visibility: hidden !important; }
        header[data-testid="stHeader"] { display: none !important; }
        #MainMenu, footer { display: none !important; }

        /* ── Dark background ── */
        .stApp, .stApp > div, section.main,
        div.block-container { background-color: #042026 !important; }
        .main .block-container {
            background-color: #042026 !important;
            padding: 0 !important;
            max-width: 100% !important;
        }
        div[data-testid="stVerticalBlock"] { gap: 0 !important; background: #042026 !important; }
        iframe { border: none !important; outline: none !important; display: block !important; }
        .element-container { padding: 0 !important; margin: 0 !important;
                              background: transparent !important; }

        /* ── Hide hidden nav trigger buttons ── */
        div[data-testid="stButton"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    cs_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Figtree:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,300&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>{_LANDING_CSS}</style>
</head>
<body>
  <div id="root"></div>
  <script>var INITIAL_LANG = 'en'; var CS_HEADER_URI = "{_logo_uri('GROCERYsim.png')}";</script>
  <script src="https://unpkg.com/react@18.3.1/umd/react.development.js" crossorigin="anonymous"></script>
  <script src="https://unpkg.com/react-dom@18.3.1/umd/react-dom.development.js" crossorigin="anonymous"></script>
  <script src="https://unpkg.com/@babel/standalone@7.29.0/babel.min.js" crossorigin="anonymous"></script>
  <script type="text/babel">
    {_CASE_STUDIES_JSX}
  </script>
  <script>
  (function(){{
    function sendH(){{
      var h = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
      window.parent.postMessage({{type:'grocerysim_resize', height:h}}, '*');
    }}
    window.addEventListener('load', function(){{ sendH(); setTimeout(sendH,600); setTimeout(sendH,1800); }});
    window.addEventListener('resize', sendH);
    if(typeof ResizeObserver!=='undefined') new ResizeObserver(sendH).observe(document.body);
  }})();
  </script>
</body>
</html>"""

    # Hidden nav triggers — clicked programmatically by the JS bridge below
    for _lc in ["en", "fi", "el", "pt"]:
        if st.button(f"→main_{_lc}", key=f"main_nav_{_lc}"):
            st.session_state["lang"] = _lc
            st.session_state["page"] = "main"
            st.rerun()
    if st.button("→back", key="back_nav_btn"):
        st.session_state["page"] = "landing"
        st.rerun()

    components.html(cs_html, height=1020)

    # JS bridge: hide trigger buttons and route React postMessages to them
    components.html("""<script>
(function(){
  var HIDE = ['→main_en','→main_fi','→main_el','→main_pt','→back'];
  var obs = new MutationObserver(function(){
    window.parent.document.querySelectorAll('[data-testid="stButton"]').forEach(function(c){
      var lbl = (c.querySelector('button p, button') || {}).textContent || '';
      if(HIDE.some(function(v){ return lbl.trim() === v; }))
        c.style.display = 'none';
    });
  });
  try { obs.observe(window.parent.document.body, {childList:true, subtree:true}); } catch(e){}
  window.parent.addEventListener('message', function(e){
    if(!e.data) return;
    if(e.data.type === 'launch_grocerysim'){
      var lang = 'en';
      try { lang = window.parent.sessionStorage.getItem('grocerysim_lang') || 'en'; } catch(e2){}
      var target = '→main_' + lang;
      window.parent.document.querySelectorAll('button').forEach(function(b){
        if((b.textContent || '').trim() === target) b.click();
      });
    } else if(e.data.type === 'launch_back'){
      window.parent.document.querySelectorAll('button').forEach(function(b){
        if((b.textContent || '').trim() === '→back') b.click();
      });
    }
    if(e.data.type === 'grocerysim_resize' && e.data.height){
      window.parent.document.querySelectorAll('iframe').forEach(function(fr){
        try { if(fr.contentWindow === e.source){ fr.style.height = e.data.height+'px'; fr.style.minHeight = e.data.height+'px'; } } catch(ex){}
      });
    }
  });
})();
</script>""", height=0)


# ===========================================================================
# SECUREFOOD SCENARIO PAGE
# Dedicated full-screen simulator for the SecureFood / Horizon Europe climate
# disruption scenario.  Two user profiles:
#   • Supply Chain Actor  — revenue, inventory, stockouts, panic dynamics
#   • Policy Maker        — consumer welfare, equity, food security, policy levers
# ===========================================================================

# ── Shared utilities ──────────────────────────────────────────────────────────

def _sf_crisis_band(fig, cri_start: int, cri_end: int, days: int):
    end = min(cri_end if cri_end > cri_start else days, days)
    fig.add_vrect(
        x0=cri_start, x1=end,
        fillcolor="rgba(220,50,50,0.10)", line_width=0,
        annotation_text="Crisis", annotation_position="top left",
        annotation_font_size=9, annotation_font_color="#c0392b",
    )
    return fig


def _sf_analysis_box(text: str):
    st.info(f"📊 **Analysis** — {text}")


def _sf_summary_box(title: str, bullets: list, recommendation: str):
    bullet_html = "".join(f"<li style='margin-bottom:6px'>{b}</li>" for b in bullets)
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d2b3e 0%,#0a3825 100%);
            border-left:5px solid #27ae60;border-radius:10px;
            padding:24px 28px;margin-top:24px;color:#ecf0f1;">
  <h3 style="color:#2ecc71;margin:0 0 14px 0;font-size:1.1rem;">📋 {title}</h3>
  <ul style="margin:0 0 16px 0;padding-left:20px;line-height:1.9;font-size:0.93rem;">
    {bullet_html}
  </ul>
  <div style="background:rgba(255,255,255,0.07);border-radius:6px;
              padding:12px 16px;border-left:3px solid #f39c12;">
    <strong style="color:#f39c12;">💡 Recommendation: </strong>
    <span style="font-size:0.92rem;">{recommendation}</span>
  </div>
</div>""", unsafe_allow_html=True)


def _sf_run_simulation(params: dict):
    """Run baseline + crisis models and return result dict, or None on failure."""
    try:
        m_base   = _make_model(params, is_crisis=False, seed=42)
        m_crisis = _make_model(params, is_crisis=True,  seed=42)
        agg_rows, prod_rows = [], []
        for day in range(1, params["days"] + 1):
            m_base.step()
            m_crisis.step()
            agg_b, pb = _collect_model_day(m_base,   day, "Baseline")
            agg_c, pc = _collect_model_day(m_crisis, day, "Crisis")
            agg_rows += [agg_b, agg_c]
            prod_rows += pb + pc
        return {
            "df":      pd.DataFrame(agg_rows),
            "df_prod": pd.DataFrame(prod_rows),
            "params":  params,
        }
    except Exception as e:
        st.error(f"Simulation error: {e}")
        return None


# ── Supply Chain results renderer ─────────────────────────────────────────────

def _render_sf_sc_results(data: dict):
    df      = data["df"]
    df_prod = data["df_prod"]
    p       = data["params"]

    df_b = df[df["Scenario"] == "Baseline"].copy().reset_index(drop=True)
    df_c = df[df["Scenario"] == "Crisis"].copy().reset_index(drop=True)

    cri_start = p["cri_start"]
    cri_dur   = p["cri_duration"]
    days      = p["days"]
    cri_end   = (cri_start + cri_dur) if cri_dur > 0 else days

    # ── Pre-compute metrics ────────────────────────────────────────────────────
    rev_b        = df_b["Revenue"].sum()
    rev_c        = df_c["Revenue"].sum()
    rev_loss     = rev_b - rev_c
    rev_loss_pct = 100.0 * rev_loss / max(rev_b, 1.0)
    lost_total   = df_c["LostSales"].sum()
    peak_panic   = float(df_c["PanicLevel"].max())
    mean_panic   = float(df_c["PanicLevel"].mean())
    waste_delta  = df_c["Waste"].sum() - df_b["Waste"].sum()
    nom_gain     = df_c["NominalRevenue"].sum() - df_b["Revenue"].sum()
    avg_p_b      = float(df_b["AvgPrice"].mean())
    avg_p_c      = float(df_c["AvgPrice"].mean())
    price_delta_pct = 100.0 * (avg_p_c / max(avg_p_b, 0.01) - 1.0)

    merged   = df_b[["Day","Revenue"]].merge(df_c[["Day","Revenue"]], on="Day", suffixes=("_b","_c"))
    post_cr  = merged[merged["Day"] > cri_end]
    rec_rows = post_cr[post_cr["Revenue_c"] >= post_cr["Revenue_b"] * 0.95]
    recovery_days = int(rec_rows.iloc[0]["Day"] - cri_end) if len(rec_rows) else None

    peak_vol_loss_day = int(df_b.loc[(df_b["Revenue"] - df_c["Revenue"]).idxmax(), "Day"])
    peak_lost_day     = int(df_c.loc[df_c["LostSales"].idxmax(), "Day"])
    peak_lost_val     = float(df_c["LostSales"].max())

    panic_threshold_day_val = None
    if (df_c["PanicLevel"] > 0.3).any():
        panic_threshold_day_val = int(df_c.loc[df_c["PanicLevel"] > 0.3, "Day"].iloc[0])
    panic_peak_day = int(df_c.loc[df_c["PanicLevel"].idxmax(), "Day"])
    sp_peak        = float(df_c["StockpilePressure"].max())

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Simulation Results")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue Loss", f"€{rev_loss:,.0f}",
              f"−{rev_loss_pct:.1f}% vs baseline", delta_color="inverse")
    k2.metric("Stockout Losses", f"€{lost_total:,.0f}",
              "unrecoverable demand", delta_color="inverse")
    k3.metric("Peak Panic Level", f"{peak_panic:.2f} / 1.0",
              f"mean {mean_panic:.2f}", delta_color="inverse")
    k4.metric("Supply Recovery",
              f"{recovery_days} days" if recovery_days else "Not within horizon",
              "after crisis end" if recovery_days else "extend simulation",
              delta_color="off")

    # ── Chart 1: Revenue Impact ───────────────────────────────────────────────
    st.markdown("#### 1 · Revenue Impact Timeline")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_b["Day"], y=df_b["Revenue"],
                              name="Baseline (constant price)",
                              line=dict(color="#2980b9", width=2.5)))
    fig1.add_trace(go.Scatter(x=df_c["Day"], y=df_c["Revenue"],
                              name="Crisis (constant price)",
                              line=dict(color="#e74c3c", width=2.5)))
    fig1.add_trace(go.Scatter(x=df_c["Day"], y=df_c["NominalRevenue"],
                              name="Crisis nominal (inflated prices)",
                              line=dict(color="#e67e22", width=1.5, dash="dot")))
    fig1.add_trace(go.Scatter(
        x=list(df_b["Day"]) + list(df_b["Day"])[::-1],
        y=list(df_b["Revenue"]) + list(df_c["Revenue"])[::-1],
        fill="toself", fillcolor="rgba(231,76,60,0.10)",
        line=dict(width=0), name="Revenue gap",
    ))
    fig1 = _sf_crisis_band(fig1, cri_start, cri_end, days)
    fig1.update_layout(
        template="plotly_white", height=400,
        xaxis_title="Simulation Day", yaxis_title="Daily Revenue (€)",
        title="Constant-Price Revenue vs Nominal Revenue — Inflation-Volume Decomposition",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig1, use_container_width=True)
    nom_dir = "rose" if nom_gain > 0 else "fell"
    _sf_analysis_box(
        f"Constant-price revenue fell by **€{rev_loss:,.0f} ({rev_loss_pct:.1f}%)**, measuring the genuine "
        f"volume shortfall. Nominal revenue (orange dotted) {nom_dir} by **€{abs(nom_gain):,.0f}** due to the "
        f"{p['inf']:.0f}% price inflation component — masking the underlying volume loss. "
        f"This decomposition is essential: actors benchmarking on nominal revenue alone will underestimate "
        f"the operational impact. Peak single-day volume loss occurred on **Day {peak_vol_loss_day}**."
    )

    # ── Chart 2: Stockout Events ──────────────────────────────────────────────
    st.markdown("#### 2 · Stockout Events & Lost Sales")
    df_c2 = df_c.copy()
    df_c2["CumLost"] = df_c2["LostSales"].cumsum()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df_c2["Day"], y=df_c2["LostSales"],
                          name="Daily Lost Sales (Crisis)",
                          marker_color="rgba(231,76,60,0.65)", marker_line_width=0))
    fig2.add_trace(go.Scatter(x=df_c2["Day"], y=df_c2["CumLost"],
                              name="Cumulative Lost Sales",
                              line=dict(color="#922b21", width=2.5), yaxis="y2"))
    fig2.add_trace(go.Scatter(x=df_b["Day"], y=df_b["LostSales"],
                              name="Baseline Lost Sales",
                              line=dict(color="#aab7b8", width=1.2, dash="dot")))
    fig2 = _sf_crisis_band(fig2, cri_start, cri_end, days)
    fig2.update_layout(
        template="plotly_white", height=360,
        xaxis_title="Simulation Day", yaxis_title="Lost Sales €/day",
        yaxis2=dict(title="Cumulative (€)", overlaying="y", side="right"),
        title="Revenue Lost to Stockouts — Daily Events and Cumulative Accumulation",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig2, use_container_width=True)
    _sf_analysis_box(
        f"Cumulative stockout losses reached **€{lost_total:,.0f}**. Peak occurred on **Day {peak_lost_day}** "
        f"(€{peak_lost_val:,.0f}), driven by the combination of panic-buying demand spikes and "
        f"delayed replenishment from the {p['dis']}-day supply disruption. "
        f"Stockout events represent unrecoverable demand — empirical studies show 21–43% of consumers "
        f"switch brands permanently after a stockout ([Gruen et al., 2002](https://www.supplychain247.com/images/pdfs/GMA_2002_Worldwide_OOS_Study.pdf)). "
        f"Grey dotted line = baseline lost sales, isolating the crisis-attributable component."
    )

    # ── Chart 3: Inventory Availability ──────────────────────────────────────
    st.markdown("#### 3 · Inventory Availability by Product Category")
    if not df_prod.empty and "Category" in df_prod.columns:
        fill_b = (df_prod[df_prod["Scenario"] == "Baseline"]
                  .groupby(["Day", "Category"])["Shelf"].sum().reset_index())
        fill_c = (df_prod[df_prod["Scenario"] == "Crisis"]
                  .groupby(["Day", "Category"])["Shelf"].sum().reset_index())
        cats    = sorted(fill_c["Category"].dropna().unique())
        palette = ["#2980b9","#27ae60","#e67e22","#8e44ad","#16a085","#d35400","#2c3e50"]
        fig3    = go.Figure()
        for i, cat in enumerate(cats):
            col = palette[i % len(palette)]
            cb  = fill_b[fill_b["Category"] == cat]
            cc  = fill_c[fill_c["Category"] == cat]
            fig3.add_trace(go.Scatter(x=cb["Day"], y=cb["Shelf"],
                                      name=f"{cat} (baseline)",
                                      line=dict(color=col, width=1.2, dash="dash"), opacity=0.55))
            fig3.add_trace(go.Scatter(x=cc["Day"], y=cc["Shelf"],
                                      name=f"{cat} (crisis)",
                                      line=dict(color=col, width=2.2)))
        fig3 = _sf_crisis_band(fig3, cri_start, cri_end, days)
        fig3.update_layout(
            template="plotly_white", height=420,
            xaxis_title="Simulation Day", yaxis_title="Total Units on Shelf",
            title="Shelf Stock by Category — Baseline (dashed) vs Crisis (solid)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig3, use_container_width=True)
        worst_cat, worst_drop = "—", 0.0
        for cat in cats:
            ab = fill_b[fill_b["Category"] == cat]["Shelf"].mean()
            ac = fill_c[fill_c["Category"] == cat]["Shelf"].mean()
            drop = (ab - ac) / max(ab, 1.0)
            if drop > worst_drop:
                worst_drop, worst_cat = drop, cat
        _sf_analysis_box(
            f"**{worst_cat}** was the most disrupted category, with shelf availability falling by "
            f"**{worst_drop*100:.0f}%** on average during the crisis. "
            f"The disruption propagates through the supply chain after **{p['lead']} day(s)** (the lead time), "
            f"when delayed deliveries begin reducing replenishment. Short-shelf-life categories are "
            f"disproportionately affected because safety stock cannot be pre-built without spoilage risk. "
            f"Dashed lines = baseline reference; the gap measures crisis-attributable depletion."
        )

    # ── Chart 4: Panic & Hoarding ─────────────────────────────────────────────
    st.markdown("#### 4 · Consumer Panic & Hoarding Dynamics")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df_c["Day"], y=df_c["PanicLevel"],
                              name="Panic Level (Crisis)",
                              fill="tozeroy", fillcolor="rgba(231,76,60,0.10)",
                              line=dict(color="#e74c3c", width=2.5)))
    fig4.add_trace(go.Scatter(x=df_b["Day"], y=df_b["PanicLevel"],
                              name="Panic Level (Baseline)",
                              line=dict(color="#5dade2", width=1.2, dash="dash")))
    fig4.add_trace(go.Scatter(x=df_c["Day"], y=df_c["StockpilePressure"],
                              name="Stockpile Pressure (Crisis)",
                              line=dict(color="#e67e22", width=2.0), yaxis="y2"))
    fig4.add_hline(y=0.3, line_dash="dot", line_color="#c0392b",
                   annotation_text="Hoarding threshold (0.30)",
                   annotation_font_size=9, annotation_position="bottom right")
    fig4 = _sf_crisis_band(fig4, cri_start, cri_end, days)
    fig4.update_layout(
        template="plotly_white", height=360,
        xaxis_title="Simulation Day",
        yaxis=dict(title="Panic Level (0–1)", range=[0, 1.05]),
        yaxis2=dict(title="Stockpile Pressure (0–1)", overlaying="y",
                    side="right", range=[0, 1.05]),
        title="Consumer Panic Level and Stockpile Pressure — Behavioural Demand Amplification",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig4, use_container_width=True)
    if panic_threshold_day_val is not None:
        panic_txt = (f"The 0.30 hoarding threshold was crossed on **Day {panic_threshold_day_val}**, "
                     f"driving excess purchases that amplified supply depletion beyond the supply-side shock alone.")
    else:
        panic_txt = "Panic remained below the 0.30 hoarding threshold throughout the simulation."
    _sf_analysis_box(
        f"Consumer panic peaked at **{peak_panic:.2f}/1.0** (Day {panic_peak_day}). {panic_txt} "
        f"Stockpile pressure (quasi-hyperbolic discounting, [O'Donoghue & Rabin 1999](https://www.jstor.org/stable/116981?seq=1)) peaked at "
        f"**{sp_peak:.2f}**, indicating consumers were building home inventories in anticipation of "
        f"further scarcity. For supply chain actors, this demand amplification — the 'bullwhip effect' "
        f"([Lee et al., 1997](https://www.jstor.org/stable/2634565?seq=1)) — means true consumer demand was below store orders during the panic phase. "
        f"Expect a demand trough post-crisis; reduce replenishment orders 10–15% during recovery."
    )

    # ── Summary box ───────────────────────────────────────────────────────────
    rec_str = (f"**{recovery_days} days** post-crisis to 95% of baseline revenue"
               if recovery_days else "**not within simulation horizon** — extend run to model full recovery")
    waste_dir = "increased" if waste_delta > 0 else "reduced"
    _sf_summary_box(
        "Supply Chain Impact Summary — SecureFood Climate Scenario",
        [
            f"Constant-price revenue loss: <b>€{rev_loss:,.0f}</b> ({rev_loss_pct:.1f}% of baseline) — genuine volume shortfall",
            f"Nominal revenue <b>{'rose' if nom_gain > 0 else 'fell'}</b> by €{abs(nom_gain):,.0f} due to inflation — verify with constant-price reporting",
            f"Stockout losses: <b>€{lost_total:,.0f}</b> of unrecoverable demand",
            f"Average retail price: <b>€{avg_p_c:.2f}</b> vs baseline <b>€{avg_p_b:.2f}</b> (+{price_delta_pct:.0f}% inflation pass-through)",
            f"Peak consumer panic: <b>{peak_panic:.2f}/1.0</b> — {'severe' if peak_panic >= 0.5 else 'moderate'} disruption with {'significant' if sp_peak > 0.4 else 'limited'} hoarding pressure",
            f"Waste delta: <b>{waste_delta:+,.0f} units</b> vs baseline — {waste_dir} ({'post-panic excess inventory decays' if waste_delta > 0 else 'stockouts prevent expiry'})",
            f"Supply recovery: {rec_str}",
        ],
        f"To maintain a 95% service level under a {p['dis']}-day disruption at {p['inf']:.0f}% inflation: "
        f"(1) Raise reorder point to ≥{min(60, int(p['reorder']*100 + p['dis']*2.5))}% to absorb lead-time extension; "
        f"(2) Pre-position ~{p['dis']} days of additional safety stock for the most affected categories; "
        f"(3) Pre-negotiate dual-sourcing contracts to cap single-supplier lead-time exposure; "
        f"(4) Reduce replenishment orders 10–15% during post-crisis recovery to counteract the bullwhip demand trough.",
    )


# ── Policy Maker results renderer ─────────────────────────────────────────────

def _render_sf_pm_results(data: dict):
    df      = data["df"]
    df_prod = data["df_prod"]
    p       = data["params"]

    df_b = df[df["Scenario"] == "Baseline"].copy().reset_index(drop=True)
    df_c = df[df["Scenario"] == "Crisis"].copy().reset_index(drop=True)

    cri_start = p["cri_start"]
    cri_dur   = p["cri_duration"]
    days      = p["days"]
    cri_end   = (cri_start + cri_dur) if cri_dur > 0 else days
    pc        = p.get("policy_cfg", {})

    # ── Pre-compute metrics ────────────────────────────────────────────────────
    peak_stress    = float(df_c["FoodStressedPct"].max()) * 100
    base_stress    = float(df_b["FoodStressedPct"].mean()) * 100
    peak_budgexh_lo = float(df_c["BudgetExh_Low"].max()) * 100
    peak_budgexh_hi = float(df_c["BudgetExh_High"].max()) * 100
    mean_gini_c    = float(df_c["GiniAccess"].mean())
    mean_gini_b    = float(df_b["GiniAccess"].mean())
    import_dep_b   = float(df_b["ImportDepPct"].mean()) * 100
    import_dep_c   = float(df_c["ImportDepPct"].mean()) * 100
    fulfill_lo_c   = float(df_c["Fulfillment_Low"].mean()) * 100
    fulfill_hi_c   = float(df_c["Fulfillment_High"].mean()) * 100
    fulfill_gap    = fulfill_hi_c - fulfill_lo_c
    fies_lo_peak   = float(df_c["FIESSevere_Low"].max()) * 100
    fies_lo_base   = float(df_b["FIESSevere_Low"].mean()) * 100
    fies_delta     = fies_lo_peak - fies_lo_base
    dom_sum_b = df_b["DomesticSales"].sum() + df_b["ImportSales"].sum()
    dom_sum_c = df_c["DomesticSales"].sum() + df_c["ImportSales"].sum()
    dom_share_b    = df_b["DomesticSales"].sum() / max(dom_sum_b, 1) * 100
    dom_share_c    = df_c["DomesticSales"].sum() / max(dom_sum_c, 1) * 100
    dom_change     = dom_share_c - dom_share_b
    low_below_80   = int((df_c["Fulfillment_Low"] < 0.80).sum())

    active_policies = [k for k in ["subsidy_active","fat_tax_active","labelling_active"] if pc.get(k)]
    has_limit  = p.get("purchase_limit") is not None
    has_media  = float(p.get("media_intensity", 0.0)) > 0
    policy_labels = []
    if pc.get("subsidy_active"):   policy_labels.append(f"Domestic subsidy ({pc.get('subsidy_rate',0)*100:.0f}%)")
    if pc.get("fat_tax_active"):   policy_labels.append(f"Fat tax (>{pc.get('fat_tax_threshold',3.5)}g, {pc.get('fat_tax_rate',0)*100:.0f}%)")
    if pc.get("labelling_active"): policy_labels.append("Nutritional labelling")
    if has_limit:                  policy_labels.append(f"Purchase cap ({p['purchase_limit']} units)")
    if has_media:                  policy_labels.append(f"Gov. comms ({p.get('communication_type','neutral')})")

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Policy Simulation Results")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Peak Food Stress Rate",           f"{peak_stress:.1f}%",
              f"baseline {base_stress:.1f}%",    delta_color="inverse")
    k2.metric("Peak Budget Exhaustion (Low Income)", f"{peak_budgexh_lo:.1f}%",
              "unable to complete basket",        delta_color="inverse")
    k3.metric("Mean Gini Access Index",           f"{mean_gini_c:.3f}",
              f"baseline {mean_gini_b:.3f}",      delta_color="inverse")
    k4.metric("Import Dependency (Crisis)",       f"{import_dep_c:.1f}%",
              f"baseline {import_dep_b:.1f}%",
              delta_color="inverse" if import_dep_c > import_dep_b else "normal")

    # ── Chart 1: Fulfilment by income ─────────────────────────────────────────
    st.markdown("#### 1 · Consumer Basket Fulfilment Rate by Income Group")
    fig1 = go.Figure()
    for col_key, color, label in [
        ("Fulfillment_Low",  "#e74c3c", "Low income (crisis)"),
        ("Fulfillment_Mid",  "#e67e22", "Mid income (crisis)"),
        ("Fulfillment_High", "#27ae60", "High income (crisis)"),
    ]:
        fig1.add_trace(go.Scatter(x=df_c["Day"], y=df_c[col_key]*100,
                                  name=label, line=dict(color=color, width=2.5)))
    fig1.add_trace(go.Scatter(x=df_b["Day"], y=df_b["FulfillmentRate"]*100,
                              name="All income (baseline)",
                              line=dict(color="#95a5a6", width=1.5, dash="dash")))
    fig1.add_hline(y=80, line_dash="dot", line_color="#c0392b",
                   annotation_text="80% welfare threshold",
                   annotation_font_size=9, annotation_position="bottom right")
    fig1 = _sf_crisis_band(fig1, cri_start, cri_end, days)
    fig1.update_layout(
        template="plotly_white", height=400,
        xaxis_title="Simulation Day", yaxis_title="Fulfilment Rate (%)",
        yaxis=dict(range=[0, 105]),
        title="Consumer Basket Fulfilment Rate by Income Group — Crisis Scenario",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig1, use_container_width=True)
    policy_hint = ("A purchase cap is active — check Chart 5 for its equity effect."
                   if has_limit else "Consider a per-visit purchase cap or targeted subsidy to narrow this gap.")
    _sf_analysis_box(
        f"During the crisis, low-income consumers fulfilled **{fulfill_lo_c:.1f}%** of their intended basket "
        f"on average, versus **{fulfill_hi_c:.1f}%** for high-income — a **{fulfill_gap:.1f} pp equity gap**. "
        f"Low-income households fell below the 80% welfare threshold on **{low_below_80} of {days} days**. "
        f"This divergence arises because lower-income agents have less budget buffer to absorb inflation and "
        f"are more likely to encounter stockouts as higher-income panic-buyers deplete shelves first "
        f"([Darmon & Drewnowski, 2008](https://pubmed.ncbi.nlm.nih.gov/18469226/)). {policy_hint}"
    )

    # ── Chart 2: FIES Food Security ───────────────────────────────────────────
    st.markdown("#### 2 · Food Security Index (FIES) by Income Group")
    fig2 = go.Figure()
    for col_key, color, label in [
        ("FIESSevere_Low",  "#922b21", "Low income"),
        ("FIESSevere_Mid",  "#d35400", "Mid income"),
        ("FIESSevere_High", "#27ae60", "High income"),
    ]:
        fig2.add_trace(go.Scatter(x=df_c["Day"], y=df_c[col_key]*100,
                                  name=f"{label} (crisis)",
                                  fill="tozeroy" if col_key == "FIESSevere_Low" else None,
                                  fillcolor="rgba(146,43,33,0.08)",
                                  line=dict(color=color, width=2.5)))
        fig2.add_trace(go.Scatter(x=df_b["Day"], y=df_b[col_key]*100,
                                  name=f"{label} (baseline)",
                                  line=dict(color=color, width=1.0, dash="dot"), opacity=0.5))
    fig2 = _sf_crisis_band(fig2, cri_start, cri_end, days)
    fig2.update_layout(
        template="plotly_white", height=400,
        xaxis_title="Simulation Day", yaxis_title="Severely Food-Insecure (%)",
        title="FIES Severe Food Insecurity by Income Bracket — Crisis vs Baseline (dotted)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig2, use_container_width=True)
    _sf_analysis_box(
        f"Severe food insecurity (FIES, [FAO 2016](https://openknowledge.fao.org/server/api/core/bitstreams/07bc7c6e-72e5-488d-b2f7-3c1499d098fb/content)) among low-income households peaked at "
        f"**{fies_lo_peak:.1f}%** during the crisis, vs a baseline of **{fies_lo_base:.1f}%** "
        f"(+**{fies_delta:.1f} pp**). FIES captures both objective access failure (budget exhaustion, "
        f"stockouts) and subjective stress indicators. Targeted subsidies for low-income consumers have "
        f"the highest FIES-reduction impact per euro spent ([Sen, 1981](https://www.jstor.org/stable/1882681?seq=1); [FAO, 2016](https://openknowledge.fao.org/server/api/core/bitstreams/07bc7c6e-72e5-488d-b2f7-3c1499d098fb/content)). "
        f"Dotted lines = counterfactual baseline for each income group."
    )

    # ── Chart 3: Budget Exhaustion & Gini ────────────────────────────────────
    st.markdown("#### 3 · Budget Exhaustion & Access Inequality (Gini)")
    fig3 = go.Figure()
    for col_key, color, label in [
        ("BudgetExh_Low",  "#e74c3c", "Low income"),
        ("BudgetExh_Mid",  "#e67e22", "Mid income"),
        ("BudgetExh_High", "#27ae60", "High income"),
    ]:
        fig3.add_trace(go.Scatter(x=df_c["Day"], y=df_c[col_key]*100,
                                  name=f"{label} (crisis)",
                                  line=dict(color=color, width=2.5)))
        fig3.add_trace(go.Scatter(x=df_b["Day"], y=df_b[col_key]*100,
                                  name=f"{label} (baseline)",
                                  line=dict(color=color, width=1.0, dash="dash"), opacity=0.5))
    fig3.add_trace(go.Scatter(x=df_c["Day"], y=df_c["GiniAccess"],
                              name="Gini Access Index (crisis)",
                              line=dict(color="#8e44ad", width=2.0, dash="dot"), yaxis="y2"))
    fig3 = _sf_crisis_band(fig3, cri_start, cri_end, days)
    fig3.update_layout(
        template="plotly_white", height=400,
        xaxis_title="Simulation Day", yaxis_title="Budget Exhausted (%)",
        yaxis2=dict(title="Gini Access Index (0–1)", overlaying="y",
                    side="right", range=[0, 1]),
        title="Budget Exhaustion by Income Group and Access Inequality (Gini Index)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig3, use_container_width=True)
    gini_action = "action required" if mean_gini_c > 0.3 else "within acceptable range"
    _sf_analysis_box(
        f"Budget exhaustion among low-income consumers peaked at **{peak_budgexh_lo:.1f}%**, "
        f"vs **{peak_budgexh_hi:.1f}%** for high-income. The Gini access index rose from "
        f"**{mean_gini_b:.3f}** (baseline) to **{mean_gini_c:.3f}** during the crisis — "
        f"{'a statistically meaningful increase in access inequality' if abs(mean_gini_c - mean_gini_b) > 0.02 else 'a marginal change'}. "
        f"A Gini above 0.30 signals structurally unequal access and typically warrants "
        f"rationing or targeted subsidy measures ({gini_action}, [Thaler & Sunstein 2008](https://psycnet.apa.org/record/2008-03730-000))."
    )

    # ── Chart 4: Import Dependency ────────────────────────────────────────────
    st.markdown("#### 4 · Domestic vs Import Sales — Food Sovereignty")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df_c["Day"], y=df_c["DomesticSales"],
                              name="Domestic (crisis)", fill="tozeroy",
                              fillcolor="rgba(39,174,96,0.15)",
                              line=dict(color="#27ae60", width=2.5)))
    fig4.add_trace(go.Scatter(x=df_c["Day"], y=df_c["ImportSales"],
                              name="Import (crisis)", fill="tozeroy",
                              fillcolor="rgba(231,76,60,0.10)",
                              line=dict(color="#e74c3c", width=2.5)))
    fig4.add_trace(go.Scatter(x=df_b["Day"], y=df_b["DomesticSales"],
                              name="Domestic (baseline)",
                              line=dict(color="#27ae60", width=1.0, dash="dash"), opacity=0.5))
    fig4.add_trace(go.Scatter(x=df_b["Day"], y=df_b["ImportSales"],
                              name="Import (baseline)",
                              line=dict(color="#e74c3c", width=1.0, dash="dash"), opacity=0.5))
    fig4 = _sf_crisis_band(fig4, cri_start, cri_end, days)
    fig4.update_layout(
        template="plotly_white", height=380,
        xaxis_title="Simulation Day", yaxis_title="Units Sold",
        title="Domestic vs Import Sales Volume — Food Sovereignty & Supply Chain Resilience",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig4, use_container_width=True)
    dom_dir = "increased" if dom_change >= 0 else "decreased"
    dom_interp = (
        "reflecting consumer preference for Finnish-origin products under uncertainty — "
        "consistent with DCE data showing mean Finnish preference of 0.65 (N=116)."
        if dom_change >= 0 else
        "indicating domestic supply was more disrupted than imports, raising food sovereignty risk."
    )
    _sf_analysis_box(
        f"Domestic sales share {dom_dir} from **{dom_share_b:.1f}%** (baseline) to "
        f"**{dom_share_c:.1f}%** during the crisis — {dom_interp} "
        f"Higher import reliance increases exposure to cross-border disruptions and exchange-rate "
        f"volatility under climate stress ([EC Farm to Fork Strategy, 2030](https://food.ec.europa.eu/system/files/2020-05/f2f_action-plan_2020_strategy-info_en.pdf))."
    )

    # ── Chart 5: Policy Effectiveness (conditional) ───────────────────────────
    if active_policies or has_limit or has_media:
        st.markdown("#### 5 · Active Policy Instruments — Key Welfare Metrics")
        metrics_names = ["Food Stress Peak %", "Budget Exhaustion\nLow Income %",
                         "Gini ×100", "Import Dep. %"]
        metrics_vals  = [peak_stress, peak_budgexh_lo, mean_gini_c * 100, import_dep_c]
        bar_colors    = ["#e74c3c", "#e67e22", "#8e44ad", "#2980b9"]
        fig5 = go.Figure(go.Bar(
            x=metrics_names, y=metrics_vals,
            marker_color=bar_colors,
            text=[f"{v:.1f}" for v in metrics_vals], textposition="outside",
        ))
        fig5.update_layout(
            template="plotly_white", height=340,
            yaxis_title="Value",
            title=f"Key Welfare Metrics — Active policies: {', '.join(policy_labels)}",
        )
        st.plotly_chart(fig5, use_container_width=True)
        _sf_analysis_box(
            f"Active policy instruments: **{', '.join(policy_labels)}**. "
            f"To quantify the isolated effect of each instrument, run the simulator twice — "
            f"once with policies active and once with all policies off — and compare the welfare metrics. "
            f"Most cost-effective interventions for food-insecure households during supply disruptions: "
            f"targeted price subsidies and per-visit purchase caps (Dréze & Sen, 1989; Thaler & Sunstein, 2008)."
        )

    # ── Summary box ───────────────────────────────────────────────────────────
    policies_str = ', '.join(policy_labels) if policy_labels else "none active"
    gini_status  = "⚠️ action required" if mean_gini_c > 0.3 else "✅ within range"
    _sf_summary_box(
        "Policy Impact Summary — SecureFood Climate Scenario",
        [
            f"Peak food stress rate: <b>{peak_stress:.1f}%</b> vs baseline <b>{base_stress:.1f}%</b> (+{peak_stress-base_stress:.1f} pp during crisis)",
            f"Equity gap: low-income fulfilment <b>{fulfill_lo_c:.1f}%</b> vs high-income <b>{fulfill_hi_c:.1f}%</b> — {fulfill_gap:.1f} pp divergence",
            f"FIES severe food insecurity (low income): <b>+{fies_delta:.1f} pp</b> above baseline at peak",
            f"Budget exhaustion (low income): peaked at <b>{peak_budgexh_lo:.1f}%</b> of households",
            f"Access inequality (Gini): <b>{mean_gini_c:.3f}</b> vs baseline <b>{mean_gini_b:.3f}</b> — {gini_status}",
            f"Food sovereignty: domestic sales share {'rose' if dom_change >= 0 else 'fell'} by <b>{abs(dom_change):.1f} pp</b> to <b>{dom_share_c:.1f}%</b>",
            f"Policy instruments active: <b>{policies_str}</b>",
        ],
        (
            f"{'Maintain the purchase cap — it reduces hoarding and access inequality. ' if has_limit else 'Introduce a per-visit purchase cap (2–3 units) — most effective single instrument for access equity. '}"
            f"{'Domestic subsidy is active — monitor fiscal cost against import dependency reduction. ' if pc.get('subsidy_active') else 'A 10–15% domestic product subsidy would support Finnish producers and reduce import dependency. '}"
            f"Coordinate calming public communication via the national food authority to dampen panic. "
            f"If food stress exceeds {peak_stress:.0f}%, activate targeted emergency food vouchers for FIES-severe households. "
            f"These measures align with SecureFood WP3 policy recommendations for Northern European dairy markets."
        ),
    )


# ── Main SecureFood page ───────────────────────────────────────────────────────

def render_securefood_page():
    """Full-screen SecureFood Scenario Simulator — Supply Chain & Policy Maker profiles."""
    st.markdown("""<style>
        section[data-testid="stSidebar"],
        div[data-testid="stSidebarCollapsedControl"],
        div[data-testid="collapsedControl"],
        [data-testid*="Sidebar"] { display: none !important; }
        header[data-testid="stHeader"] { display: none !important; }
        #MainMenu, footer { display: none !important; }
    </style>""", unsafe_allow_html=True)

    c_back, c_title = st.columns([1, 9])
    with c_back:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("← Back", key="sf_back_btn"):
            st.session_state["page"] = "main"
            st.rerun()
    with c_title:
        st.markdown("## 🌿 SecureFood Scenario Simulator")
        st.caption(
            "**Climate Disruption & Finnish Dairy Supply Chain** · "
            "Horizon Europe SecureFood Consortium (Grant No. 101136583) · IAMO XR Lab"
        )

    if st.session_state.config_data is None:
        st.warning(
            "⚠️ Population data not loaded. Return to the main app, open the "
            "**🏠 Data & Population** tab, and load the dataset first."
        )
        return

    st.divider()

    sc_tab, pm_tab = st.tabs(["🏭 Supply Chain Actor", "🏛️ Policy Maker"])

    # ══════════════════════════════════════════════════════════════════════════
    # SUPPLY CHAIN ACTOR
    # ══════════════════════════════════════════════════════════════════════════
    with sc_tab:
        st.markdown(
            "_For **producers, distributors, and retailers** managing Finnish dairy supply chains "
            "under climate-driven disruption. Focus: operational resilience, revenue, and inventory._"
        )
        st.markdown("### ⚙️ Scenario Parameters")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**📅 General**")
            sc_days = st.slider("Duration (Days)", 30, 365, 90, 5, key="sf_sc_days",
                help="Simulation length in days. 90 days captures the full shock-and-recovery arc of a typical supply chain disruption (Sheffi, 2005). Extend to 180+ to model seasonal recovery.")
            sc_consumers = st.number_input("Base Daily Consumers", 50, 2000, 200, 50,
                key="sf_sc_consumers",
                help="Average shoppers per day. Finnish supermarkets: 800–1,200/day (PTY, 2023). Use 50–200 for neighbourhood stores, 500+ for hypermarkets. Scales shelf capacity automatically.")
            sc_month = st.selectbox("Start Month", list(range(1, 13)), index=0,
                key="sf_sc_month",
                help="Simulation start month. Demand varies seasonally: December +30%, June–July +10%. January is standard for annual scenario comparisons.")

        with c2:
            st.markdown("**🚚 Logistics**")
            sc_lead = st.slider("Lead Time (Days)", 1, 14, 3, 1, key="sf_sc_lead",
                help="Days from order placement to delivery. Finnish domestic dairy: 2–3 days. Imported: 5–7 days (Ruokatieto, 2024). Longer lead times increase stockout risk under supply shocks.")
            sc_reorder = st.slider("Reorder Point (% of storage)", 10, 60, 30, 5,
                key="sf_sc_reorder",
                help="Restocking triggered when storage drops below this threshold. Industry standard: 25–35% (Silver et al., 2017). Raise to build safety stock ahead of anticipated disruptions.") / 100.0
            sc_target = st.slider("Restock Target (% of storage)", 70, 100, 90, 5,
                key="sf_sc_target",
                help="Target storage fill after replenishment. 90% provides a safety buffer; higher values raise perishable waste risk for short-shelf-life dairy products (Chopra & Meindl, 2016).") / 100.0

        with c3:
            st.markdown("**🔴 Climate Crisis**")
            sc_cri_start = st.slider("Crisis Start Day", 5, max(6, sc_days - 10),
                                     min(30, sc_days - 10), key="sf_sc_cri_start",
                help="Day the climate disruption begins. Allow at least 20 pre-crisis days to establish a stable baseline for comparison.")
            sc_disruption = st.slider("Supply Disruption (Days delay)", 0, 30, 7, 1,
                key="sf_sc_disruption",
                help="Delivery delay per event. Finnish dairy logistics disruptions from extreme weather: 3–10 days (Luke, 2023). 0 = price effect only, no delivery delay.")
            sc_inflation = st.slider("Price Inflation (%)", 0, 100, 25, 5,
                key="sf_sc_inflation",
                help="Retail price increase driven by production cost rises under climate stress. +25% aligns with IPCC AR6 projections for Northern European food systems by 2040 (IPCC, 2022).")
            sc_cri_dur = st.slider("Crisis Duration (Days)", 0,
                                   max(1, sc_days - sc_cri_start), 30, 5,
                key="sf_sc_cri_dur",
                help="How long the disruption persists before recovery. 0 = runs to end of simulation. 30–60 days models a temporary weather event; 0 models structural climate change impact.")

        with st.expander("🧠 Consumer Behaviour (advanced)", expanded=False):
            cb1, cb2 = st.columns(2)
            sc_panic = cb1.slider("Panic Sensitivity", 0.0, 1.0, 0.50, 0.05,
                key="sf_sc_panic",
                help="Consumer propensity to panic-buy when scarcity is perceived. Calibrated from Finnish DCE data (N=116). Higher values = faster inventory depletion and stronger bullwhip effect.")
            sc_hoard = cb2.slider("Hoarding Factor", 1.0, 3.0, 1.5, 0.1,
                key="sf_sc_hoard",
                help="Purchase quantity multiplier during panic. 1.5 = consumers buy 50% more than usual. Empirically validated range for Finnish panel data (Hendel & Nevo, 2006).")

        col_run, _ = st.columns([2, 6])
        if col_run.button("▶ Run Supply Chain Simulation", type="primary",
                          key="sf_sc_run", use_container_width=True):
            _no_policy = {
                "fat_tax_active": False, "fat_tax_threshold": 3.5, "fat_tax_rate": 0.0,
                "subsidy_active": False, "subsidy_target": "domestic", "subsidy_rate": 0.0,
                "domestic_shock_active": False, "domestic_shock_day": sc_cri_start,
                "domestic_shock_duration": 30, "domestic_shock_severity": 0.5,
                "labelling_active": False, "labelling_day": 1,
                "labelling_health_boost": 0.0, "labelling_organic_boost": 0.0,
            }
            sc_params = {
                "days": sc_days, "month": sc_month, "base_con": int(sc_consumers),
                "reorder": sc_reorder, "target": sc_target, "lead": sc_lead,
                "cri_start": sc_cri_start, "cri_duration": int(sc_cri_dur),
                "inf": float(sc_inflation), "dis": int(sc_disruption),
                "panic": sc_panic, "hoard": sc_hoard, "mc_runs": 1,
                "policy_cfg": _no_policy,
                "purchase_limit": None, "media_intensity": 0.0,
                "communication_type": "neutral", "stockpile_days": None,
            }
            with st.spinner("Running baseline and crisis simulations…"):
                result = _sf_run_simulation(sc_params)
                if result:
                    st.session_state["sf_results_sc"] = result

        if st.session_state.get("sf_results_sc"):
            _render_sf_sc_results(st.session_state["sf_results_sc"])

    # ══════════════════════════════════════════════════════════════════════════
    # POLICY MAKER
    # ══════════════════════════════════════════════════════════════════════════
    with pm_tab:
        st.markdown(
            "_For **government agencies, regulators, and food system authorities**. "
            "Focus: consumer welfare, equity, food security, and policy intervention effectiveness._"
        )
        st.markdown("### ⚙️ Scenario Parameters")
        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown("**🔴 Crisis Severity**")
            pm_days = st.slider("Duration (Days)", 60, 365, 120, 10, key="sf_pm_days",
                help="Longer horizons capture recovery and long-run policy effects. 120 days recommended for policy assessment (crisis + early recovery). Extend to 365 for structural impact.")
            pm_consumers = st.number_input("Base Daily Consumers", 50, 2000, 200, 50,
                key="sf_pm_consumers",
                help="Store traffic level. Policy simulations are robust across store sizes, but larger stores show more pronounced income-stratified effects due to greater product diversity.")
            pm_cri_start = st.slider("Crisis Start Day", 5, max(6, pm_days - 20),
                                     min(30, pm_days - 20), key="sf_pm_cri_start",
                help="Allow ≥20 days of baseline before crisis onset to establish welfare reference levels for comparison.")
            pm_disruption = st.slider("Supply Disruption (Days delay)", 0, 30, 7, 1,
                key="sf_pm_disruption",
                help="Supply-side shock severity. 7 days = significant but recoverable. 14+ days = severe climate event or geopolitical supply cut. Tests the resilience of food assistance programmes.")
            pm_inflation = st.slider("Price Inflation (%)", 0, 100, 25, 5,
                key="sf_pm_inflation",
                help="+25% is the IPCC AR6 central estimate for Northern European food under 2°C warming. Higher values test the effectiveness of price-stabilisation and subsidy policies.")
            pm_cri_dur = st.slider("Crisis Duration (Days)", 0,
                                   max(1, pm_days - pm_cri_start), 45, 5,
                key="sf_pm_cri_dur",
                help="0 = permanent structural change; 30–60 days = temporary shock. Policy effects are best evaluated over the full crisis + recovery arc to capture persistence.")

        with p2:
            st.markdown("**🧠 Consumer Behaviour**")
            pm_panic = st.slider("Panic Sensitivity", 0.0, 1.0, 0.50, 0.05,
                key="sf_pm_panic",
                help="Baseline panic propensity. Calibrated from Finnish DCE data (N=116, mean 0.55). Increase to model lower trust in food system resilience or high pre-existing food anxiety.")
            pm_hoard = st.slider("Hoarding Factor", 1.0, 3.0, 1.5, 0.1,
                key="sf_pm_hoard",
                help="Average purchase multiplier during panic. 1.5 = Finnish panel baseline. 2.0–2.5 models severe panic buying such as observed during COVID-19 supply events (Grashuis et al., 2020).")
            pm_month = st.selectbox("Start Month", list(range(1, 13)), index=0,
                key="sf_pm_month",
                help="December scenarios produce higher baseline demand, making supply shortfalls more severe and equity effects more pronounced.")
            pm_lead = st.slider("Lead Time (Days)", 1, 14, 3, 1, key="sf_pm_lead",
                help="Policy note: longer lead times amplify the equity impact — low-income consumers are hit first as safety stock depletes, because they have less ability to pre-stock at home.")

        with p3:
            st.markdown("**🏛️ Policy Instruments**")
            pm_pl_on = st.checkbox("Enable Purchase Rationing", False, key="sf_pm_pl_on",
                help="Per-visit purchase cap to reduce panic hoarding. Most effective short-term access equity tool (Thaler & Sunstein, 2008). Trade-off: enforcement cost and consumer resistance.")
            pm_pl_val = st.slider("Max Units per Product per Visit", 1, 10, 3,
                key="sf_pm_pl_val",
                help="A cap of 2–3 units reduces Gini access inequality by 15–25% in crisis simulations without significantly reducing total sales (Gruen et al., 2002). Run without cap to quantify effect.")
            pm_purchase_limit = pm_pl_val if pm_pl_on else None

            pm_sub_on = st.checkbox("Domestic Product Subsidy", False, key="sf_pm_sub_on",
                help="Price subsidy on Finnish-origin dairy. Supports domestic producers, reduces import dependency, and partly mitigates price inflation for lower-income consumers.")
            pm_sub_rate = st.slider("Subsidy Rate (%)", 5, 40, 15, 5, key="sf_pm_sub_rate",
                help="15% is the median effective rate in EU food sovereignty programmes. Higher rates increase fiscal cost — balance against FIES-reduction benefit.") / 100.0 \
                if pm_sub_on else 0.0

            pm_lab_on = st.checkbox("Nutritional Labelling Policy", False, key="sf_pm_lab_on",
                help="Mandatory health-oriented labelling shifts preferences toward healthier choices over time. Effect grows slowly — most visible in simulations >60 days (Sunstein, 2014).")

            pm_comm = st.selectbox("Government Communication Strategy",
                ["neutral", "calming", "panic"], key="sf_pm_comm",
                help="calming = coordinated reassuring messaging reduces daily panic; neutral = factual reporting (no panic effect); panic = sensationalist coverage (models information failure). McCombs & Shaw (1972).")
            pm_media = st.slider("Communication Intensity", 0.0, 1.0, 0.3, 0.05,
                key="sf_pm_media",
                help="Strength of daily communication effect on consumer panic. 0.3 = moderate coordinated government campaign. Set to 0 to isolate the crisis without communication intervention.") \
                if pm_comm != "neutral" else 0.0

        col_run2, _ = st.columns([2, 6])
        if col_run2.button("▶ Run Policy Simulation", type="primary",
                           key="sf_pm_run", use_container_width=True):
            pm_policy_cfg = {
                "fat_tax_active": False, "fat_tax_threshold": 3.5, "fat_tax_rate": 0.0,
                "subsidy_active": pm_sub_on, "subsidy_target": "domestic",
                "subsidy_rate": pm_sub_rate,
                "domestic_shock_active": False, "domestic_shock_day": pm_cri_start,
                "domestic_shock_duration": 30, "domestic_shock_severity": 0.5,
                "labelling_active": pm_lab_on, "labelling_day": pm_cri_start,
                "labelling_health_boost": 0.10, "labelling_organic_boost": 0.05,
            }
            pm_params = {
                "days": pm_days, "month": pm_month, "base_con": int(pm_consumers),
                "reorder": 0.30, "target": 0.90, "lead": pm_lead,
                "cri_start": pm_cri_start, "cri_duration": int(pm_cri_dur),
                "inf": float(pm_inflation), "dis": int(pm_disruption),
                "panic": pm_panic, "hoard": pm_hoard, "mc_runs": 1,
                "policy_cfg": pm_policy_cfg,
                "purchase_limit": pm_purchase_limit,
                "media_intensity": pm_media,
                "communication_type": pm_comm,
                "stockpile_days": None,
            }
            with st.spinner("Running policy simulation…"):
                result = _sf_run_simulation(pm_params)
                if result:
                    st.session_state["sf_results_pm"] = result

        if st.session_state.get("sf_results_pm"):
            _render_sf_pm_results(st.session_state["sf_results_pm"])


# ===========================================================================
# 1. SESSION STATE INITIALISATION
# ===========================================================================

defaults = {
    "config_data":     None,
    "page":            "landing",
    "lang":            "en",
    # SecureFood Scenario results
    "sf_results_sc":   None,
    "sf_results_pm":   None,
    # Simulation results
    "sim_results":     None,
    "sim_stock":       None,
    "sim_scm_log":     None,
    "sim_waste":       None,
    "sim_product_recs": None,
    "sim_model_crisis": None,
    "sim_pref_drift":   None,
    # Scientific workflow state
    "mc_stage":        0,
    "data_base_raw":   None,
    "data_base_opt":   None,
    "data_crisis":     None,
    "ai_recs":         None,
    "active_baseline": "Baseline (Raw)",
    "prod_stats_raw":  None,
    # Policy analysis results
    "policy_baseline":  None,   # DataFrame: daily records, no policy
    "policy_scenario":  None,   # DataFrame: daily records, with policy active
    "policy_label":     None,   # human-readable name of the active policy run
    # Multi-scenario store: list of {"label": str, "df": DataFrame}
    "policy_scenarios": [],
    # Onboarding tour (1 = first step, 0 = hidden)
    "tour_step": 1,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===========================================================================
# TRANSLATIONS — main application UI strings
# ===========================================================================
_MAIN_T = {
    "en": {
        "subtitle": "**Agent-Based Model for Consumer Behaviour & Supply Chain Stress-Testing** | SecureFood / Horizon Europe — IAMO XR Lab",
        "tabs": ["🏠 Data & Population", "🎮 Interactive Demo", "🔬 Scientific Analysis", "♻️ Food Waste", "📦 Per-Product", "🏛️ Policy Analysis", "👔 Stakeholder View", "🎚️ Sensitivity Analysis", "🧪 Behavioural Theory", "📥 Export"],
        "sidebar_title": "⚙️ Simulation Parameters",
        "sidebar_general": "📅 General",
        "duration_days": "Duration (Days)",
        "start_month": "Start Month",
        "base_consumers": "Base Daily Consumers",
        "sidebar_logistics": "🚚 Logistics",
        "reorder_pt": "Reorder Point (% of max storage)",
        "restock_target": "Restock Target (% of max storage)",
        "lead_time": "Lead Time (Days)",
        "sidebar_crisis": "🔴 Crisis Scenario",
        "crisis_start": "Crisis Start Day",
        "price_inflation": "Price Inflation (%)",
        "supply_disruption": "Supply Disruption (Days)",
        "crisis_duration": "Crisis Duration (Days)",
        "sidebar_behaviour": "🧠 Consumer Behaviour",
        "panic_sensitivity": "Panic Sensitivity",
        "hoarding_factor": "Hoarding Factor",
        "sidebar_interventions": "🛡️ Behavioural Interventions",
        "sidebar_mc": "🔬 Monte Carlo",
        "mc_runs": "Monte Carlo Runs (N)",
        "sidebar_policy": "🏛️ Policy Scenarios",
        "header_data": "🏠 Data & Population",
        "header_demo": "🎮 Interactive Demo",
        "header_science": "🔬 Scientific Analysis (Monte Carlo Workflow)",
        "header_waste": "♻️ Food Waste Dashboard",
        "header_product": "📦 Per-Product Deep Dive",
        "header_behaviour": "🧪 Behavioural Theory Dashboard",
        "header_export": "📥 Export Simulation Data",
        "header_policy": "🏛️ Policy Analysis",
        "header_stakeholder": "👔 Stakeholder View",
        "header_sensitivity": "🎚️ Sensitivity Analysis",
        "btn_run_demo": "▶️ Run Comparative Simulation",
        "btn_run_baseline": "🚀 Run Baseline Analysis",
        "btn_run_crisis": "🔥 Run Crisis Simulation",
        "btn_run_policy": "▶️ Run & Compare",
        "btn_run_sensitivity": "▶️ Run Sensitivity Analysis",
        "demo_subtabs": ["📈 Financials", "👥 Footfall", "🚚 Logistics", "💥 Crisis Breakdown", "🧠 Behavioural Drift", "👤 By Buyer Type"],
        "sub_demographics": "👥 Population Demographics",
        "sub_archetypes": "🏷️ Behavioural Archetypes (Real Participants)",
        "sub_dce": "🧠 DCE-Derived Preference Distributions",
        "sub_catalogue": "🛒 Product Catalogue",
        "animation_speed": "Animation Speed (delay per day, seconds)",
        "securefood_btn": "📄 Scenario Instructions",
        "sidebar_interventions_caption": "These levers are visible in the 🧪 Behavioural Theory tab.",
        "exp_nudge": "🛒 Nudge — Purchase Limit",
        "nudge_cap_on": "Enable per-visit purchase cap",
        "nudge_cap_val": "Max units per product per visit",
        "exp_media": "📡 Media / Communication",
        "media_intensity": "Media intensity (0 = off)",
        "comm_type": "Communication type",
        "exp_stockpile": "🏠 Stockpile Horizon",
        "stockpile_on": "Override stockpile horizon",
        "stockpile_days": "Stockpile days (planning horizon)",
        "sidebar_policy_caption": "Configure policy levers for the Policy Analysis tab.",
        "exp_fat_tax": "🧀 Fat Tax",
        "fat_tax_on": "Enable Fat Tax",
        "fat_tax_threshold": "Fat% threshold",
        "fat_tax_rate_lbl": "Surcharge rate (%)",
        "exp_subsidy": "🌿 Domestic / Organic Subsidy",
        "subsidy_on": "Enable Subsidy",
        "subsidy_target": "Subsidy target",
        "subsidy_rate": "Discount rate (%)",
        "exp_shock": "🐄 Domestic Supply Shock",
        "shock_on": "Enable Supply Shock",
        "shock_start": "Shock Start Day",
        "shock_duration": "Shock Duration (Days)",
        "shock_severity": "Shock Severity (fraction blocked)",
        "exp_labelling": "🏷️ Nutritional Labelling",
        "labelling_on": "Enable Labelling",
        "labelling_start": "Labelling Start Day",
        "labelling_health": "Health preference boost",
        "labelling_organic": "Organic preference boost",
        "chart_age": "Age Distribution",
        "label_age": "Age",
        "label_count": "Count",
        "chart_gender": "Gender Distribution",
        "chart_income": "Income Groups",
        "participants": "participants",
        "chart_radar_title": "Mean Questionnaire Factor Scores by Archetype",
        "factor_labels": ["Price", "Health", "Environment", "Animal Welfare", "Sensory/Habit"],
        "pref_finnish": "Finnish Preference",
        "pref_organic": "Organic Preference",
        "pref_price": "Price Sensitivity",
        "pref_fat": "Preferred Fat %",
        "arch_names": {
            "price_champion":  "💸 Price Champion",
            "green_buyer":     "🌿 Green Buyer",
            "health_optimizer":"💪 Health Optimizer",
            "habitual_buyer":  "🔁 Habitual Buyer",
            "price_conscious": "💰 Price Conscious",
        },
    },
    "fi": {
        "subtitle": "**Agenttipohjainen malli kuluttajakäyttäytymiseen & toimitusketjun stressitestaukseen** | SecureFood / Horizon Europe — IAMO XR Lab",
        "tabs": ["🏠 Data & väestö", "🎮 Interaktiivinen demo", "🔬 Tieteellinen analyysi", "♻️ Ruokahävikki", "📦 Tuotekohtainen", "🏛️ Politiikka-analyysi", "👔 Sidosryhmänäkymä", "🎚️ Herkkyysanalyysi", "🧪 Käyttäytymisteoria", "📥 Vienti"],
        "sidebar_title": "⚙️ Simulaatioparametrit",
        "sidebar_general": "📅 Yleiset",
        "duration_days": "Kesto (päivät)",
        "start_month": "Aloituskuukausi",
        "base_consumers": "Päivittäiset peruskuluttajat",
        "sidebar_logistics": "🚚 Logistiikka",
        "reorder_pt": "Tilausraja (% max varastosta)",
        "restock_target": "Täydennystavoite (% max varastosta)",
        "lead_time": "Toimitusaika (päivät)",
        "sidebar_crisis": "🔴 Kriisiskenaario",
        "crisis_start": "Kriisin alkupäivä",
        "price_inflation": "Hintainflaatio (%)",
        "supply_disruption": "Toimitushäiriö (päivät)",
        "crisis_duration": "Kriisin kesto (päivät)",
        "sidebar_behaviour": "🧠 Kuluttajakäyttäytyminen",
        "panic_sensitivity": "Paniikkiherkkyys",
        "hoarding_factor": "Hamstrauskerroin",
        "sidebar_interventions": "🛡️ Käyttäytymisinterventiot",
        "sidebar_mc": "🔬 Monte Carlo",
        "mc_runs": "Monte Carlo -ajot (N)",
        "sidebar_policy": "🏛️ Politiikkaskenaariot",
        "header_data": "🏠 Data ja väestö",
        "header_demo": "🎮 Interaktiivinen demo",
        "header_science": "🔬 Tieteellinen analyysi (Monte Carlo -työnkulku)",
        "header_waste": "♻️ Ruokahävikin kojelauta",
        "header_product": "📦 Tuotekohtainen syväanalyysi",
        "header_behaviour": "🧪 Käyttäytymisteorian kojelauta",
        "header_export": "📥 Vie simulaatiodata",
        "header_policy": "🏛️ Politiikka-analyysi",
        "header_stakeholder": "👔 Sidosryhmänäkymä",
        "header_sensitivity": "🎚️ Herkkyysanalyysi",
        "btn_run_demo": "▶️ Suorita vertailusimulaatio",
        "btn_run_baseline": "🚀 Suorita perusanalyysi",
        "btn_run_crisis": "🔥 Suorita kriisisimulaatio",
        "btn_run_policy": "▶️ Suorita ja vertaa",
        "btn_run_sensitivity": "▶️ Suorita herkkyysanalyysi",
        "demo_subtabs": ["📈 Talous", "👥 Kävijämäärä", "🚚 Logistiikka", "💥 Kriisianalyysi", "🧠 Käyttäytymismuutos", "👤 Ostajatyypin mukaan"],
        "sub_demographics": "👥 Väestörakenne",
        "sub_archetypes": "🏷️ Käyttäytymisarkkityypit (Oikeat osallistujat)",
        "sub_dce": "🧠 DCE-johdetut preferenssijakaumat",
        "sub_catalogue": "🛒 Tuoteluettelo",
        "animation_speed": "Animaationopeus (viive päivää kohti, sekuntia)",
        "securefood_btn": "📄 Skenaarion ohjeet",
        "sidebar_interventions_caption": "Nämä vipuvarret näkyvät 🧪 Käyttäytymisteoria-välilehdellä.",
        "exp_nudge": "🛒 Kehotus — Ostorajoitus",
        "nudge_cap_on": "Ota käyttöön käyntikohtainen ostorajoitus",
        "nudge_cap_val": "Enimmäismäärä per tuote per käynti",
        "exp_media": "📡 Media / Viestintä",
        "media_intensity": "Mediaintensiteetti (0 = pois)",
        "comm_type": "Viestintätyyppi",
        "exp_stockpile": "🏠 Varastointihorisontti",
        "stockpile_on": "Ohita varastointihorisontti",
        "stockpile_days": "Varastopäivät (suunnitteluhorisontti)",
        "sidebar_policy_caption": "Aseta politiikkavipuvarret Politiikka-analyysi-välilehdelle.",
        "exp_fat_tax": "🧀 Rasvaavero",
        "fat_tax_on": "Ota rasvaavero käyttöön",
        "fat_tax_threshold": "Rasva%-raja",
        "fat_tax_rate_lbl": "Lisämaksuprosentti (%)",
        "exp_subsidy": "🌿 Kotimainen / Luomutuki",
        "subsidy_on": "Ota tuki käyttöön",
        "subsidy_target": "Tuen kohde",
        "subsidy_rate": "Alennusprosentti (%)",
        "exp_shock": "🐄 Kotimainen tarjontashokki",
        "shock_on": "Ota tarjontashokki käyttöön",
        "shock_start": "Shokin alkupäivä",
        "shock_duration": "Shokin kesto (päivät)",
        "shock_severity": "Shokin vakavuus (osuus estetty)",
        "exp_labelling": "🏷️ Ravintoarvojen merkintä",
        "labelling_on": "Ota merkintä käyttöön",
        "labelling_start": "Merkinnän alkupäivä",
        "labelling_health": "Terveysmieltymyksen lisäys",
        "labelling_organic": "Luomomieltymyksen lisäys",
        "chart_age": "Ikäjakauma",
        "label_age": "Ikä",
        "label_count": "Määrä",
        "chart_gender": "Sukupuolijakauma",
        "chart_income": "Tuloryhmät",
        "participants": "osallistujaa",
        "chart_radar_title": "Keskimääräiset kyselytekijäpisteet arkkityypin mukaan",
        "factor_labels": ["Hinta", "Terveys", "Ympäristö", "Eläinten hyvinvointi", "Aistimus/Tapa"],
        "pref_finnish": "Suomalaisuusmieltymys",
        "pref_organic": "Luomomieltymys",
        "pref_price": "Hintaherkkyys",
        "pref_fat": "Suosittu rasva %",
        "arch_names": {
            "price_champion":  "💸 Hintataistelija",
            "green_buyer":     "🌿 Vihreä ostaja",
            "health_optimizer":"💪 Terveysoptimoija",
            "habitual_buyer":  "🔁 Tottumuksen ostaja",
            "price_conscious": "💰 Hintatietoinen",
        },
    },
    "el": {
        "subtitle": "**Μοντέλο Πράκτορα για Καταναλωτική Συμπεριφορά & Ανθεκτικότητα Αλυσίδας Εφοδιασμού** | SecureFood / Horizon Europe — IAMO XR Lab",
        "tabs": ["🏠 Δεδομένα & Πληθυσμός", "🎮 Διαδραστικό Demo", "🔬 Επιστημονική Ανάλυση", "♻️ Απώλεια Τροφίμων", "📦 Ανά Προϊόν", "🏛️ Ανάλυση Πολιτικής", "👔 Προβολή Ενδιαφερομένων", "🎚️ Ανάλυση Ευαισθησίας", "🧪 Θεωρία Συμπεριφοράς", "📥 Εξαγωγή"],
        "sidebar_title": "⚙️ Παράμετροι Προσομοίωσης",
        "sidebar_general": "📅 Γενικά",
        "duration_days": "Διάρκεια (ημέρες)",
        "start_month": "Μήνας Έναρξης",
        "base_consumers": "Βασικοί Ημερήσιοι Καταναλωτές",
        "sidebar_logistics": "🚚 Εφοδιαστική",
        "reorder_pt": "Σημείο Επαναπαραγγελίας (% μέγ. αποθήκευσης)",
        "restock_target": "Στόχος Ανεφοδιασμού (% μέγ. αποθήκευσης)",
        "lead_time": "Χρόνος Παράδοσης (ημέρες)",
        "sidebar_crisis": "🔴 Σενάριο Κρίσης",
        "crisis_start": "Ημέρα Έναρξης Κρίσης",
        "price_inflation": "Πληθωρισμός Τιμών (%)",
        "supply_disruption": "Διακοπή Εφοδιασμού (ημέρες)",
        "crisis_duration": "Διάρκεια Κρίσης (ημέρες)",
        "sidebar_behaviour": "🧠 Καταναλωτική Συμπεριφορά",
        "panic_sensitivity": "Ευαισθησία Πανικού",
        "hoarding_factor": "Συντελεστής Αποθεματισμού",
        "sidebar_interventions": "🛡️ Συμπεριφορικές Παρεμβάσεις",
        "sidebar_mc": "🔬 Monte Carlo",
        "mc_runs": "Εκτελέσεις Monte Carlo (N)",
        "sidebar_policy": "🏛️ Σενάρια Πολιτικής",
        "header_data": "🏠 Δεδομένα & Πληθυσμός",
        "header_demo": "🎮 Διαδραστικό Demo",
        "header_science": "🔬 Επιστημονική Ανάλυση (Ροή Monte Carlo)",
        "header_waste": "♻️ Πίνακας Απώλειας Τροφίμων",
        "header_product": "📦 Βαθιά Ανάλυση ανά Προϊόν",
        "header_behaviour": "🧪 Πίνακας Θεωρίας Συμπεριφοράς",
        "header_export": "📥 Εξαγωγή Δεδομένων Προσομοίωσης",
        "header_policy": "🏛️ Ανάλυση Πολιτικής",
        "header_stakeholder": "👔 Προβολή Ενδιαφερομένων",
        "header_sensitivity": "🎚️ Ανάλυση Ευαισθησίας",
        "btn_run_demo": "▶️ Εκτέλεση Συγκριτικής Προσομοίωσης",
        "btn_run_baseline": "🚀 Εκτέλεση Βασικής Ανάλυσης",
        "btn_run_crisis": "🔥 Εκτέλεση Προσομοίωσης Κρίσης",
        "btn_run_policy": "▶️ Εκτέλεση & Σύγκριση",
        "btn_run_sensitivity": "▶️ Εκτέλεση Ανάλυσης Ευαισθησίας",
        "demo_subtabs": ["📈 Οικονομικά", "👥 Επισκεψιμότητα", "🚚 Εφοδιαστική", "💥 Ανάλυση Κρίσης", "🧠 Συμπεριφορική Μεταβολή", "👤 Ανά Τύπο Αγοραστή"],
        "sub_demographics": "👥 Δημογραφικά Στοιχεία",
        "sub_archetypes": "🏷️ Αρχέτυπα Συμπεριφοράς (Πραγματικοί Συμμετέχοντες)",
        "sub_dce": "🧠 Κατανομές Προτιμήσεων DCE",
        "sub_catalogue": "🛒 Κατάλογος Προϊόντων",
        "animation_speed": "Ταχύτητα Κινούμενης Εικόνας (καθυστέρηση ανά ημέρα, δευτ.)",
        "securefood_btn": "📄 Οδηγίες Σεναρίου",
        "sidebar_interventions_caption": "Αυτές οι ρυθμίσεις εμφανίζονται στην καρτέλα 🧪 Θεωρία Συμπεριφοράς.",
        "exp_nudge": "🛒 Ώθηση — Όριο Αγοράς",
        "nudge_cap_on": "Ενεργοποίηση ορίου αγοράς ανά επίσκεψη",
        "nudge_cap_val": "Μέγ. τεμάχια ανά προϊόν ανά επίσκεψη",
        "exp_media": "📡 Μέσα / Επικοινωνία",
        "media_intensity": "Ένταση μέσων (0 = απενεργ.)",
        "comm_type": "Τύπος επικοινωνίας",
        "exp_stockpile": "🏠 Ορίζοντας Αποθεμάτων",
        "stockpile_on": "Παράκαμψη ορίζοντα αποθεμάτων",
        "stockpile_days": "Ημέρες αποθεμάτων (ορίζοντας)",
        "sidebar_policy_caption": "Ρυθμίστε τις πολιτικές για την καρτέλα Ανάλυσης Πολιτικής.",
        "exp_fat_tax": "🧀 Φόρος Λίπους",
        "fat_tax_on": "Ενεργοποίηση φόρου λίπους",
        "fat_tax_threshold": "Όριο λίπους %",
        "fat_tax_rate_lbl": "Ποσοστό προσαύξησης (%)",
        "exp_subsidy": "🌿 Εγχώρια / Βιολογική Επιδότηση",
        "subsidy_on": "Ενεργοποίηση επιδότησης",
        "subsidy_target": "Στόχος επιδότησης",
        "subsidy_rate": "Ποσοστό έκπτωσης (%)",
        "exp_shock": "🐄 Εγχώριο Σοκ Εφοδιασμού",
        "shock_on": "Ενεργοποίηση σοκ εφοδιασμού",
        "shock_start": "Ημέρα έναρξης σοκ",
        "shock_duration": "Διάρκεια σοκ (ημέρες)",
        "shock_severity": "Σοβαρότητα σοκ (κλάσμα αποκλεισμού)",
        "exp_labelling": "🏷️ Διατροφική Επισήμανση",
        "labelling_on": "Ενεργοποίηση επισήμανσης",
        "labelling_start": "Ημέρα έναρξης επισήμανσης",
        "labelling_health": "Ενίσχυση υγιεινής προτίμησης",
        "labelling_organic": "Ενίσχυση βιολογικής προτίμησης",
        "chart_age": "Κατανομή Ηλικίας",
        "label_age": "Ηλικία",
        "label_count": "Πλήθος",
        "chart_gender": "Κατανομή Φύλου",
        "chart_income": "Ομάδες Εισοδήματος",
        "participants": "συμμετέχοντες",
        "chart_radar_title": "Μέσες Βαθμολογίες Παραγόντων Ερωτηματολογίου ανά Αρχέτυπο",
        "factor_labels": ["Τιμή", "Υγεία", "Περιβάλλον", "Ευζωία Ζώων", "Αισθητικό/Συνήθεια"],
        "pref_finnish": "Προτίμηση Φινλανδικών",
        "pref_organic": "Βιολογική Προτίμηση",
        "pref_price": "Ευαισθησία στην Τιμή",
        "pref_fat": "Προτιμώμενο Λίπος %",
        "arch_names": {
            "price_champion":  "💸 Κυνηγός Τιμών",
            "green_buyer":     "🌿 Πράσινος Αγοραστής",
            "health_optimizer":"💪 Υγειομανής",
            "habitual_buyer":  "🔁 Συνήθης Αγοραστής",
            "price_conscious": "💰 Τιμοσυνείδητος",
        },
    },
    "pt": {
        "subtitle": "**Modelo Baseado em Agentes para Comportamento do Consumidor & Cadeia de Abastecimento** | SecureFood / Horizon Europe — IAMO XR Lab",
        "tabs": ["🏠 Dados & População", "🎮 Demo Interativo", "🔬 Análise Científica", "♻️ Desperdício Alimentar", "📦 Por Produto", "🏛️ Análise de Políticas", "👔 Visão das Partes", "🎚️ Análise de Sensibilidade", "🧪 Teoria Comportamental", "📥 Exportar"],
        "sidebar_title": "⚙️ Parâmetros da Simulação",
        "sidebar_general": "📅 Geral",
        "duration_days": "Duração (dias)",
        "start_month": "Mês de Início",
        "base_consumers": "Consumidores Diários Base",
        "sidebar_logistics": "🚚 Logística",
        "reorder_pt": "Ponto de Reabastecimento (% do máx.)",
        "restock_target": "Meta de Reabastecimento (% do máx.)",
        "lead_time": "Prazo de Entrega (dias)",
        "sidebar_crisis": "🔴 Cenário de Crise",
        "crisis_start": "Dia de Início da Crise",
        "price_inflation": "Inflação de Preços (%)",
        "supply_disruption": "Interrupção de Fornecimento (dias)",
        "crisis_duration": "Duração da Crise (dias)",
        "sidebar_behaviour": "🧠 Comportamento do Consumidor",
        "panic_sensitivity": "Sensibilidade ao Pânico",
        "hoarding_factor": "Fator de Acumulação",
        "sidebar_interventions": "🛡️ Intervenções Comportamentais",
        "sidebar_mc": "🔬 Monte Carlo",
        "mc_runs": "Execuções Monte Carlo (N)",
        "sidebar_policy": "🏛️ Cenários de Política",
        "header_data": "🏠 Dados & População",
        "header_demo": "🎮 Demo Interativo",
        "header_science": "🔬 Análise Científica (Fluxo Monte Carlo)",
        "header_waste": "♻️ Painel de Desperdício Alimentar",
        "header_product": "📦 Análise Detalhada por Produto",
        "header_behaviour": "🧪 Painel de Teoria Comportamental",
        "header_export": "📥 Exportar Dados da Simulação",
        "header_policy": "🏛️ Análise de Políticas",
        "header_stakeholder": "👔 Visão das Partes Interessadas",
        "header_sensitivity": "🎚️ Análise de Sensibilidade",
        "btn_run_demo": "▶️ Executar Simulação Comparativa",
        "btn_run_baseline": "🚀 Executar Análise de Base",
        "btn_run_crisis": "🔥 Executar Simulação de Crise",
        "btn_run_policy": "▶️ Executar & Comparar",
        "btn_run_sensitivity": "▶️ Executar Análise de Sensibilidade",
        "demo_subtabs": ["📈 Financeiro", "👥 Afluência", "🚚 Logística", "💥 Análise da Crise", "🧠 Deriva Comportamental", "👤 Por Tipo de Comprador"],
        "sub_demographics": "👥 Dados Demográficos",
        "sub_archetypes": "🏷️ Arquétipos Comportamentais (Participantes Reais)",
        "sub_dce": "🧠 Distribuições de Preferências DCE",
        "sub_catalogue": "🛒 Catálogo de Produtos",
        "animation_speed": "Velocidade de Animação (atraso por dia, segundos)",
        "securefood_btn": "📄 Instruções do Cenário",
        "sidebar_interventions_caption": "Estas alavancas são visíveis no separador 🧪 Teoria Comportamental.",
        "exp_nudge": "🛒 Incentivo — Limite de Compra",
        "nudge_cap_on": "Ativar limite de compra por visita",
        "nudge_cap_val": "Máx. unidades por produto por visita",
        "exp_media": "📡 Média / Comunicação",
        "media_intensity": "Intensidade dos média (0 = desligado)",
        "comm_type": "Tipo de comunicação",
        "exp_stockpile": "🏠 Horizonte de Acumulação",
        "stockpile_on": "Substituir horizonte de acumulação",
        "stockpile_days": "Dias de acumulação (horizonte)",
        "sidebar_policy_caption": "Configure as alavancas de política para o separador de Análise de Políticas.",
        "exp_fat_tax": "🧀 Imposto sobre Gordura",
        "fat_tax_on": "Ativar imposto sobre gordura",
        "fat_tax_threshold": "Limiar de gordura %",
        "fat_tax_rate_lbl": "Taxa de sobretaxa (%)",
        "exp_subsidy": "🌿 Subsídio Doméstico / Biológico",
        "subsidy_on": "Ativar subsídio",
        "subsidy_target": "Alvo do subsídio",
        "subsidy_rate": "Taxa de desconto (%)",
        "exp_shock": "🐄 Choque de Fornecimento Doméstico",
        "shock_on": "Ativar choque de fornecimento",
        "shock_start": "Dia de início do choque",
        "shock_duration": "Duração do choque (dias)",
        "shock_severity": "Gravidade do choque (fração bloqueada)",
        "exp_labelling": "🏷️ Rotulagem Nutricional",
        "labelling_on": "Ativar rotulagem",
        "labelling_start": "Dia de início da rotulagem",
        "labelling_health": "Aumento de preferência saudável",
        "labelling_organic": "Aumento de preferência biológica",
        "chart_age": "Distribuição por Idade",
        "label_age": "Idade",
        "label_count": "Contagem",
        "chart_gender": "Distribuição por Género",
        "chart_income": "Grupos de Rendimento",
        "participants": "participantes",
        "chart_radar_title": "Pontuações Médias dos Fatores do Questionário por Arquétipo",
        "factor_labels": ["Preço", "Saúde", "Ambiente", "Bem-estar Animal", "Sensorial/Hábito"],
        "pref_finnish": "Preferência Finlandesa",
        "pref_organic": "Preferência Biológica",
        "pref_price": "Sensibilidade ao Preço",
        "pref_fat": "Gordura % Preferida",
        "arch_names": {
            "price_champion":  "💸 Caçador de Preços",
            "green_buyer":     "🌿 Comprador Verde",
            "health_optimizer":"💪 Otimizador de Saúde",
            "habitual_buyer":  "🔁 Comprador Habitual",
            "price_conscious": "💰 Consciente do Preço",
        },
    },
}

def _t(key: str) -> str:
    """Return the translated string for `key` based on st.session_state['lang']."""
    lang = st.session_state.get("lang", "en")
    return _MAIN_T.get(lang, _MAIN_T["en"]).get(key, _MAIN_T["en"].get(key, key))

def _arch_name(arch_key: str) -> str:
    """Return the translated archetype display name."""
    lang = st.session_state.get("lang", "en")
    names = _MAIN_T.get(lang, _MAIN_T["en"]).get("arch_names", _MAIN_T["en"]["arch_names"])
    return names.get(arch_key, arch_key.replace("_", " ").title())

# ---------------------------------------------------------------------------
# Bundled data loader
# ---------------------------------------------------------------------------
# Firebase export  → Streamlit Secrets  (st.secrets["firebase"]["data"])
#                    Never committed to GitHub — paste your JSON in the
#                    Streamlit Cloud dashboard: Settings → Secrets
#
# Product catalogue → data/master_products.json  (committed to GitHub,
#                    not sensitive)
# ---------------------------------------------------------------------------
import pathlib as _pl

_DATA_DIR      = _pl.Path(__file__).parent / "data"
_PRODUCTS_PATH = str(_DATA_DIR / "master_products.json")

def _load_bundled_data():
    """Return (firebase_dict, products_dict) from Secrets + data/ folder.
    Returns (None, None) if either source is unavailable."""
    # ── Firebase: Streamlit Secrets first, then bundled enriched file ────────
    try:
        firebase_dict = json.loads(st.secrets["firebase"]["data"])
    except Exception:
        firebase_dict = None

    if firebase_dict is None:
        try:
            bundled_path = _DATA_DIR / "food-finland-enriched.json"
            firebase_dict = json.loads(bundled_path.read_text(encoding="utf-8"))
        except Exception:
            firebase_dict = None

    # ── Product catalogue: data/master_products.json ─────────────────────────
    try:
        prod_path = _DATA_DIR / "master_products.json"
        products_dict = json.loads(prod_path.read_text(encoding="utf-8"))
    except Exception:
        products_dict = None

    return firebase_dict, products_dict

if st.session_state.config_data is None:
    try:
        _firebase_dict, _products_dict = _load_bundled_data()
        if _firebase_dict is not None and _products_dict is not None:
            st.session_state.config_data = run_pipeline_from_data(
                _firebase_dict, _products_dict, pool_size=2000, n_archetypes=4
            )
    except Exception:
        pass   # silently skip — user can still upload manually in Data & Population tab

# ===========================================================================
# 2. HELPERS
# ===========================================================================

def render_footer():
    st.markdown("""
    <div class="footer">
        <div class="eu-text">The SecureFood project is funded by the European Union's
        Horizon Europe research and innovation programme — grant agreement No. 101136583.</div>
        <div>© 2026 IAMO XR Lab | GROCERYsim ABM v2.0</div>
    </div>""", unsafe_allow_html=True)


def _product_agents(model):
    return [a for a in model.schedule.agents if isinstance(a, ProductAgent)]


def _collect_model_day(model, day: int, scenario_label: str,
                        collect_products: bool = True) -> tuple[dict, list]:
    """Extract aggregate + optional per-product records from a model after step()."""
    agents   = _product_agents(model)
    d_rev         = sum(a.daily_base_revenue for a in agents)   # constant-price revenue
    d_rev_nominal = sum(a.daily_revenue      for a in agents)   # nominal revenue (inflated prices)
    d_waste  = sum(a.daily_waste      for a in agents)
    d_lost   = sum(a.daily_lost_sales for a in agents)
    d_sales  = sum(a.daily_sales      for a in agents)

    # Pull the latest daily_record written by model.step() for policy/env metrics
    last_rec = model.daily_records[-1] if model.daily_records else {}

    agg = {
        "Day":            day,
        "Scenario":       scenario_label,
        "Revenue":        d_rev,           # constant-price (base_price × units) — falls with inflation/disruption
        "NominalRevenue": d_rev_nominal,   # nominal cash (current_price × units) — rises with inflation
        "AvgPrice":       last_rec.get("AvgPrice", 0.0),   # mean product price; rises with inflation
        "CrisisPhase":    last_rec.get("CrisisPhase",    "pre"),
        "ScenarioEndDay": last_rec.get("ScenarioEndDay", 0),
        "Waste":      d_waste,
        "LostSales":  d_lost,
        "Sales":      d_sales,
        "Consumers":  model.daily_consumer_count,
        "PanicLevel": model.global_panic_level,
        # Environmental
        "CO2Sales":          last_rec.get("CO2Sales",          0.0),
        "CO2Waste":          last_rec.get("CO2Waste",          0.0),
        "CO2Total":          last_rec.get("CO2Total",          0.0),
        "ImportDepPct":      last_rec.get("ImportDepPct",      0.0),
        "DomesticSales":     last_rec.get("DomesticSales",     0),
        "ImportSales":       last_rec.get("ImportSales",       0),
        # Consumer welfare — aggregate
        "BudgetExhaustionRate": last_rec.get("BudgetExhaustionRate", 0.0),
        "FoodStressedPct":      last_rec.get("FoodStressedPct",      0.0),
        "FulfillmentRate":      last_rec.get("FulfillmentRate",      1.0),
        "MeanFatPurchased":     last_rec.get("MeanFatPurchased",     0.0),
        # Consumer welfare — income brackets
        "BudgetExh_Low":    last_rec.get("BudgetExh_Low",    0.0),
        "BudgetExh_Mid":    last_rec.get("BudgetExh_Mid",    0.0),
        "BudgetExh_High":   last_rec.get("BudgetExh_High",   0.0),
        "Fulfillment_Low":  last_rec.get("Fulfillment_Low",  1.0),
        "Fulfillment_Mid":  last_rec.get("Fulfillment_Mid",  1.0),
        "Fulfillment_High": last_rec.get("Fulfillment_High", 1.0),
        "MeanFat_Low":      last_rec.get("MeanFat_Low",      0.0),
        "MeanFat_Mid":      last_rec.get("MeanFat_Mid",      0.0),
        "MeanFat_High":     last_rec.get("MeanFat_High",     0.0),
        "N_Low":            last_rec.get("N_Low",  0),
        "N_Mid":            last_rec.get("N_Mid",  0),
        "N_High":           last_rec.get("N_High", 0),
        # ── Behavioural Theory columns ──────────────────────────────────────
        # Nudge / Rationing (Thaler & Sunstein 2008)
        "GiniAccess":         last_rec.get("GiniAccess",         0.0),
        "PurchaseLimitOn":    last_rec.get("PurchaseLimitOn",    0),
        "PurchaseLimit":      last_rec.get("PurchaseLimit",      0),
        # Theory of Planned Behaviour (Ajzen 1991)
        "AvgSubjectiveNorm":  last_rec.get("AvgSubjectiveNorm",  0.0),
        "AvgTPBIntention":    last_rec.get("AvgTPBIntention",    0.0),
        # FIES Food Security (FAO 2016) — mean score per bracket
        "FIES_Low":           last_rec.get("FIES_Low",           0.0),
        "FIES_Mid":           last_rec.get("FIES_Mid",           0.0),
        "FIES_High":          last_rec.get("FIES_High",          0.0),
        # FIES — fraction severely food-insecure per bracket
        "FIESSevere_Low":     last_rec.get("FIESSevere_Low",     0.0),
        "FIESSevere_Mid":     last_rec.get("FIESSevere_Mid",     0.0),
        "FIESSevere_High":    last_rec.get("FIESSevere_High",    0.0),
        # Stockpile pressure (O'Donoghue & Rabin 1999)
        "StockpilePressure":  last_rec.get("StockpilePressure",  0.0),
        # Media / Communication Channel (McCombs & Shaw 1972)
        "MediaIntensity":     last_rec.get("MediaIntensity",     0.0),
        "MediaType":          last_rec.get("MediaType",          "neutral"),
        "MediaPanicEffect":   last_rec.get("MediaPanicEffect",   0.0),
        # Store calibration & daily loss breakdown
        "StoreTier":          last_rec.get("StoreTier",          "Unknown"),
        "DailyLossStockout":  last_rec.get("DailyLossStockout",  0.0),
        "DailyLossPrice":     last_rec.get("DailyLossPrice",     0.0),
    }

    prod_rows = []
    if collect_products:
        for a in agents:
            prod_rows.append({
                "Day":        day,
                "Scenario":   scenario_label,
                "Product":    a.name,
                "Category":   a.category,
                "Shelf":      a.snap_shelf,
                "Storage":    a.snap_storage,
                "Pending":    a.snap_pending,
                "Revenue":    a.daily_revenue,
                "Sales":      a.daily_sales,
                "Waste":      a.daily_waste,
                "LostSales":  a.daily_lost_sales,
                "Price":      a.current_price,
                "NearExpiry": a.daily_near_expiry_sold,
                "CO2Sales":   round(a.daily_co2_sales,  2),
                "CO2Waste":   round(a.daily_co2_waste,  2),
                "DomesticSales": a.daily_domestic_sales,
                "ImportSales":   a.daily_import_sales,
                "Origin":        a.origin,
                "FatContent":    a.fat_content,
            })
    return agg, prod_rows


# ===========================================================================
# 2b. CHART ANALYSIS HELPER
# ===========================================================================

def _render_analysis(
    df: pd.DataFrame,
    metric: str,
    params: dict,
    *,
    baseline_label: str = "Baseline",
    crisis_label: str = "Crisis",
    prefix: str = "",
    suffix: str = "",
    higher_is_better: bool = True,
    decimals: int = 1,
    recovery_tolerance: float = 0.05,
) -> None:
    """
    Render a compact, always-visible statistical analysis row below a chart.

    Two modes
    ---------
    Comparative  (df has both baseline_label and crisis_label)
        → columns: Baseline avg | Crisis phase avg | Recovery phase avg | Days to recover
        → one-line conclusion sentence with direction + severity
    Single-series (only one Scenario in df)
        → columns: Overall avg | Peak value | Lowest value | Trend
    """
    if df is None or df.empty or metric not in df.columns:
        return

    scenarios = df["Scenario"].unique().tolist()
    is_comparative = (baseline_label in scenarios and crisis_label in scenarios)

    cri_start = params.get("cri_start", 0)
    cri_dur   = params.get("cri_duration", 0)
    max_day   = int(df["Day"].max())
    cri_end   = (cri_start + cri_dur) if cri_dur > 0 else (max_day + 1)

    def _avg(d: pd.DataFrame, lo: int = 1, hi: int = None):
        hi = hi if hi is not None else (max_day + 1)
        sub = d[(d["Day"] >= lo) & (d["Day"] < hi)][metric]
        return float(sub.mean()) if not sub.empty else None

    def _pct(new, ref):
        if new is None or ref is None or abs(ref) < 1e-9:
            return None
        return (new - ref) / abs(ref) * 100

    def _fmt(v):
        if v is None:
            return "—"
        return f"{prefix}{v:,.{decimals}f}{suffix}".strip()

    def _delta_color(pct_val):
        """'normal' = green when positive (for higher-is-better); 'inverse' = red when positive."""
        if pct_val is None:
            return "off"
        return "normal" if (pct_val >= 0) == higher_is_better else "inverse"

    # ── Comparative mode ─────────────────────────────────────────────────────
    if is_comparative:
        df_b = df[df["Scenario"] == baseline_label]
        df_c = df[df["Scenario"] == crisis_label]

        b_full = _avg(df_b)
        b_dur  = _avg(df_b, cri_start, cri_end)
        c_dur  = _avg(df_c, cri_start, cri_end)
        has_post = (cri_dur > 0 and cri_end <= max_day)
        b_post = _avg(df_b, cri_end) if has_post else None
        c_post = _avg(df_c, cri_end) if has_post else None

        pct_dur  = _pct(c_dur,  b_dur)
        pct_post = _pct(c_post, b_post) if has_post else None
        pct_all  = _pct(_avg(df_c), b_full)

        # Recovery day: first day after crisis_end where crisis ≈ baseline (within tolerance)
        recovery_days = None
        if has_post:
            for d in range(cri_end, max_day + 1):
                cv = df_c.loc[df_c["Day"] == d, metric].values
                bv = df_b.loc[df_b["Day"] == d, metric].values
                if len(cv) and len(bv) and abs(bv[0]) > 1e-9:
                    if abs(cv[0] - bv[0]) / abs(bv[0]) <= recovery_tolerance:
                        recovery_days = d - cri_end
                        break

        n_cols = 4 if has_post else 3
        cols = st.columns(n_cols)

        cols[0].metric(
            "Baseline avg",
            _fmt(b_full),
            help="Mean across the full simulation (no-crisis scenario)",
        )
        cols[1].metric(
            "Crisis phase avg",
            _fmt(c_dur),
            delta=(f"{pct_dur:+.1f}% vs baseline" if pct_dur is not None else None),
            delta_color=_delta_color(pct_dur),
            help=f"Days {cri_start}–{cri_end - 1} of the crisis scenario",
        )
        if has_post:
            cols[2].metric(
                "Recovery phase avg",
                _fmt(c_post),
                delta=(f"{pct_post:+.1f}% vs baseline" if pct_post is not None else None),
                delta_color=_delta_color(pct_post),
                help=f"Post-crisis days {cri_end}–{max_day}",
            )
            if recovery_days is not None:
                cols[3].metric(
                    "Days to recover",
                    str(recovery_days),
                    help=f"First day the crisis metric came within {int(recovery_tolerance*100)}% of baseline",
                )
            else:
                cols[3].metric(
                    "Days to recover",
                    "Outside window",
                    delta="no full recovery",
                    delta_color="off",
                )
        else:
            cols[2].metric(
                "Full-sim crisis avg",
                _fmt(_avg(df_c)),
                delta=(f"{pct_all:+.1f}% vs baseline" if pct_all is not None else None),
                delta_color=_delta_color(pct_all),
                help="Crisis scenario average across all days (no recovery phase configured)",
            )

        # Conclusion sentence
        if pct_dur is not None:
            direction = "higher" if pct_dur > 0 else "lower"
            good = (pct_dur > 0) == higher_is_better
            icon = "🟢" if good else ("🟡" if abs(pct_dur) < 10 else "🔴")
            label = metric.replace("_", " ")
            parts = [
                f"{icon} During the crisis, **{label}** was **{abs(pct_dur):.1f}% {direction}** "
                f"than the baseline ({_fmt(c_dur)} vs {_fmt(b_dur)})."
            ]
            if has_post and pct_post is not None:
                rec_icon = "🟢" if abs(pct_post) < 5 else ("🟡" if abs(pct_post) < 15 else "🔴")
                rec_word = "fully recovered" if abs(pct_post) < 5 else "partially recovered"
                parts.append(
                    f"{rec_icon} Recovery phase: {rec_word} "
                    f"({_fmt(c_post)}, {pct_post:+.1f}% vs baseline)."
                )
                if recovery_days is not None:
                    parts.append(f"⏱️ Returned to within {int(recovery_tolerance*100)}% of baseline after **{recovery_days} day(s)**.")
                else:
                    parts.append("⚠️ Full recovery not observed within the simulation window.")
            st.caption("  ".join(parts))

    # ── Single-series mode ───────────────────────────────────────────────────
    else:
        sel_sc = scenarios[0] if scenarios else None
        df_s   = df[df["Scenario"] == sel_sc] if sel_sc else df

        s_mean = _avg(df_s)
        s_max  = float(df_s[metric].max()) if not df_s.empty else None
        s_min  = float(df_s[metric].min()) if not df_s.empty else None
        peak_day = int(df_s.loc[df_s[metric].idxmax(), "Day"]) if not df_s.empty and s_max else None

        # Simple trend: compare second half vs first half
        mid = max_day // 2
        first_half = _avg(df_s, 1, mid)
        second_half = _avg(df_s, mid, max_day + 1)
        trend_pct = _pct(second_half, first_half)

        cols = st.columns(4)
        cols[0].metric("Average", _fmt(s_mean), help="Mean across the simulation")
        cols[1].metric("Peak", _fmt(s_max), help=f"Highest value (day {peak_day})" if peak_day else None)
        cols[2].metric("Minimum", _fmt(s_min))
        if trend_pct is not None:
            direction = "up" if trend_pct > 0 else "down"
            cols[3].metric(
                "2nd-half trend",
                f"{trend_pct:+.1f}%",
                delta=f"vs first half",
                delta_color=_delta_color(trend_pct),
                help="Change from first half to second half of the simulation",
            )
        if peak_day:
            label = metric.replace("_", " ")
            st.caption(
                f"📌 **{label}** averaged {_fmt(s_mean)}, "
                f"peaking at {_fmt(s_max)} on day {peak_day}."
            )


# ===========================================================================
# 3. SIDEBAR — SHARED PARAMETERS
# ===========================================================================

def build_sidebar_params():
    st.sidebar.title(_t("sidebar_title"))
    if st.sidebar.button("🎓 Tour", help="Restart the guided tour", key="restart_tour_btn"):
        st.session_state["tour_step"] = 1
        st.rerun()
    st.sidebar.divider()

    st.sidebar.header(_t("sidebar_general"))
    days_to_run    = st.sidebar.slider(_t("duration_days"), 7, 1825, 60,
                                       help="1 year = 365 days | 5 years = 1825 days")
    start_month    = st.sidebar.selectbox(_t("start_month"), list(range(1, 13)), index=0)
    base_consumers = st.sidebar.number_input(_t("base_consumers"), 10, 5000, 100,
                                              help="Approximate — actual count varies by weekday & season")

    # Auto-calibrated store tier display
    _tier_info = {
        "Small (neighbourhood shop)":    ("🏪", "< 200/day",   "compact neighbourhood shop"),
        "Medium (supermarket)":          ("🏬", "200–499/day",  "mid-size supermarket"),
        "Large (hypermarket)":           ("🏢", "500–1499/day", "large-format hypermarket"),
        "Very Large (wholesale / hyper)":("🏭", "≥ 1500/day",  "wholesale / hypermarket"),
    }
    _tier = SupermarketModel.store_tier_label(int(base_consumers))
    _icon, _range, _desc = _tier_info.get(_tier, ("🏪", "", ""))
    st.sidebar.info(
        f"**Auto-calibrated:** {_icon} **{_tier}** ({_range})\n\n"
        f"Shelf, storage & initial stock scale automatically to this traffic."
    )

    with st.sidebar.expander("ℹ️ Store size & calibration logic"):
        st.markdown(f"""
**Current tier:** {_icon} {_tier}

---
**Store tiers** (by base consumers/day)

| Tier | Consumers/day |
|---|---|
| 🏪 Small | < 200 |
| 🏬 Medium | 200 – 499 |
| 🏢 Large | 500 – 1 499 |
| 🏭 Very Large | ≥ 1 500 |

---
**Auto-calibrated capacities** (per product)

The model estimates each product's daily demand as:

> `demand = base_consumers × avg_basket_qty`

Then sizes shelves and storage accordingly:

| Product perishability | Shelf cover | Shelf cap formula |
|---|---|---|
| Perishable (≤ 7 days) | 1.5 days | `demand × 1.5` |
| Medium (8 – 30 days) | 2.5 days | `demand × 2.5` |
| Dry / canned (> 30 days) | 4.0 days | `demand × 4.0` |

Storage capacity = `demand × (lead_time + 4-day safety buffer)`, minimum 2× shelf cap.

**Initial fill:**  shelf = 75% of max | storage = 60% of max.

---
**Reorder logic**

- When `storage < reorder_point × max_storage` → an order is placed automatically
- Order fills storage back to `target_stock × max_storage`
- Delivery arrives after **lead time** days

---
**Daily traffic variation**

Actual visitors = `base × weekday factor × month factor × noise (±10%)`

| Weekday | Factor | Month | Factor |
|---|---|---|---|
| Monday | 0.80 | Jan–Feb | 0.90 |
| Tuesday | 0.90 | Mar–May | 1.00 |
| Wednesday | 0.90 | Jun–Jul | 1.10 |
| Thursday | 1.00 | Aug–Oct | 1.00 |
| Friday | 1.10 | November | 1.10 |
| Saturday | 1.30 | December | 1.30 |
| Sunday | 0.70 | | |
""")

    st.sidebar.header(_t("sidebar_logistics"))
    reorder_pt  = st.sidebar.slider(_t("reorder_pt"), 10, 90, 30) / 100.0
    target_stock = st.sidebar.slider(_t("restock_target"), 50, 100, 90) / 100.0
    lead_time   = st.sidebar.slider(_t("lead_time"), 1, 14, 2)

    st.sidebar.header(_t("sidebar_crisis"))
    cri_start    = st.sidebar.slider(_t("crisis_start"), 1, max(2, days_to_run - 1),
                                     min(int(days_to_run / 2), days_to_run - 1))
    inflation    = st.sidebar.slider(_t("price_inflation"), 0, 150, 25)
    disruption   = st.sidebar.slider(_t("supply_disruption"), 0, 30, 5)
    # Crisis Duration: how many days the crisis lasts before conditions normalise.
    # 0 = indefinite (crisis runs to end of simulation — legacy behaviour).
    # >0 = crisis ends after this many days; prices revert, supply resumes,
    #      panic decays naturally → a full "crisis + recovery" arc is visible.
    max_dur      = max(1, days_to_run - cri_start)
    cri_duration = st.sidebar.slider(
        _t("crisis_duration"), 0, max_dur, 0,
        help="0 = crisis runs to end of simulation.  Set >0 to model a temporary "
             "shock (e.g. fuel price spike) — prices and supply return to normal "
             "after this many days so you can measure the full recovery arc."
    )

    st.sidebar.header(_t("sidebar_behaviour"))
    panic_sens   = st.sidebar.slider(_t("panic_sensitivity"), 0.0, 1.0, 0.50, 0.05)
    hoarding     = st.sidebar.slider(_t("hoarding_factor"), 1.0, 3.0, 1.5, 0.1)

    st.sidebar.header(_t("sidebar_interventions"))
    st.sidebar.caption(_t("sidebar_interventions_caption"))

    with st.sidebar.expander(_t("exp_nudge"), expanded=False):
        purchase_limit_on = st.checkbox(
            _t("nudge_cap_on"), False, key="nudge_limit_on",
            help="Rationing: cap the number of units any one consumer can buy per product per visit."
        )
        purchase_limit_val = st.slider(
            _t("nudge_cap_val"), 1, 20, 3,
            key="nudge_limit_val",
            help="Thaler & Sunstein (2008): a purchase cap reduces panic-hoarding but may lower equity."
        )
        purchase_limit = purchase_limit_val if purchase_limit_on else None

    with st.sidebar.expander(_t("exp_media"), expanded=False):
        media_intensity = st.slider(
            _t("media_intensity"), 0.0, 1.0, 0.0, 0.05,
            key="media_intensity",
            help="How strongly media amplifies or dampens panic each day."
        )
        communication_type = st.selectbox(
            _t("comm_type"), ["neutral", "panic", "calming"],
            key="comm_type",
            help=(
                "panic = sensationalist coverage (raises global panic); "
                "calming = reassuring coverage (lowers panic); "
                "neutral = factual reporting (no panic effect)."
            ),
        )

    with st.sidebar.expander(_t("exp_stockpile"), expanded=False):
        stockpile_days_on = st.checkbox(
            _t("stockpile_on"), False, key="stockpile_on",
            help="Override the agent's default stockpile planning horizon (β-δ quasi-hyperbolic model)."
        )
        stockpile_days_val = st.slider(
            _t("stockpile_days"), 1, 14, 3,
            key="stockpile_days_val",
            help="O'Donoghue & Rabin (1999): agents plan to hold this many days of supply at home."
        )
        stockpile_days_override = stockpile_days_val if stockpile_days_on else None

    st.sidebar.header(_t("sidebar_mc"))
    mc_runs = st.sidebar.number_input(_t("mc_runs"), 3, 100, 10,
                                       help="More runs = tighter confidence intervals")

    # -----------------------------------------------------------------------
    st.sidebar.header(_t("sidebar_policy"))
    st.sidebar.caption(_t("sidebar_policy_caption"))

    with st.sidebar.expander(_t("exp_fat_tax"), expanded=False):
        fat_tax_active    = st.checkbox(_t("fat_tax_on"), False, key="pol_fat_active")
        fat_tax_threshold = st.slider(_t("fat_tax_threshold"), 0.5, 5.0, 3.5, 0.5,
                                      key="pol_fat_thresh",
                                      help="Products with fat_content ≥ this value are taxed")
        fat_tax_rate      = st.slider(_t("fat_tax_rate_lbl"), 5, 50, 20, 5,
                                      key="pol_fat_rate") / 100.0

    with st.sidebar.expander(_t("exp_subsidy"), expanded=False):
        sub_active = st.checkbox(_t("subsidy_on"), False, key="pol_sub_active")
        sub_target = st.selectbox(_t("subsidy_target"), ["domestic", "organic", "both"],
                                  key="pol_sub_target")
        sub_rate   = st.slider(_t("subsidy_rate"), 5, 40, 15, 5,
                                key="pol_sub_rate") / 100.0

    with st.sidebar.expander(_t("exp_shock"), expanded=False):
        shock_active   = st.checkbox(_t("shock_on"), False, key="pol_shock_active")
        shock_day      = st.slider(_t("shock_start"), 1, max(2, days_to_run - 1),
                                   min(30, max(1, days_to_run // 2)), key="pol_shock_day")
        shock_duration = st.slider(_t("shock_duration"), 1, 120, 30, key="pol_shock_dur")
        shock_severity = st.slider(_t("shock_severity"), 0.1, 1.0, 0.70, 0.05,
                                   key="pol_shock_sev")

    with st.sidebar.expander(_t("exp_labelling"), expanded=False):
        lab_active        = st.checkbox(_t("labelling_on"), False, key="pol_lab_active")
        lab_day           = st.slider(_t("labelling_start"), 1, max(2, days_to_run - 1),
                                      1, key="pol_lab_day")
        lab_health_boost  = st.slider(_t("labelling_health"), 0.0, 0.4, 0.15, 0.05,
                                      key="pol_lab_health")
        lab_organic_boost = st.slider(_t("labelling_organic"), 0.0, 0.3, 0.10, 0.05,
                                      key="pol_lab_organic")

    policy_cfg = {
        "fat_tax_active":     fat_tax_active,
        "fat_tax_threshold":  fat_tax_threshold,
        "fat_tax_rate":       fat_tax_rate,
        "subsidy_active":     sub_active,
        "subsidy_target":     sub_target,
        "subsidy_rate":       sub_rate,
        "domestic_shock_active":   shock_active,
        "domestic_shock_day":      shock_day,
        "domestic_shock_duration": shock_duration,
        "domestic_shock_severity": shock_severity,
        "labelling_active":        lab_active,
        "labelling_day":           lab_day,
        "labelling_health_boost":  lab_health_boost,
        "labelling_organic_boost": lab_organic_boost,
    }

    return {
        "days":                    days_to_run,
        "month":                   start_month,
        "base_con":                int(base_consumers),
        "reorder":                 reorder_pt,
        "target":                  target_stock,
        "lead":                    lead_time,
        "cri_start":               cri_start,
        "cri_duration":            int(cri_duration),   # 0 = indefinite, >0 = days the crisis lasts
        "inf":                     float(inflation),
        "dis":                     int(disruption),
        "panic":                   panic_sens,
        "hoard":                   hoarding,
        "mc_runs":                 int(mc_runs),
        "policy_cfg":              policy_cfg,
        # Behavioural interventions
        "purchase_limit":          purchase_limit,
        "media_intensity":         media_intensity,
        "communication_type":      communication_type,
        "stockpile_days":          stockpile_days_override,
    }


def _make_model(
    params: dict,
    is_crisis: bool,
    seed: int,
    ai_recs=None,
    policy_cfg: dict = None,
) -> SupermarketModel:
    return SupermarketModel(
        config_data          = st.session_state.config_data,
        base_consumers       = params["base_con"],
        start_month          = params["month"],
        reorder_pt           = params["reorder"],
        target_stock         = params["target"],
        lead_time            = params["lead"],
        is_crisis_mode       = is_crisis,
        scenario_start_day   = params["cri_start"],
        crisis_duration      = params.get("cri_duration", 0),
        inflation_pct        = params["inf"],
        disruption_days      = params["dis"],
        panic_sens           = params["panic"],
        hoarding_fac         = params["hoard"],
        fixed_seed           = seed,
        ai_recs              = ai_recs,
        policy_cfg           = policy_cfg,
        purchase_limit            = params.get("purchase_limit"),
        media_intensity           = params.get("media_intensity", 0.0),
        communication_type        = params.get("communication_type", "neutral"),
        stockpile_days_override   = params.get("stockpile_days"),
    )


# ===========================================================================
# 4. TAB: DATA & POPULATION
# ===========================================================================

def render_data_tab():
    st.header(_t("header_data"))

    # ── Bundled-data status banner ────────────────────────────────────────────
    _has_secret  = "firebase" in st.secrets and "data" in st.secrets["firebase"]
    _has_catalogue = (_DATA_DIR / "master_products.json").exists()
    if st.session_state.config_data is not None:
        cfg_stats = st.session_state.config_data.get("stats", {})
        n_real  = cfg_stats.get("n_real", "?")
        n_pool  = cfg_stats.get("pool_size", "?")
        n_prods = len(st.session_state.config_data.get("products", []))
        st.success(
            f"✅ **Initial data loaded** — {n_real} real participants · "
            f"{n_pool} synthetic agents · {n_prods} products in catalogue.\n\n"
            "You can jump straight to **🎮 Interactive Demo**. "
            "Use the expander below only if you want to load a different dataset."
        )
    elif not (_has_secret and _has_catalogue):
        missing = []
        if not _has_secret:    missing.append("Firebase secret (add in Streamlit Cloud → Settings → Secrets)")
        if not _has_catalogue: missing.append("product catalogue (data/master_products.json)")
        st.warning("⚠️ Bundled data not fully configured. Missing: " + " · ".join(missing) +
                   ". Upload files manually below or configure the missing source.")

    with st.expander("🔄 Reload / Override Data Files", expanded=(st.session_state.config_data is None)):
        st.markdown(
            "Upload a new **Firebase export** and/or **product catalogue** to rebuild the "
            "agent population pool from scratch."
        )

        col_fb, col_prod = st.columns(2)
        with col_fb:
            fb_file = st.file_uploader(
                "📂 Firebase Export JSON",
                type=["json"],
                key="upload_firebase",
                help="The JSON file exported from your Firebase Realtime Database",
            )
        with col_prod:
            prod_file = st.file_uploader(
                "📂 Product Catalogue JSON",
                type=["json"],
                key="upload_products",
                help="master_products.json exported from Unity",
            )

        pool_size    = st.number_input("Population Pool Size", 100, 50000, 2000,
                                       help="Total synthetic agents to generate (real + bootstrapped)")
        n_archetypes = st.selectbox("Number of Behavioural Archetypes", [2, 3, 4, 5], index=2)

        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("🔄 Process Uploaded Files", type="primary"):
            if fb_file is None or prod_file is None:
                st.error("Please upload both JSON files before processing.")
                return
            with st.spinner("Parsing profiles, running K-Means clustering, bootstrapping…"):
                try:
                    firebase_dict  = json.load(fb_file)
                    products_dict  = json.load(prod_file)
                    config         = run_pipeline_from_data(
                        firebase_dict, products_dict,
                        pool_size=int(pool_size), n_archetypes=int(n_archetypes),
                    )
                    st.session_state.config_data = config
                    for k in ["sim_results","sim_stock","sim_scm_log","sim_waste",
                              "sim_product_recs","sim_model_crisis",
                              "mc_stage","data_base_raw","data_base_opt",
                              "data_crisis","ai_recs","prod_stats_raw"]:
                        st.session_state[k] = None if k != "mc_stage" else 0
                    st.success("✅ Population rebuilt from uploaded files!")
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    return

        if col_btn2.button("♻️ Reload Bundled Files"):
            try:
                _firebase_dict, _products_dict = _load_bundled_data()
                if _firebase_dict is None or _products_dict is None:
                    st.error("Bundled data not available. Firebase secret or product catalogue missing.")
                else:
                    st.session_state.config_data = run_pipeline_from_data(
                        _firebase_dict, _products_dict,
                        pool_size=int(pool_size), n_archetypes=int(n_archetypes),
                    )
                    for k in ["sim_results","sim_stock","sim_scm_log","sim_waste",
                              "sim_product_recs","sim_model_crisis",
                              "mc_stage","data_base_raw","data_base_opt",
                              "data_crisis","ai_recs","prod_stats_raw"]:
                        st.session_state[k] = None if k != "mc_stage" else 0
                    st.success("✅ Reloaded from bundled files!")
            except Exception as e:
                st.error(f"Reload failed: {e}")

    if st.session_state.config_data is None:
        st.info("No data loaded yet. Upload files and click **Process Data**.")
        return

    cfg   = st.session_state.config_data
    stats = cfg["stats"]
    pool  = cfg["population"]
    prods = cfg["products"]

    # ---- Summary metrics ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Real Participants", stats["n_real"])
    c2.metric("Synthetic Agents",  stats["pool_size"] - stats["n_real"])
    c3.metric("Total Pool Size",   stats["pool_size"])
    c4.metric("Products in Catalogue", len(prods))

    if stats.get("n_skipped", 0):
        st.warning(
            f"⚠️ {stats['n_skipped']} participant(s) were skipped "
            "(basket had no products matching the catalogue)."
        )

    st.divider()
    st.subheader(_t("sub_demographics"))

    real_profiles = [p for p in pool if p.get("is_real")]
    df_demo = pd.DataFrame(real_profiles)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if "age" in df_demo:
            fig_age = px.histogram(df_demo, x="age", nbins=10, title=_t("chart_age"),
                                   color_discrete_sequence=["#003399"])
            fig_age.update_layout(template="plotly_white", showlegend=False,
                                  xaxis_title=_t("label_age"), yaxis_title=_t("label_count"))
            st.plotly_chart(fig_age, use_container_width=True, config=_PLOTLY_CFG)

    with col_b:
        if "gender" in df_demo:
            gender_counts = df_demo["gender"].value_counts().reset_index()
            gender_counts.columns = ["Gender", "Count"]
            fig_gen = px.pie(gender_counts, names="Gender", values="Count",
                             title=_t("chart_gender"),
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig_gen.update_layout(template="plotly_white")
            st.plotly_chart(fig_gen, use_container_width=True, config=_PLOTLY_CFG)

    with col_c:
        if "income_group" in df_demo:
            inc_counts = df_demo["income_group"].value_counts().reset_index()
            inc_counts.columns = ["Income", "Count"]
            fig_inc = px.bar(inc_counts, x="Income", y="Count",
                             title=_t("chart_income"),
                             color_discrete_sequence=["#003399"])
            fig_inc.update_layout(template="plotly_white", xaxis_title="",
                                  xaxis_tickangle=-30)
            st.plotly_chart(fig_inc, use_container_width=True, config=_PLOTLY_CFG)

    # ---- Archetype distribution ----
    _arch_hdr, _arch_info = st.columns([8, 1])
    with _arch_hdr:
        st.subheader(_t("sub_archetypes"))
    with _arch_info:
        with st.popover("ℹ️"):
            st.markdown("""
**Buyer-type profiles** are derived from PCA on 5 survey attitude dimensions
(price · health · environment · animal welfare · sensory/habit) and assigned to
each participant via k-means clustering.

| Type | Core trait | Crisis behaviour |
|------|-----------|-----------------|
| 💸 **Price Champion** | Maximises value; switches brands freely when prices rise | Early substitution, lower budget exhaustion, mild hoarding |
| 🌿 **Green Buyer** | Prefers organic & domestic; accepts premium pricing | Higher stockout risk on niche lines; slow to panic |
| 💪 **Health Optimizer** | Fat/nutrition-focused; continuously adjusts diet | Basket disruption when preferred fat-profile products are absent |
| 🔁 **Habitual Buyer** | Brand-loyal; resists change until forced | Highest panic score; hardest hit by stockouts |

**Theoretical grounding**
- Substitution tolerance, hoarding multiplier, and price tolerance differ per archetype
  (Prospect Theory λ = 2.25; TPB intention weights 0.49 / 0.26 / 0.39).
- β-δ temporal discounting means present-biased agents (esp. Habitual Buyers)
  stockpile more aggressively as panic rises (Hendel & Nevo 2006).
- FIES food-insecurity flags are summed per agent daily (0 = none → 4 = severe).
""")
    arch_data = stats.get("archetypes_real", {})
    if arch_data:
        col_arch, col_radar = st.columns([1, 2])
        with col_arch:
            for a, n in arch_data.items():
                emoji = ARCHETYPE_EMOJI.get(a, "•")
                color = ARCHETYPE_COLORS.get(a, "#999")
                pct   = n / max(stats["n_real"], 1) * 100
                st.markdown(
                    f'<div style="margin:6px 0; padding:8px 12px; '
                    f'border-left:4px solid {color}; border-radius:4px; '
                    f'background:#f8f9fa; color:#1a2035;">'
                    f'<b>{_arch_name(a)}</b>: '
                    f'{n} {_t("participants")} ({pct:.0f}%)</div>',
                    unsafe_allow_html=True,
                )

        with col_radar:
            # Radar chart of mean factor scores per archetype
            factor_cols = ["q_price", "q_health", "q_environment",
                           "q_animal_welfare", "q_sensory_habit"]
            factor_labels = _t("factor_labels")
            df_real = pd.DataFrame(real_profiles)

            fig_radar = go.Figure()
            for arch in ARCHETYPE_LABELS:
                sub = df_real[df_real["archetype"] == arch]
                if sub.empty:
                    continue
                means = [sub[c].mean() for c in factor_cols if c in sub.columns]
                if len(means) < len(factor_labels):
                    continue
                means += means[:1]
                labels_loop = factor_labels + factor_labels[:1]
                fig_radar.add_trace(go.Scatterpolar(
                    r=means, theta=labels_loop,
                    fill="toself", name=_arch_name(arch),
                    line_color=ARCHETYPE_COLORS.get(arch, "#999"),
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=_t("chart_radar_title"),
                template="plotly_white",
            )
            st.plotly_chart(fig_radar, use_container_width=True, config=_PLOTLY_CFG)

    # ---- DCE preference distributions ----
    st.subheader(_t("sub_dce"))
    pref_cols = {
        "finnish_preference": _t("pref_finnish"),
        "organic_preference": _t("pref_organic"),
        "price_sensitivity":  _t("pref_price"),
        "preferred_fat":      _t("pref_fat"),
    }
    cols_pref = st.columns(4)
    df_all = pd.DataFrame(pool)
    for (col_key, col_label), col_widget in zip(pref_cols.items(), cols_pref):
        if col_key in df_all.columns:
            fig = px.histogram(df_all, x=col_key, nbins=15, title=col_label,
                               color_discrete_sequence=["#2980b9"])
            fig.update_layout(template="plotly_white", showlegend=False,
                              xaxis_title="", yaxis_title=_t("label_count"), height=250)
            col_widget.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)

    # ---- Product catalogue preview ----
    st.subheader(_t("sub_catalogue"))
    df_prods = pd.DataFrame(prods)
    cols_show = [c for c in ["name","category","price","fat_content",
                             "is_bio","is_plant_based","shelf_life_days",
                             "initial_stock_shelf","initial_stock_storage"]
                 if c in df_prods.columns]
    st.dataframe(df_prods[cols_show].sort_values("category"), use_container_width=True)

    render_footer()


# ===========================================================================
# 5. TAB: INTERACTIVE DEMO
# ===========================================================================

def render_demo_tab(params: dict):
    st.header(_t("header_demo"))
    st.markdown(
        "Run a single paired simulation (Baseline vs Crisis) and watch the "
        "results update in real time."
    )

    if st.session_state.config_data is None:
        st.warning("⚠️ Load data in the **🏠 Data & Population** tab first.")
        return

    run_speed = st.slider(_t("animation_speed"), 0.0, 0.2, 0.02, 0.01)

    if st.button(_t("btn_run_demo"), type="primary"):
        SEED = 42
        model_base   = _make_model(params, is_crisis=False, seed=SEED)
        model_crisis = _make_model(params, is_crisis=True,  seed=SEED)

        results      = []
        stock_rows   = []
        progress     = st.progress(0, text="Simulating…")
        chart_spot   = st.empty()

        for day in range(1, params["days"] + 1):
            model_base.step()
            model_crisis.step()
            model_base.collect_product_snapshot()
            model_crisis.collect_product_snapshot()
            model_base.collect_preference_snapshot()
            model_crisis.collect_preference_snapshot()

            agg_b, prod_b = _collect_model_day(model_base,   day, "Baseline")
            agg_c, prod_c = _collect_model_day(model_crisis, day, "Crisis")
            results.append(agg_b)
            results.append(agg_c)
            stock_rows.extend(prod_b)
            stock_rows.extend(prod_c)

            if day % max(1, params["days"] // 100) == 0 or day == params["days"]:
                df_live = pd.DataFrame(results)
                fig = px.line(
                    df_live, x="Day", y="Revenue", color="Scenario",
                    title="Daily Revenue at Baseline Prices — Baseline vs Crisis  (falls with inflation & disruption)",
                    color_discrete_map={"Baseline": "#2E8B57", "Crisis": "#DC143C"},
                    line_dash="Scenario",
                    line_dash_map={"Baseline": "solid", "Crisis": "dash"},
                )
                fig.add_vline(x=params["cri_start"], line_dash="dot",
                              line_color="orange", annotation_text="Crisis Start")
                if params.get("cri_duration", 0) > 0:
                    cri_end_day = params["cri_start"] + params["cri_duration"]
                    fig.add_vline(x=cri_end_day, line_dash="dash",
                                  line_color="steelblue", annotation_text="Crisis End / Recovery")
                fig.update_layout(template="plotly_white",
                                  yaxis_title="Revenue (€, constant baseline prices)")
                chart_spot.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)
                progress.progress(day / params["days"], text=f"Day {day}/{params['days']}")
                time.sleep(run_speed)

        progress.progress(1.0, text="Complete ✓")
        time.sleep(0.4)
        progress.empty()

        # Assemble SCM log
        log_b = pd.DataFrame(model_base.truck.log)
        log_c = pd.DataFrame(model_crisis.truck.log)
        if not log_b.empty:
            log_b["Scenario"] = "Baseline"
        if not log_c.empty:
            log_c["Scenario"] = "Crisis"

        st.session_state.sim_results      = pd.DataFrame(results)
        st.session_state.sim_stock        = pd.DataFrame(stock_rows)
        st.session_state.sim_scm_log      = pd.concat([log_b, log_c], ignore_index=True)
        st.session_state.sim_model_crisis = model_crisis

        # Behavioural learning preference drift snapshots
        pref_b = pd.DataFrame(getattr(model_base,   "_pref_snapshots", []))
        pref_c = pd.DataFrame(getattr(model_crisis, "_pref_snapshots", []))
        if not pref_b.empty:
            pref_b["Scenario"] = "Baseline"
        if not pref_c.empty:
            pref_c["Scenario"] = "Crisis"
        st.session_state.sim_pref_drift = pd.concat([pref_b, pref_c], ignore_index=True)

        waste_b = pd.DataFrame(model_base.food_waste_log.records)
        waste_c = pd.DataFrame(model_crisis.food_waste_log.records)
        if not waste_b.empty:
            waste_b["Scenario"] = "Baseline"
        if not waste_c.empty:
            waste_c["Scenario"] = "Crisis"
        st.session_state.sim_waste = pd.concat([waste_b, waste_c], ignore_index=True)

        st.success("Simulation complete — explore the other tabs for detailed analysis.")

    if st.session_state.sim_results is None:
        return

    df         = st.session_state.sim_results
    df_stock   = st.session_state.sim_stock
    df_log     = st.session_state.sim_scm_log
    model_cris = st.session_state.sim_model_crisis

    sub1, sub2, sub3, sub4, sub5, sub6 = st.tabs(_t("demo_subtabs"))

    with sub1:
        # ── Helper: add crisis start/end vlines to any figure ────────────────
        def _add_crisis_lines(fig):
            if params.get("cri_start"):
                fig.add_vline(x=params["cri_start"], line_dash="dot",
                              line_color="orange", annotation_text="Crisis Start",
                              annotation_position="top left")
            cri_dur = params.get("cri_duration", 0)
            if cri_dur and cri_dur > 0:
                cri_end = params["cri_start"] + cri_dur
                fig.add_vline(x=cri_end, line_dash="dash",
                              line_color="steelblue", annotation_text="Recovery",
                              annotation_position="top right")

        # ── Recovery KPI banner (only shown when crisis_duration > 0) ────────
        cri_dur = params.get("cri_duration", 0)
        if cri_dur and cri_dur > 0:
            cri_end = params["cri_start"] + cri_dur
            df_base_c = df[(df["Scenario"] == "Baseline") & (df["Day"] > params["cri_start"])]
            df_cris_a = df[(df["Scenario"] == "Crisis")   & (df["Day"] >= params["cri_start"]) & (df["Day"] < cri_end)]
            df_cris_r = df[(df["Scenario"] == "Crisis")   & (df["Day"] >= cri_end)]
            baseline_avg   = df_base_c["Revenue"].mean() if len(df_base_c) else 1
            active_avg     = df_cris_a["Revenue"].mean() if len(df_cris_a) else 0
            recovery_avg   = df_cris_r["Revenue"].mean() if len(df_cris_r) else 0
            impact_pct     = round((active_avg   - baseline_avg) / max(0.01, baseline_avg) * 100, 1)
            recovery_pct   = round((recovery_avg - baseline_avg) / max(0.01, baseline_avg) * 100, 1)
            total_lost     = round((baseline_avg - active_avg) * cri_dur, 0)

            st.info(
                f"**Crisis Phase** ({params['cri_start']}→{cri_end}):  "
                f"Avg revenue {impact_pct:+.1f}% vs baseline  |  "
                f"**Estimated revenue loss: €{total_lost:,.0f}**  |  "
                f"**Recovery Phase** (post day {cri_end}): "
                f"Avg revenue {recovery_pct:+.1f}% vs baseline"
            )

        # ── Row 1: Revenue (constant prices) + Nominal Revenue ───────────────
        c1, c2 = st.columns(2)
        fig_rev = px.line(df, x="Day", y="Revenue", color="Scenario",
                          title="Daily Revenue at Baseline Prices (€)  ↓ with inflation & disruption",
                          color_discrete_map={"Baseline":"#2E8B57","Crisis":"#DC143C"})
        fig_rev.update_layout(template="plotly_white",
                              yaxis_title="Revenue (€, constant baseline prices)")
        _add_crisis_lines(fig_rev)
        c1.plotly_chart(fig_rev, use_container_width=True, config=_PLOTLY_CFG)

        fig_nom = px.line(df, x="Day", y="NominalRevenue", color="Scenario",
                          title="Nominal Revenue (€, inflated prices)  — store cash-flow view",
                          color_discrete_map={"Baseline":"#2E8B57","Crisis":"#DC143C"})
        fig_nom.update_layout(template="plotly_white",
                              yaxis_title="Revenue (€, current prices)")
        _add_crisis_lines(fig_nom)
        c2.plotly_chart(fig_nom, use_container_width=True, config=_PLOTLY_CFG)

        with st.expander("📊 Revenue Analysis", expanded=True):
            st.markdown("**Constant-price Revenue** (demand signal — falls when consumers buy less)")
            _render_analysis(df, "Revenue", params, prefix="€", decimals=0,
                             higher_is_better=True)
            st.markdown("**Nominal Revenue** (store cash-flow — can rise during inflation even as volume falls)")
            _render_analysis(df, "NominalRevenue", params, prefix="€", decimals=0,
                             higher_is_better=True)

        # ── Row 2: Avg Price + Lost Sales ────────────────────────────────────
        c3, c4 = st.columns(2)
        fig_price = px.line(df, x="Day", y="AvgPrice", color="Scenario",
                            title="Avg Product Price (€)  — rises with inflation, drops at recovery",
                            color_discrete_map={"Baseline":"#2E8B57","Crisis":"#DC143C"})
        fig_price.update_layout(template="plotly_white",
                                yaxis_title="Mean price across catalogue (€)")
        _add_crisis_lines(fig_price)
        c3.plotly_chart(fig_price, use_container_width=True, config=_PLOTLY_CFG)

        fig_lost = px.bar(df, x="Day", y="LostSales", color="Scenario", barmode="group",
                          title="Lost Sales (€)",
                          color_discrete_map={"Baseline":"#87CEEB","Crisis":"#8B0000"})
        fig_lost.update_layout(template="plotly_white")
        _add_crisis_lines(fig_lost)
        c4.plotly_chart(fig_lost, use_container_width=True, config=_PLOTLY_CFG)

        with st.expander("📊 Price & Lost-Sales Analysis", expanded=True):
            st.markdown("**Average Product Price** — measures inflation pass-through and recovery speed")
            _render_analysis(df, "AvgPrice", params, prefix="€", decimals=3,
                             higher_is_better=False)
            st.markdown("**Lost Sales** — revenue foregone due to stockouts or price refusals")
            _render_analysis(df, "LostSales", params, prefix="€", decimals=1,
                             higher_is_better=False)

        # ── Row 3: Panic Level + Waste ────────────────────────────────────────
        c5, c6 = st.columns(2)
        fig_panic = px.line(df, x="Day", y="PanicLevel", color="Scenario",
                            title="Global Panic Level  (decays during recovery)",
                            color_discrete_map={"Baseline":"#2E8B57","Crisis":"#DC143C"})
        fig_panic.update_layout(template="plotly_white")
        _add_crisis_lines(fig_panic)
        c5.plotly_chart(fig_panic, use_container_width=True, config=_PLOTLY_CFG)

        fig_waste = px.bar(df, x="Day", y="Waste", color="Scenario", barmode="group",
                           title="Daily Waste (units)",
                           color_discrete_map={"Baseline":"#90EE90","Crisis":"#FF6347"})
        fig_waste.update_layout(template="plotly_white")
        _add_crisis_lines(fig_waste)
        c6.plotly_chart(fig_waste, use_container_width=True, config=_PLOTLY_CFG)

        with st.expander("📊 Panic & Waste Analysis", expanded=True):
            st.markdown("**Global Panic Level** (0–1 scale) — driven by stockouts, crowding, and media")
            _render_analysis(df, "PanicLevel", params, decimals=3, higher_is_better=False)
            st.markdown("**Daily Food Waste** — units expired or refused due to storage overflow")
            _render_analysis(df, "Waste", params, suffix=" units", decimals=1,
                             higher_is_better=False)

    with sub2:
        df_base = df[df["Scenario"] == "Baseline"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Daily Visitors",   f"{int(df_base['Consumers'].mean())}")
        m2.metric("Total Simulation Visits", f"{int(df_base['Consumers'].sum())}")
        m3.metric("Peak Day",             f"Day {df_base.loc[df_base['Consumers'].idxmax(), 'Day']}")
        fig_foot = px.bar(df_base, x="Day", y="Consumers", title="Daily Footfall (Baseline)",
                          color_discrete_sequence=["#4682B4"])
        fig_foot.update_layout(template="plotly_white")
        st.plotly_chart(fig_foot, use_container_width=True, config=_PLOTLY_CFG)
        with st.expander("📊 Footfall Analysis", expanded=True):
            st.markdown("**Daily Store Visitors** — footfall during the baseline (no-crisis) run")
            _render_analysis(df, "Consumers", params, suffix=" visitors", decimals=0,
                             higher_is_better=True)

    with sub3:
        if df_stock is None or df_stock.empty:
            st.info("No stock data available.")
        else:
            # ── Info panel ────────────────────────────────────────────────
            with st.expander("ℹ️  How to read this chart", expanded=False):
                st.markdown("""
**This chart shows the complete inventory lifecycle for one product across three panels:**

| Panel | What it shows |
|---|---|
| 📦 **Shelf Stock** | Units currently visible and purchasable on the store shelf.  Falls as customers buy. Jumps up when staff move stock from the storage room. |
| 🏪 **Storage Stock** | Units in the backroom.  Falls when the shelf is restocked.  Jumps up when a delivery arrives. |
| 🚚 **Orders & Deliveries** | Orange bars = orders placed to the central warehouse.  Blue bars = deliveries received. |

**Threshold lines**

- 🟠 **Orange dashed line (Shelf)** — when shelf falls below **30 % of shelf capacity**, staff immediately move stock from storage to the shelf.
- 🔴 **Red dashed line (Storage)** — when storage falls below the configured **Reorder Point**, a new order is placed automatically.  The delivery arrives after the configured **Lead Time** (days).

**Markers**

- ❌ **Red × on shelf panel** — units removed from the shelf because they reached their **expiry date** (counted as food waste).
- 🟡 **Yellow ◆ on shelf panel** — units sold at **50 % discount** because they were within 2 days of expiry.
- ⏳ **Dotted line on storage panel** — units already on order but not yet delivered (**pipeline stock**).

> **Tip:** Hover over any point to see exact values.  Use the scenario toggle below to compare Baseline vs Crisis side by side.
                """)

            all_prods = sorted(df_stock["Product"].unique())
            sel_prod  = st.selectbox("Select Product:", all_prods, key="demo_prod_sel")

            # Infer capacity limits from the data for threshold lines
            prod_rows   = df_stock[df_stock["Product"] == sel_prod]
            max_shelf   = int(prod_rows["Shelf"].max())   if not prod_rows.empty else 20
            max_storage = int(prod_rows["Storage"].max()) if not prod_rows.empty else 50
            # Fallback: ensure sensible minimums
            max_shelf   = max(max_shelf,   10)
            max_storage = max(max_storage, 10)

            shelf_trigger   = max_shelf   * 0.30
            storage_trigger = max_storage * params["reorder"]

            # Waste data for this product
            df_waste_local = st.session_state.sim_waste

            def draw_scm_v2(scenario: str, accent: str) -> go.Figure:
                from plotly.subplots import make_subplots

                d = df_stock[
                    (df_stock["Product"]  == sel_prod) &
                    (df_stock["Scenario"] == scenario)
                ].copy()
                if d.empty:
                    return go.Figure()

                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.10,
                    subplot_titles=(
                        "📦 Shelf Stock (units available to customers)",
                        "🏪 Storage Stock (backroom inventory)",
                        "🚚 Orders Placed & Deliveries Received",
                    ),
                    row_heights=[0.38, 0.35, 0.27],
                )

                # ── ROW 1: Shelf ──────────────────────────────────────────
                fig.add_trace(go.Scatter(
                    x=d["Day"], y=d["Shelf"],
                    mode="lines", fill="tozeroy",
                    name="Shelf Stock",
                    line=dict(color=accent, width=2.5),
                    fillcolor=f"rgba({int(accent[1:3],16)},{int(accent[3:5],16)},{int(accent[5:7],16)},0.18)",
                    hovertemplate=(
                        "<b>Day %{x}</b><br>"
                        "Shelf: <b>%{y} units</b><br>"
                        "<extra></extra>"
                    ),
                ), row=1, col=1)

                # Shelf refill threshold
                fig.add_hline(
                    y=shelf_trigger,
                    line_dash="dash", line_color="orange", line_width=1.5,
                    annotation_text=f"Shelf refill trigger ({int(shelf_trigger)} units)",
                    annotation_position="top right",
                    annotation_font_size=10,
                    row=1, col=1,
                )

                # Waste events (expiry) on shelf panel
                if df_waste_local is not None and not df_waste_local.empty:
                    w = df_waste_local[
                        (df_waste_local["Product"]  == sel_prod) &
                        (df_waste_local["Scenario"] == scenario) &
                        (df_waste_local["Reason"]   == "Expiry")
                    ]
                    if not w.empty:
                        fig.add_trace(go.Scatter(
                            x=w["Day"],
                            y=[shelf_trigger * 0.5] * len(w),
                            mode="markers",
                            name="❌ Expiry Waste",
                            marker=dict(color="red", size=11, symbol="x",
                                        line=dict(width=2, color="darkred")),
                            customdata=w["Quantity"].values,
                            hovertemplate=(
                                "<b>Day %{x}</b><br>"
                                "⚠️ Expired & removed: <b>%{customdata} units</b><br>"
                                "(counted as food waste)<extra></extra>"
                            ),
                        ), row=1, col=1)

                # Near-expiry sold (50 % discount)
                if "NearExpiry" in d.columns:
                    ne = d[d["NearExpiry"] > 0]
                    if not ne.empty:
                        fig.add_trace(go.Scatter(
                            x=ne["Day"], y=ne["Shelf"],
                            mode="markers",
                            name="🏷️ Near-Expiry Sold (−50 %)",
                            marker=dict(color="#f39c12", size=9, symbol="diamond",
                                        line=dict(width=1, color="#d35400")),
                            customdata=ne["NearExpiry"].values,
                            hovertemplate=(
                                "<b>Day %{x}</b><br>"
                                "Near-expiry units sold at 50 %% off: <b>%{customdata}</b>"
                                "<extra></extra>"
                            ),
                        ), row=1, col=1)

                # ── ROW 2: Storage ────────────────────────────────────────
                fig.add_trace(go.Scatter(
                    x=d["Day"], y=d["Storage"],
                    mode="lines", fill="tozeroy",
                    name="Storage Stock",
                    line=dict(color="#2980b9", width=2.5),
                    fillcolor="rgba(41,128,185,0.15)",
                    hovertemplate=(
                        "<b>Day %{x}</b><br>"
                        "Storage: <b>%{y} units</b><br>"
                        "<extra></extra>"
                    ),
                ), row=2, col=1)

                # Pipeline (pending orders not yet delivered)
                if "Pending" in d.columns:
                    fig.add_trace(go.Scatter(
                        x=d["Day"], y=d["Pending"],
                        mode="lines",
                        name="⏳ On Order (pipeline)",
                        line=dict(color="#8e44ad", width=1.5, dash="dot"),
                        hovertemplate=(
                            "<b>Day %{x}</b><br>"
                            "Units on order (not yet delivered): <b>%{y}</b>"
                            "<extra></extra>"
                        ),
                    ), row=2, col=1)

                # Storage reorder threshold
                fig.add_hline(
                    y=storage_trigger,
                    line_dash="dash", line_color="red", line_width=1.5,
                    annotation_text=f"Reorder trigger ({int(params['reorder']*100)} % = {int(storage_trigger)} units)",
                    annotation_position="top right",
                    annotation_font_size=10,
                    row=2, col=1,
                )

                # ── ROW 3: Orders & Deliveries ────────────────────────────
                if df_log is not None and not df_log.empty:
                    dl = df_log[
                        (df_log["Product"]  == sel_prod) &
                        (df_log["Scenario"] == scenario)
                    ]
                    ords = dl[dl["Action"] == "Order"]
                    devs = dl[dl["Action"] == "Delivery"]

                    if not ords.empty:
                        fig.add_trace(go.Bar(
                            x=ords["Day"], y=ords["Quantity"],
                            name="🛒 Order Placed",
                            marker_color="#e67e22",
                            hovertemplate=(
                                "<b>Day %{x}</b><br>"
                                "Order placed: <b>%{y} units</b><br>"
                                f"(delivery in {params['lead']} day(s))"
                                "<extra></extra>"
                            ),
                        ), row=3, col=1)

                    if not devs.empty:
                        fig.add_trace(go.Bar(
                            x=devs["Day"], y=devs["Quantity"],
                            name="📬 Delivery Received",
                            marker_color="#2980b9",
                            hovertemplate=(
                                "<b>Day %{x}</b><br>"
                                "Delivered: <b>%{y} units</b><br>"
                                "<extra></extra>"
                            ),
                        ), row=3, col=1)

                        # Highlight refused deliveries if present
                        refused = devs[devs.get("Refused", pd.Series(0, index=devs.index)) > 0] \
                            if "Refused" in devs.columns else pd.DataFrame()
                        if not refused.empty:
                            fig.add_trace(go.Bar(
                                x=refused["Day"], y=refused["Refused"],
                                name="🚫 Refused (storage full)",
                                marker_color="#e74c3c",
                                hovertemplate=(
                                    "<b>Day %{x}</b><br>"
                                    "Refused (storage full): <b>%{y} units</b>"
                                    "<extra></extra>"
                                ),
                            ), row=3, col=1)

                # ── Layout ────────────────────────────────────────────────
                fig.update_layout(
                    title=dict(
                        text=f"<b>{scenario} Scenario — {sel_prod}</b>",
                        font=dict(size=15),
                        y=0.98,
                    ),
                    template="plotly_white",
                    height=780,
                    barmode="group",
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="top",   y=-0.09,
                        xanchor="center", x=0.5,
                        font=dict(size=11),
                        tracegroupgap=4,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#dee2e6",
                        borderwidth=1,
                    ),
                    margin=dict(t=80, b=180, l=60, r=20),
                )
                fig.update_yaxes(title_text="Units on Shelf",   row=1, col=1, rangemode="tozero")
                fig.update_yaxes(title_text="Units in Storage", row=2, col=1, rangemode="tozero")
                fig.update_yaxes(title_text="Units",            row=3, col=1, rangemode="tozero")
                fig.update_xaxes(title_text="Day",              row=3, col=1)
                # Add crisis start line if applicable
                if scenario == "Crisis" and params.get("cri_start"):
                    for r in [1, 2, 3]:
                        fig.add_vline(
                            x=params["cri_start"],
                            line_dash="dot", line_color="red", line_width=1,
                            row=r, col=1,
                        )
                return fig

            # Scenario tabs
            sc_tab_b, sc_tab_c = st.tabs(["🟢 Baseline", "🔴 Crisis"])
            with sc_tab_b:
                st.plotly_chart(draw_scm_v2("Baseline", "#27ae60"), use_container_width=True, config=_PLOTLY_CFG)
            with sc_tab_c:
                st.plotly_chart(draw_scm_v2("Crisis", "#c0392b"), use_container_width=True, config=_PLOTLY_CFG)

            # Quick summary table below the chart
            st.markdown("##### 📋 Product Summary")
            for sc in ["Baseline", "Crisis"]:
                d_sum = df_stock[
                    (df_stock["Product"] == sel_prod) & (df_stock["Scenario"] == sc)
                ]
                if d_sum.empty:
                    continue
                total_waste = (
                    df_waste_local[
                        (df_waste_local["Product"] == sel_prod) &
                        (df_waste_local["Scenario"] == sc)
                    ]["Quantity"].sum()
                    if df_waste_local is not None and not df_waste_local.empty else 0
                )
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric(f"[{sc}] Total Revenue",  f"€{d_sum['Revenue'].sum():,.2f}")
                c2.metric("Units Sold",              f"{int(d_sum['Sales'].sum()):,}")
                c3.metric("Lost Sales",              f"{int(d_sum['LostSales'].sum()):,}")
                c4.metric("Avg Shelf Level",         f"{d_sum['Shelf'].mean():.1f}")
                c5.metric("Total Waste",             f"{int(total_waste):,} units")

    with sub4:
        if model_cris is None:
            st.info("Run the simulation first.")
        else:
            total_rev  = df[df["Scenario"] == "Crisis"]["Revenue"].sum()
            lost_stock = model_cris.loss_reasons.get("Stockout", 0)
            lost_price = model_cris.loss_reasons.get("Price",    0)

            fig_sankey = go.Figure(data=[go.Sankey(
                textfont=dict(size=13, color="black"),
                node=dict(
                    pad=20, thickness=25,
                    label=["Potential Revenue", "Realised Revenue",
                           "Lost: Out of Stock", "Lost: Price Too High"],
                    color=["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"],
                ),
                link=dict(
                    source=[0, 0, 0], target=[1, 2, 3],
                    value=[max(0, total_rev), max(0, lost_stock), max(0, lost_price)],
                ),
            )])
            fig_sankey.update_layout(title="Revenue Flow — Crisis Scenario", height=400)
            st.plotly_chart(fig_sankey, use_container_width=True, config=_PLOTLY_CFG)

            st.caption(
                "The Sankey diagram shows how potential revenue was split between "
                "realised sales, stockout losses, and price-driven refusals."
            )

    with sub5:
        st.markdown(
            "**Behavioural Drift** — how agent preferences evolve over the simulation "
            "as a result of repeated purchase experience. Each archetype has a different "
            "learning rule (see sidebar for details)."
        )
        df_drift = st.session_state.get("sim_pref_drift")
        if df_drift is None or df_drift.empty:
            st.info("Run the simulation first to see preference drift.")
        else:
            ARCHETYPE_COLORS_DRIFT = {
                "price_champion":   "#e74c3c",
                "green_buyer":      "#27ae60",
                "health_optimizer": "#2980b9",
                "habitual_buyer":   "#8e44ad",
            }
            scenarios  = df_drift["Scenario"].unique().tolist()
            sel_sc_dr  = st.selectbox("Scenario:", scenarios, key="drift_scenario")
            df_d       = df_drift[df_drift["Scenario"] == sel_sc_dr]

            metrics = [
                ("MeanPriceSensitivity", "Mean Price Sensitivity"),
                ("MeanOrganicPref",      "Mean Organic Preference"),
                ("MeanPreferredFat",     "Mean Preferred Fat %"),
            ]
            col_m1, col_m2 = st.columns(2)
            col_m3, _      = st.columns(2)
            for (metric, label), col in zip(metrics, [col_m1, col_m2, col_m3]):
                fig_d = go.Figure()
                for arch in df_d["Archetype"].unique():
                    sub_d = df_d[df_d["Archetype"] == arch]
                    fig_d.add_trace(go.Scatter(
                        x=sub_d["Day"], y=sub_d[metric],
                        name=arch, mode="lines",
                        line=dict(color=ARCHETYPE_COLORS_DRIFT.get(arch, "#999")),
                    ))
                fig_d.update_layout(
                    title=label, template="plotly_white",
                    xaxis_title="Day", yaxis_title=label,
                    legend=dict(orientation="h", y=-0.3),
                )
                col.plotly_chart(fig_d, use_container_width=True, config=_PLOTLY_CFG)

            st.markdown(
                "**How to read this:** Each line represents the mean preference value "
                "across all agents of that archetype. Convergence toward a new level "
                "signals that the population is collectively learning from experience. "
                "Divergence between scenarios (Baseline vs Crisis) shows how the crisis "
                "reshapes consumer behaviour over time."
            )

    with sub6:
        df_drift6 = st.session_state.get("sim_pref_drift")
        if df_drift6 is None or df_drift6.empty:
            st.info("Run the simulation first to see per-buyer-type results.")
        else:
            _ARCH_COLORS6 = {
                "price_champion":   "#e74c3c",
                "green_buyer":      "#27ae60",
                "health_optimizer": "#2980b9",
                "habitual_buyer":   "#8e44ad",
            }
            _ARCH_LABELS6 = {
                "price_champion":   "💸 Price Champion",
                "green_buyer":      "🌿 Green Buyer",
                "health_optimizer": "💪 Health Optimizer",
                "habitual_buyer":   "🔁 Habitual Buyer",
            }
            _ARCH_ORDER = ["price_champion", "green_buyer", "health_optimizer", "habitual_buyer"]
            _has_beh = "BudgetExhaustionRate" in df_drift6.columns

            # ── Summary scorecards ────────────────────────────────────────────────
            st.markdown("### Crisis impact by buyer type")
            st.caption(
                "Each card shows the **average across all simulation days** for that archetype. "
                "The Δ arrow compares Crisis vs Baseline — red = worsened, green = improved."
            )
            _sc_cols = st.columns(4)
            for _col, _arch in zip(_sc_cols, _ARCH_ORDER):
                _base = df_drift6[(df_drift6["Archetype"] == _arch) & (df_drift6["Scenario"] == "Baseline")]
                _cris = df_drift6[(df_drift6["Archetype"] == _arch) & (df_drift6["Scenario"] == "Crisis")]
                with _col:
                    st.markdown(
                        f"<div style='border-left:4px solid {_ARCH_COLORS6[_arch]};padding:8px 12px;"
                        f"background:#f8f9fa;border-radius:4px;margin-bottom:8px;color:#1a2035'>"
                        f"<b>{_ARCH_LABELS6[_arch]}</b></div>",
                        unsafe_allow_html=True,
                    )
                    if _has_beh and not _base.empty and not _cris.empty:
                        for _metric, _label, _higher_bad in [
                            ("MeanFulfillment",      "Basket Fulfillment",    False),
                            ("BudgetExhaustionRate", "Budget Exhausted",      True),
                            ("MeanPanicLevel",       "Panic Level",           True),
                            ("MeanFIES",             "Food Insecurity (FIES)", True),
                        ]:
                            if _metric not in _base.columns:
                                continue
                            _b_val = _base[_metric].mean()
                            _c_val = _cris[_metric].mean()
                            _delta = _c_val - _b_val
                            _arrow = "↑" if _delta > 0 else "↓"
                            _color = ("#c0392b" if (_delta > 0) == _higher_bad else "#27ae60") if abs(_delta) > 0.005 else "#666"
                            st.markdown(
                                f"<div style='font-size:12px;margin:4px 0;color:#1a2035'>"
                                f"<span style='color:#555'>{_label}</span><br>"
                                f"<b style='font-size:15px'>{_b_val:.1%}</b>"
                                f"&nbsp;<span style='color:{_color};font-weight:700'>"
                                f"{_arrow} {abs(_delta):.1%}</span></div>",
                                unsafe_allow_html=True,
                            )
                    elif not _base.empty:
                        st.caption("Run both scenarios for comparison.")

            st.divider()

            if _has_beh:
                # ── Grouped bar: Baseline vs Crisis for each behavioral metric ──────
                st.markdown("### Behavioral outcomes — Baseline vs Crisis")
                _beh_metrics = [
                    ("MeanFulfillment",      "Basket Fulfillment (%)",    "Higher is better"),
                    ("BudgetExhaustionRate", "Budget Exhaustion Rate (%)", "Lower is better"),
                    ("MeanPanicLevel",       "Mean Panic Level",          "Lower is better"),
                    ("MeanFIES",             "Food Insecurity Score (0–4)", "Lower is better"),
                ]
                _bar_c1, _bar_c2 = st.columns(2)
                for (_bm, _bl, _note), _bcol in zip(_beh_metrics, [_bar_c1, _bar_c2, _bar_c1, _bar_c2]):
                    if _bm not in df_drift6.columns:
                        continue
                    _rows = []
                    for _arch in _ARCH_ORDER:
                        for _sc in ["Baseline", "Crisis"]:
                            _sub = df_drift6[(df_drift6["Archetype"] == _arch) & (df_drift6["Scenario"] == _sc)]
                            if not _sub.empty:
                                _rows.append({"Archetype": _ARCH_LABELS6.get(_arch, _arch),
                                              "Scenario": _sc, _bl: _sub[_bm].mean()})
                    if not _rows:
                        continue
                    _df_bar = pd.DataFrame(_rows)
                    _fig_bar = px.bar(
                        _df_bar, x="Archetype", y=_bl, color="Scenario", barmode="group",
                        color_discrete_map={"Baseline": "#4a90d9", "Crisis": "#e74c3c"},
                        title=f"{_bl}<br><sup style='color:#888'>{_note}</sup>",
                    )
                    _fig_bar.update_layout(
                        template="plotly_white", height=320,
                        legend=dict(orientation="h", y=-0.3),
                        xaxis_title="", margin=dict(t=60, b=80),
                    )
                    _bcol.plotly_chart(_fig_bar, use_container_width=True, config=_PLOTLY_CFG)

                st.divider()

                # ── Panic & FIES trajectories over time ───────────────────────────
                st.markdown("### Daily trajectories by archetype")
                _traj_scenario = st.selectbox("Scenario for trajectory charts:", ["Baseline", "Crisis"],
                                               key="buyer_traj_sc")
                _df_traj = df_drift6[df_drift6["Scenario"] == _traj_scenario]
                _t1, _t2 = st.columns(2)
                for (_tm, _tl), _tcol in [
                    (("MeanPanicLevel", "Panic Level"),      _t1),
                    (("MeanFulfillment", "Basket Fulfillment"), _t2),
                    (("BudgetExhaustionRate", "Budget Exhaustion Rate"), _t1),
                    (("MeanFIES", "Food Insecurity Score"), _t2),
                ]:
                    if _tm not in _df_traj.columns:
                        continue
                    _fig_t = go.Figure()
                    for _arch in _ARCH_ORDER:
                        _sub_t = _df_traj[_df_traj["Archetype"] == _arch]
                        if _sub_t.empty:
                            continue
                        _fig_t.add_trace(go.Scatter(
                            x=_sub_t["Day"], y=_sub_t[_tm],
                            name=_ARCH_LABELS6.get(_arch, _arch), mode="lines",
                            line=dict(color=_ARCH_COLORS6.get(_arch, "#999"), width=2),
                        ))
                    if params.get("cri_start") and _traj_scenario == "Crisis":
                        _fig_t.add_vline(x=params["cri_start"], line_dash="dot",
                                         line_color="orange", annotation_text="Crisis")
                    _fig_t.update_layout(
                        title=_tl, template="plotly_white", height=300,
                        xaxis_title="Day", legend=dict(orientation="h", y=-0.35),
                        margin=dict(t=50, b=80),
                    )
                    _tcol.plotly_chart(_fig_t, use_container_width=True, config=_PLOTLY_CFG)

                st.divider()

                # ── Transition delta table ────────────────────────────────────────
                st.markdown("### Who changed most? (Crisis − Baseline averages)")
                _delta_rows = []
                for _arch in _ARCH_ORDER:
                    _b = df_drift6[(df_drift6["Archetype"] == _arch) & (df_drift6["Scenario"] == "Baseline")]
                    _c = df_drift6[(df_drift6["Archetype"] == _arch) & (df_drift6["Scenario"] == "Crisis")]
                    if _b.empty or _c.empty:
                        continue
                    _row = {"Buyer Type": _ARCH_LABELS6.get(_arch, _arch)}
                    for _m, _lbl in [
                        ("MeanPriceSensitivity", "Price Sens. Δ"),
                        ("MeanOrganicPref",      "Organic Pref. Δ"),
                        ("MeanSubTolerance",     "Sub. Tolerance Δ"),
                        ("MeanFulfillment",      "Fulfillment Δ"),
                        ("BudgetExhaustionRate", "Budget Exh. Δ"),
                        ("MeanPanicLevel",       "Panic Δ"),
                        ("MeanFIES",             "FIES Δ"),
                    ]:
                        if _m in _b.columns and _m in _c.columns:
                            _row[_lbl] = round(_c[_m].mean() - _b[_m].mean(), 4)
                    _delta_rows.append(_row)

                if _delta_rows:
                    _df_delta = pd.DataFrame(_delta_rows).set_index("Buyer Type")

                    def _color_delta(val):
                        if not isinstance(val, (int, float)):
                            return ""
                        color = "#c0392b" if val > 0.005 else ("#27ae60" if val < -0.005 else "#666")
                        return f"color: {color}; font-weight: bold"

                    st.dataframe(
                        _df_delta.style.map(_color_delta),
                        width=900,
                    )
                    _biggest = _df_delta.abs().sum(axis=1).idxmax()
                    st.caption(
                        f"**{_biggest}** showed the largest overall behavioral shift across "
                        f"all metrics (sum of absolute deltas = "
                        f"{_df_delta.abs().sum(axis=1).max():.3f})."
                    )


# ===========================================================================
# 6. MONTE CARLO RUNNER
# ===========================================================================

def _run_mc_batch(n_runs: int, days: int, params: dict,
                  is_crisis: bool, ai_recs=None,
                  progress_label: str = "Running…") -> tuple[pd.DataFrame, dict]:
    """
    Run n_runs independent Monte Carlo replications and return:
      (aggregate_dataframe, product_stats_dict)
    """
    agg_rows  = []
    prod_stats: dict[str, dict] = {}

    bar = st.progress(0, text=progress_label)
    total = n_runs * days

    for run_id in range(n_runs):
        seed  = 1000 + run_id
        model = _make_model(params, is_crisis=is_crisis, seed=seed, ai_recs=ai_recs)

        for day in range(1, days + 1):
            model.step()
            agg, _ = _collect_model_day(model, day, "Crisis" if is_crisis else "Baseline",
                                         collect_products=False)
            agg["Run"] = run_id
            agg_rows.append(agg)

            for a in _product_agents(model):
                if a.name not in prod_stats:
                    prod_stats[a.name] = {
                        "Category":    a.category,
                        "ShelfLife":   a.max_shelf_life,
                        "StorageCap":  a.max_storage_capacity,
                        "AggSales":    0, "AggLost": 0, "AggWaste": 0,
                    }
                prod_stats[a.name]["AggSales"] += a.daily_sales
                prod_stats[a.name]["AggLost"]  += a.daily_lost_sales
                prod_stats[a.name]["AggWaste"] += a.daily_waste

            step_done = run_id * days + day
            if step_done % max(1, total // 50) == 0:
                bar.progress(step_done / total, text=progress_label)

    bar.progress(1.0, text="Complete ✓")
    time.sleep(0.3)
    bar.empty()

    return pd.DataFrame(agg_rows), prod_stats


# ===========================================================================
# 7. TAB: SCIENTIFIC ANALYSIS
# ===========================================================================

def render_science_tab(params: dict):
    st.header(_t("header_science"))

    if st.session_state.config_data is None:
        st.warning("⚠️ Load data in the **🏠 Data & Population** tab first.")
        return

    n_runs = params["mc_runs"]
    days   = params["days"]

    # ---- STAGE 0: Run baseline ----
    if st.session_state.mc_stage == 0:
        st.markdown(
            '<div class="step-card"><h3>Step 1 — Establish Baseline</h3>'
            "<p>Simulate business-as-usual (no crisis) across "
            f"{n_runs} independent runs to characterise normal performance "
            "and identify inventory inefficiencies.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button(_t("btn_run_baseline"), type="primary"):
            df_raw, prod_stats = _run_mc_batch(n_runs, days, params, is_crisis=False,
                                                progress_label="Simulating Baseline…")
            # Compute AI storage recommendations
            total_periods = n_runs * days
            new_recs = {}
            for p_name, data in prod_stats.items():
                avg_waste = data["AggWaste"] / total_periods
                avg_lost  = data["AggLost"]  / total_periods
                avg_sales = data["AggSales"] / total_periods
                curr_cap  = data["StorageCap"]
                rec_cap   = curr_cap
                if avg_waste > 0.5:
                    rec_cap = max(10, int((avg_sales + avg_lost) * data["ShelfLife"] * 0.55))
                elif avg_lost > 0.5:
                    rec_cap = min(300, max(int(curr_cap * 1.25),
                                          int((avg_sales + avg_lost) * params["lead"] * 4)))
                new_recs[p_name] = rec_cap

            st.session_state.data_base_raw  = df_raw
            st.session_state.ai_recs        = new_recs
            st.session_state.prod_stats_raw = prod_stats
            st.session_state.mc_stage       = 1
            st.rerun()

    # ---- STAGE 1: Review & optimise ----
    elif st.session_state.mc_stage == 1:
        st.markdown(
            '<div class="step-card"><h3>Step 2 — Review AI Suggestions</h3>'
            "<p>Inspect baseline performance and choose whether to apply "
            "AI-recommended storage capacity adjustments.</p></div>",
            unsafe_allow_html=True,
        )
        df_raw = st.session_state.data_base_raw
        _plot_ci_band(df_raw, "Daily Revenue — Baseline (Raw) [95 % CI]",
                      color="gray")

        st.divider()
        st.markdown("#### 🤖 AI Storage Capacity Recommendations")
        recs_df = pd.DataFrame([
            {"Product": p, "Current Capacity": st.session_state.prod_stats_raw[p]["StorageCap"],
             "Recommended Capacity": c}
            for p, c in st.session_state.ai_recs.items()
        ])
        st.dataframe(recs_df, use_container_width=True)

        col1, col2 = st.columns(2)
        if col1.button("✅ Accept AI Recommendations & Re-run Baseline"):
            df_opt, _ = _run_mc_batch(n_runs, days, params, is_crisis=False,
                                       ai_recs=st.session_state.ai_recs,
                                       progress_label="Simulating Optimised Baseline…")
            st.session_state.data_base_opt  = df_opt
            st.session_state.active_baseline = "Baseline (Optimised)"
            st.session_state.mc_stage       = 2
            st.rerun()
        if col2.button("⏩ Skip — Use Raw Baseline"):
            st.session_state.data_base_opt   = None
            st.session_state.active_baseline = "Baseline (Raw)"
            st.session_state.mc_stage        = 2
            st.rerun()

    # ---- STAGE 2: Baseline comparison ----
    elif st.session_state.mc_stage == 2:
        st.markdown(
            '<div class="step-card"><h3>Step 3 — Baseline Comparison</h3>'
            "<p>Compare Raw vs Optimised baseline, then run the Crisis scenario.</p></div>",
            unsafe_allow_html=True,
        )

        if st.session_state.data_base_opt is not None:
            df_r = st.session_state.data_base_raw
            df_o = st.session_state.data_base_opt
            r_mean, r_min = df_r["Revenue"].mean(), df_r.groupby("Run")["Revenue"].min().mean()
            o_mean, o_min = df_o["Revenue"].mean(), df_o.groupby("Run")["Revenue"].min().mean()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Daily Rev (Raw)",  f"€{r_mean:,.1f}")
            c2.metric("Avg Daily Rev (Opt)",  f"€{o_mean:,.1f}",
                      f"{(o_mean-r_mean)/max(r_mean,0.01)*100:.1f}%")
            c3.metric("Avg Min Rev (Raw)",    f"€{r_min:,.1f}")
            c4.metric("Avg Min Rev (Opt)",    f"€{o_min:,.1f}",
                      f"{(o_min-r_min)/max(r_min,0.01)*100:.1f}%")

            r_waste = df_r["Waste"].sum() / n_runs
            o_waste = df_o["Waste"].sum() / n_runs
            r_lost  = df_r["LostSales"].sum() / n_runs
            o_lost  = df_o["LostSales"].sum() / n_runs

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name="Waste", x=["Raw","Optimised"], y=[r_waste, o_waste],
                                     marker_color="salmon",
                                     text=[f"{r_waste:.0f}", f"{o_waste:.0f}"], textposition="auto"))
            fig_cmp.add_trace(go.Bar(name="Lost Sales", x=["Raw","Optimised"], y=[r_lost, o_lost],
                                     marker_color="orange",
                                     text=[f"{r_lost:.0f}", f"{o_lost:.0f}"], textposition="auto"))
            fig_cmp.update_layout(barmode="group", title="Baseline Efficiency Comparison",
                                  template="plotly_white", yaxis_title="Units (avg per run)")
            st.plotly_chart(fig_cmp, use_container_width=True, config=_PLOTLY_CFG)
        else:
            st.info("Using Raw Baseline (optimisation skipped).")

        st.divider()
        st.markdown(f"✅ Ready to test **{st.session_state.active_baseline}** vs **Crisis**.")
        if st.button(_t("btn_run_crisis"), type="primary"):
            recs = st.session_state.ai_recs if st.session_state.data_base_opt is not None else None
            df_cri, _ = _run_mc_batch(n_runs, days, params, is_crisis=True,
                                       ai_recs=recs, progress_label="Simulating Crisis…")
            st.session_state.data_crisis = df_cri
            st.session_state.mc_stage    = 3
            st.rerun()

    # ---- STAGE 3: Final impact analysis ----
    elif st.session_state.mc_stage == 3:
        st.markdown(
            '<div class="step-card"><h3>Step 4 — Impact Analysis</h3></div>',
            unsafe_allow_html=True,
        )

        label_base = st.session_state.active_baseline
        df_base = (st.session_state.data_base_opt
                   if st.session_state.data_base_opt is not None
                   else st.session_state.data_base_raw)
        df_cri  = st.session_state.data_crisis

        df_base = df_base.copy(); df_base["Scenario"] = label_base
        df_cri  = df_cri.copy();  df_cri["Scenario"]  = "Crisis"
        df_full = pd.concat([df_base, df_cri], ignore_index=True)

        # Confidence interval plot
        _plot_ci_dual(df_base, df_cri, label_base)
        with st.expander("📊 Revenue Trend Analysis (CI)", expanded=True):
            st.markdown("**Daily Revenue** — baseline vs crisis across all Monte Carlo runs")
            _render_analysis(df_full, "Revenue", params, prefix="€", decimals=0,
                             higher_is_better=True, baseline_label=label_base, crisis_label="Crisis")

        # Key metrics
        rev_base = df_base["Revenue"].sum() / n_runs
        rev_cri  = df_cri["Revenue"].sum()  / n_runs
        diff_pct = (rev_cri - rev_base) / max(rev_base, 0.01) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Total Revenue ({label_base})", f"€{rev_base:,.0f}")
        c2.metric("Total Revenue (Crisis)",         f"€{rev_cri:,.0f}",
                  f"{diff_pct:.1f}%")
        waste_drop = (df_cri["Waste"].sum() - df_base["Waste"].sum()) / n_runs
        c3.metric("Extra Waste (Crisis)",           f"{waste_drop:+.0f} units/run")

        # Distribution violin
        fig_vio = px.violin(df_full, x="Scenario", y="Revenue", color="Scenario",
                            box=True, points="outliers",
                            title="Daily Revenue Distribution",
                            color_discrete_map={label_base:"#2E8B57","Crisis":"#DC143C"})
        fig_vio.update_layout(template="plotly_white")
        st.plotly_chart(fig_vio, use_container_width=True, config=_PLOTLY_CFG)
        with st.expander("📊 Revenue Distribution Analysis", expanded=True):
            st.markdown("**Revenue Distribution** — spread and central tendency across all runs and days")
            _render_analysis(df_full, "Revenue", params, prefix="€", decimals=0,
                             higher_is_better=True, baseline_label=label_base, crisis_label="Crisis")

        # Correlation heatmaps
        st.subheader("📊 Correlation Analysis — Systemic Coupling")
        c_base = df_base[["Revenue","Waste","LostSales"]].corr()
        c_cri  = df_cri[["Revenue","Waste","LostSales"]].corr()

        fig_heat, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig_heat.patch.set_facecolor("white")
        sns.heatmap(c_base, annot=True, cmap="Greens", ax=ax1, vmin=-1, vmax=1,
                    fmt=".2f", annot_kws={"color": "black"})
        ax1.set_title(label_base, color="black")
        ax1.set_facecolor("white")
        sns.heatmap(c_cri,  annot=True, cmap="Reds",   ax=ax2, vmin=-1, vmax=1,
                    fmt=".2f", annot_kws={"color": "black"})
        ax2.set_title("Crisis", color="black")
        ax2.set_facecolor("white")
        fig_heat.tight_layout()
        st.pyplot(fig_heat)
        plt.close(fig_heat)

        # Downloads
        st.divider()
        csv_bytes = df_full.to_csv(index=False).encode("utf-8")
        c_dl1, c_dl2, c_dl3 = st.columns(3)
        c_dl1.download_button("📥 Download MC Data (CSV)", csv_bytes,
                              "monte_carlo_results.csv", "text/csv")
        if c_dl2.button("📄 Generate PDF Report"):
            try:
                pdf_bytes = _make_pdf_report(df_full, label_base, n_runs)
                c_dl2.download_button("📥 Download PDF", pdf_bytes,
                                      "GROCERYsim_Report.pdf", "application/pdf")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

        if c_dl3.button("🔄 Reset Workflow"):
            for k in ["mc_stage","data_base_raw","data_base_opt","data_crisis",
                      "ai_recs","prod_stats_raw"]:
                st.session_state[k] = 0 if k == "mc_stage" else None
            st.rerun()


def _percentile_band(df: pd.DataFrame, col: str = "Revenue"):
    """Return a per-Day stats DataFrame with mean, median, p10, p25, p75, p90."""
    g = df.groupby("Day")[col]
    stats = g.agg(["mean"]).reset_index()
    stats["median"] = g.median().values
    stats["p10"]    = g.quantile(0.10).values
    stats["p25"]    = g.quantile(0.25).values
    stats["p75"]    = g.quantile(0.75).values
    stats["p90"]    = g.quantile(0.90).values
    return stats


def _band_traces(fig: go.Figure, stats: pd.DataFrame, name: str,
                 color_hex: str, show_iqr: bool = True):
    """Add p10–p90 outer band, optional p25–p75 IQR band, and median line."""
    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
    days_fwd = stats["Day"]
    days_rev = stats["Day"][::-1]

    # Outer band p10–p90
    fig.add_trace(go.Scatter(
        x=pd.concat([days_fwd, days_rev]),
        y=pd.concat([stats["p90"], stats["p10"][::-1]]),
        fill="toself",
        fillcolor=f"rgba({r},{g},{b},0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        legendgroup=name,
        name=f"{name} p10–p90",
        showlegend=True,
        hoverinfo="skip",
    ))
    # IQR band p25–p75
    if show_iqr:
        fig.add_trace(go.Scatter(
            x=pd.concat([days_fwd, days_rev]),
            y=pd.concat([stats["p75"], stats["p25"][::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.20)",
            line=dict(color="rgba(0,0,0,0)"),
            legendgroup=name,
            name=f"{name} IQR",
            showlegend=True,
            hoverinfo="skip",
        ))
    # Median line
    fig.add_trace(go.Scatter(
        x=stats["Day"], y=stats["median"],
        line=dict(color=color_hex, width=2.5),
        legendgroup=name,
        name=f"{name} median",
    ))


def _plot_ci_band(df: pd.DataFrame, title: str, color: str = "#44A1A0"):
    """Single-scenario revenue chart with p10/p25/p75/p90 percentile bands."""
    stats = _percentile_band(df)
    fig = go.Figure()
    _band_traces(fig, stats, "Revenue", color)
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Day",
        yaxis_title="Revenue (€)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)


def _plot_ci_dual(df_base: pd.DataFrame, df_cri: pd.DataFrame, label_base: str):
    """Baseline vs Crisis revenue chart with dual percentile confidence bands."""
    s_b = _percentile_band(df_base)
    s_c = _percentile_band(df_cri)

    fig = go.Figure()
    _band_traces(fig, s_b, label_base, "#44A1A0")   # teal  = baseline
    _band_traces(fig, s_c, "Crisis",   "#DC143C")   # red   = crisis

    fig.update_layout(
        title="Daily Revenue — Baseline vs Crisis  [p10 / IQR / p90 bands]",
        xaxis_title="Day",
        yaxis_title="Revenue (€)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)


# ===========================================================================
# 8. TAB: FOOD WASTE
# ===========================================================================

def render_waste_tab():
    st.header(_t("header_waste"))

    if st.session_state.sim_waste is None:
        st.info("Run the Interactive Demo simulation first to populate waste data.")
        return

    df = st.session_state.sim_waste
    if df.empty:
        st.success("No food waste recorded in this simulation run.")
        return

    scenarios = df["Scenario"].unique().tolist()
    sel_sc = st.selectbox("Scenario:", scenarios, key="waste_scenario")
    dfw = df[df["Scenario"] == sel_sc]

    total_w = dfw["Quantity"].sum()
    expiry  = dfw[dfw["Reason"] == "Expiry"]["Quantity"].sum()
    refused = dfw[dfw["Reason"] == "Refused Delivery"]["Quantity"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Waste (units)", f"{total_w:,}")
    c2.metric("Expiry Waste",        f"{expiry:,}", f"{expiry/max(total_w,1)*100:.0f}%")
    c3.metric("Refused Delivery Waste", f"{refused:,}", f"{refused/max(total_w,1)*100:.0f}%")

    col_a, col_b = st.columns(2)

    with col_a:
        by_cat = dfw.groupby("Category")["Quantity"].sum().reset_index()
        fig_cat = px.bar(by_cat, x="Category", y="Quantity",
                         title="Waste by Category",
                         color="Category",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_cat.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True, config=_PLOTLY_CFG)

    with col_b:
        by_reason = dfw.groupby("Reason")["Quantity"].sum().reset_index()
        fig_rea = px.pie(by_reason, names="Reason", values="Quantity",
                         title="Waste by Reason",
                         color_discrete_sequence=["#e74c3c","#e67e22","#f1c40f"])
        fig_rea.update_layout(template="plotly_white")
        st.plotly_chart(fig_rea, use_container_width=True, config=_PLOTLY_CFG)

    # Time series
    daily_waste = dfw.groupby("Day")["Quantity"].sum().reset_index()
    fig_time = px.area(daily_waste, x="Day", y="Quantity",
                       title="Daily Waste Over Simulation",
                       color_discrete_sequence=["#e74c3c"])
    fig_time.update_layout(template="plotly_white")
    st.plotly_chart(fig_time, use_container_width=True, config=_PLOTLY_CFG)
    with st.expander("📊 Waste Trend Analysis", expanded=True):
        st.markdown("**Daily Food Waste** — units discarded (expiry + refused deliveries) per day")
        _waste_ts = daily_waste.copy()
        _waste_ts["Scenario"] = sel_sc
        _render_analysis(_waste_ts, "Quantity", {}, suffix=" units", decimals=1, higher_is_better=False)

    # Heatmap: product × category waste
    waste_pivot = dfw.groupby(["Product","Category"])["Quantity"].sum().reset_index()
    top_waste   = waste_pivot.nlargest(20, "Quantity")
    fig_hm = px.bar(top_waste, x="Quantity", y="Product", color="Category",
                    orientation="h", title="Top 20 Products by Waste",
                    color_discrete_sequence=px.colors.qualitative.Set2)
    fig_hm.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_hm, use_container_width=True, config=_PLOTLY_CFG)

    # Download
    st.download_button("📥 Download Waste Data (CSV)",
                       dfw.to_csv(index=False).encode("utf-8"),
                       "food_waste_data.csv", "text/csv")


# ===========================================================================
# 9. TAB: PER-PRODUCT
# ===========================================================================

def render_product_tab():
    st.header(_t("header_product"))

    if st.session_state.sim_stock is None:
        st.info("Run the Interactive Demo simulation first.")
        return

    df_stock = st.session_state.sim_stock
    scenarios = sorted(df_stock["Scenario"].unique())
    prods     = sorted(df_stock["Product"].unique())

    c_sel1, c_sel2 = st.columns(2)
    sel_prod = c_sel1.selectbox("Product:", prods, key="pp_product")
    sel_sc   = c_sel2.selectbox("Scenario:", scenarios, key="pp_scenario")

    df = df_stock[(df_stock["Product"] == sel_prod) & (df_stock["Scenario"] == sel_sc)]

    if df.empty:
        st.warning("No data for this selection.")
        return

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Revenue",    f"€{df['Revenue'].sum():,.2f}")
    k2.metric("Units Sold",       f"{int(df['Sales'].sum()):,}")
    k3.metric("Lost Sales",       f"{int(df['LostSales'].sum()):,}")
    k4.metric("Total Waste",      f"{int(df['Waste'].sum()):,}")
    k5.metric("Avg Shelf Stock",  f"{df['Shelf'].mean():.1f}")

    # Stock trajectory
    fig_stock = go.Figure()
    fig_stock.add_trace(go.Scatter(x=df["Day"], y=df["Storage"], name="Storage",
                                   line=dict(color="#2980b9", width=2)))
    fig_stock.add_trace(go.Scatter(x=df["Day"], y=df["Shelf"], name="Shelf",
                                   line=dict(color="#27ae60", width=2, dash="dash")))
    fig_stock.add_trace(go.Scatter(x=df["Day"], y=df["Storage"] + df["Pending"],
                                   name="Total Pipeline",
                                   line=dict(color="#8e44ad", width=1, dash="dot")))
    fig_stock.update_layout(title=f"Stock Trajectory — {sel_prod}",
                             template="plotly_white", xaxis_title="Day", yaxis_title="Units")
    st.plotly_chart(fig_stock, use_container_width=True, config=_PLOTLY_CFG)
    with st.expander("📊 Stock Analysis", expanded=True):
        st.markdown("**Shelf Stock** — units available to customers each day")
        _render_analysis(df, "Shelf", {}, suffix=" units", decimals=1, higher_is_better=True)
        st.markdown("**Storage Stock** — backroom inventory each day")
        _render_analysis(df, "Storage", {}, suffix=" units", decimals=1, higher_is_better=True)

    col_r, col_w = st.columns(2)

    with col_r:
        fig_rev = px.bar(df, x="Day", y="Revenue",
                         title="Daily Revenue",
                         color_discrete_sequence=["#003399"])
        fig_rev.update_layout(template="plotly_white")
        st.plotly_chart(fig_rev, use_container_width=True, config=_PLOTLY_CFG)
        with st.expander("📊 Revenue Analysis", expanded=True):
            _render_analysis(df, "Revenue", {}, prefix="€", decimals=2, higher_is_better=True)

    with col_w:
        fig_waste = px.bar(df, x="Day", y="Waste",
                           title="Daily Waste (units)",
                           color_discrete_sequence=["#e74c3c"])
        fig_waste.update_layout(template="plotly_white")
        st.plotly_chart(fig_waste, use_container_width=True, config=_PLOTLY_CFG)
        with st.expander("📊 Waste Analysis", expanded=True):
            _render_analysis(df, "Waste", {}, suffix=" units", decimals=1, higher_is_better=False)

    fig_price = px.line(df, x="Day", y="Price",
                        title="Daily Selling Price",
                        color_discrete_sequence=["#e67e22"])
    fig_price.update_layout(template="plotly_white", yaxis_title="Price (€)")
    st.plotly_chart(fig_price, use_container_width=True, config=_PLOTLY_CFG)
    with st.expander("📊 Price Analysis", expanded=True):
        st.markdown("**Selling Price** — price per unit each day (includes inflation pass-through and discounts)")
        _render_analysis(df, "Price", {}, prefix="€", decimals=3, higher_is_better=False)

    if "NearExpiry" in df.columns:
        fig_ne = px.bar(df, x="Day", y="NearExpiry",
                        title="Near-Expiry Units Sold (50 % discount)",
                        color_discrete_sequence=["#f39c12"])
        fig_ne.update_layout(template="plotly_white")
        st.plotly_chart(fig_ne, use_container_width=True, config=_PLOTLY_CFG)
        with st.expander("📊 Near-Expiry Analysis", expanded=True):
            _render_analysis(df, "NearExpiry", {}, suffix=" units", decimals=1, higher_is_better=False)


# ===========================================================================
# 10. TAB: BEHAVIOURAL THEORY
# ===========================================================================

def render_behaviour_tab(params):
    st.header(_t("header_behaviour"))
    st.markdown(
        "Visualises how each embedded behavioural science theory shapes simulation "
        "outcomes. All charts update after running the **Interactive Demo** simulation."
    )

    df = st.session_state.get("sim_results")
    if df is None or df.empty:
        st.info("Run the **Interactive Demo** simulation first to populate these charts.")
        return

    import plotly.express as px
    import plotly.graph_objects as go

    # ── Row 1: TPB + Prospect Theory ────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("👥 Theory of Planned Behaviour")
        st.caption("Ajzen (1991) — Subjective Norm vs Panic Level over simulation days")
        if "AvgSubjectiveNorm" in df.columns:
            fig_tpb = go.Figure()
            fig_tpb.add_trace(go.Scatter(
                x=df["Day"], y=df["AvgSubjectiveNorm"],
                mode="lines", name="Avg Subjective Norm",
                line=dict(color="#92DDDB", width=2)
            ))
            fig_tpb.add_trace(go.Scatter(
                x=df["Day"], y=df["PanicLevel"],
                mode="lines", name="Global Panic",
                line=dict(color="#DBA159", width=2, dash="dot")
            ))
            if "AvgTPBIntention" in df.columns:
                fig_tpb.add_trace(go.Scatter(
                    x=df["Day"], y=df["AvgTPBIntention"],
                    mode="lines", name="TPB Purchase Intention",
                    line=dict(color="#BCDC8B", width=2, dash="dash")
                ))
            fig_tpb.update_layout(
                title="Theory of Planned Behaviour",
                xaxis_title="Day", yaxis_title="Score (0–1)",
                legend=dict(orientation="h", y=-0.25),
                height=320, margin=dict(t=40, b=60),
                template="plotly_white",
            )
            st.plotly_chart(fig_tpb, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 TPB Analysis", expanded=False):
                st.markdown("**Subjective Norm** — social influence driving purchase intention")
                _render_analysis(df, "AvgSubjectiveNorm", params, decimals=3, higher_is_better=True)
                if "AvgTPBIntention" in df.columns:
                    st.markdown("**TPB Purchase Intention** — combined attitude + norm + PBC score")
                    _render_analysis(df, "AvgTPBIntention", params, decimals=3, higher_is_better=True)
        else:
            st.warning("TPB columns not found in results. Re-run the simulation.")

    with c2:
        st.subheader("💰 Prospect Theory (Kahneman & Tversky 1979)")
        st.caption("Loss aversion — share of consumers with exhausted budgets vs daily revenue")
        if "BudgetExhaustionRate" in df.columns:
            pct_exhausted = (df["BudgetExhaustionRate"] * 100).clip(0, 100)
            fig_kt = go.Figure()
            fig_kt.add_trace(go.Bar(
                x=df["Day"], y=pct_exhausted,
                name="Budget Exhausted (%)",
                marker_color="#FCC995"
            ))
            if "Revenue" in df.columns:
                fig_kt.add_trace(go.Scatter(
                    x=df["Day"], y=df["Revenue"],
                    mode="lines", name="Daily Revenue (€)",
                    yaxis="y2",
                    line=dict(color="#44A1A0", width=2)
                ))
                fig_kt.update_layout(
                    yaxis2=dict(overlaying="y", side="right", title="Revenue (€)")
                )
            fig_kt.update_layout(
                title="Prospect Theory — Loss Aversion",
                xaxis_title="Day", yaxis_title="Budget Exhausted (%)",
                legend=dict(orientation="h", y=-0.25),
                height=320, margin=dict(t=40, b=60),
                template="plotly_white",
            )
            st.plotly_chart(fig_kt, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 Loss Aversion Analysis", expanded=False):
                st.markdown("**Budget Exhaustion Rate** — share of consumers who ran out of budget (proxy for loss aversion)")
                _render_analysis(df, "BudgetExhaustionRate", params, decimals=3, higher_is_better=False)
        else:
            st.warning("BudgetExhaustionRate not found. Re-run the simulation.")

    st.divider()

    # ── Row 2: Nudge (Gini) + FIES ──────────────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("⚖️ Nudge / Rationing — Gini Coefficient of Access")
        st.caption("Thaler & Sunstein (2008) · Sen (1973) — Purchase equity over time")
        if "GiniAccess" in df.columns:
            fig_gini = go.Figure()
            fig_gini.add_trace(go.Scatter(
                x=df["Day"], y=df["GiniAccess"],
                mode="lines+markers", name="Gini Access",
                line=dict(color="#DBA159", width=2),
                marker=dict(size=4)
            ))
            fig_gini.add_hrect(y0=0.0, y1=0.25, fillcolor="#BCDC8B", opacity=0.08,
                               annotation_text="Equitable", annotation_position="top right")
            fig_gini.add_hrect(y0=0.25, y1=0.50, fillcolor="#FCC995", opacity=0.08,
                               annotation_text="Moderate inequality")
            fig_gini.add_hrect(y0=0.50, y1=1.00, fillcolor="#FF5A5A", opacity=0.06,
                               annotation_text="High inequality")
            fig_gini.update_layout(
                title="Gini Coefficient of Access",
                xaxis_title="Day", yaxis_title="Gini (0=equal, 1=unequal)",
                yaxis=dict(range=[0, 1]),
                height=320, margin=dict(t=40, b=40),
                template="plotly_white",
            )
            rationing_on = params.get("purchase_limit") is not None
            st.plotly_chart(fig_gini, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 Gini Equity Analysis", expanded=False):
                st.markdown("**Gini Access Coefficient** — 0 = perfectly equal access, 1 = one consumer gets everything")
                _render_analysis(df, "GiniAccess", params, decimals=3, higher_is_better=False)
            if rationing_on:
                st.success(f"Purchase limit active: {params['purchase_limit']} units — check equity effect above.")
            else:
                st.info("Enable **Nudge — Purchase Limit** in the sidebar to see rationing effects.")
        else:
            st.warning("GiniAccess column not found. Re-run the simulation.")

    with c4:
        st.subheader("🍽️ FIES Food Security (FAO 2016)")
        st.caption("Food Insecurity Experience Scale — % severely food-insecure by income bracket")
        fies_cols = ["FIESSevere_Low", "FIESSevere_Mid", "FIESSevere_High"]
        available = [c for c in fies_cols if c in df.columns]
        if available:
            fig_fies = go.Figure()
            colors_fies = {"FIESSevere_Low": "#FF5A5A", "FIESSevere_Mid": "#FCC995", "FIESSevere_High": "#BCDC8B"}
            labels_fies = {"FIESSevere_Low": "Low income", "FIESSevere_Mid": "Mid income", "FIESSevere_High": "High income"}
            for col in available:
                fig_fies.add_trace(go.Scatter(
                    x=df["Day"], y=df[col] * 100,
                    mode="lines", name=labels_fies.get(col, col),
                    line=dict(color=colors_fies.get(col, "#92DDDB"), width=2)
                ))
            fig_fies.update_layout(
                title="FIES Food Security by Income Bracket",
                xaxis_title="Day", yaxis_title="Severely Food-Insecure (%)",
                legend=dict(orientation="h", y=-0.25),
                height=320, margin=dict(t=40, b=60),
                template="plotly_white",
            )
            st.plotly_chart(fig_fies, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 Food Security Analysis", expanded=False):
                for _fc in available:
                    _lbl = {"FIESSevere_Low": "Low-income severely food-insecure %",
                            "FIESSevere_Mid": "Mid-income severely food-insecure %",
                            "FIESSevere_High": "High-income severely food-insecure %"}.get(_fc, _fc)
                    st.markdown(f"**{_lbl}**")
                    _render_analysis(df, _fc, params, suffix=" (0–1)", decimals=3, higher_is_better=False)
        else:
            st.warning("FIES columns not found. Re-run the simulation.")

    st.divider()

    # ── Row 3: Stockpile Pressure + Media Channel ────────────────────────────
    c5, c6 = st.columns(2)

    with c5:
        st.subheader("🏠 Temporal Discounting & Stockpiling")
        st.caption("O'Donoghue & Rabin (1999) — β-δ quasi-hyperbolic discounting · stockpile pressure vs stockouts")
        if "StockpilePressure" in df.columns:
            fig_sp = go.Figure()
            fig_sp.add_trace(go.Scatter(
                x=df["Day"], y=df["StockpilePressure"],
                mode="lines", name="Stockpile Pressure",
                fill="tozeroy", fillcolor="rgba(220,161,89,0.15)",
                line=dict(color="#DBA159", width=2)
            ))
            if "LostSales" in df.columns:
                fig_sp.add_trace(go.Scatter(
                    x=df["Day"], y=df["LostSales"],
                    mode="lines", name="Lost Sales (€)",
                    yaxis="y2",
                    line=dict(color="#FF5A5A", width=2, dash="dot")
                ))
                fig_sp.update_layout(
                    yaxis2=dict(overlaying="y", side="right", title="Lost Sales (€)")
                )
            fig_sp.update_layout(
                title="Temporal Discounting & Stockpile Pressure",
                xaxis_title="Day", yaxis_title="Avg Stockpile Pressure",
                legend=dict(orientation="h", y=-0.25),
                height=320, margin=dict(t=40, b=60),
                template="plotly_white",
            )
            st.plotly_chart(fig_sp, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 Stockpile Analysis", expanded=False):
                st.markdown("**Stockpile Pressure** — avg urgency to accumulate pantry stock (β-δ discounting effect)")
                _render_analysis(df, "StockpilePressure", params, decimals=3, higher_is_better=False)
                if "LostSales" in df.columns:
                    st.markdown("**Lost Sales** — demand unmet due to stockouts driven by hoarding")
                    _render_analysis(df, "LostSales", params, prefix="€", decimals=1, higher_is_better=False)
        else:
            st.warning("StockpilePressure column not found. Re-run the simulation.")

    with c6:
        st.subheader("📡 Media / Communication Channel")
        st.caption("McCombs & Shaw (1972) — Agenda-setting · media type effect on global panic")
        if "MediaPanicEffect" in df.columns:
            fig_media = go.Figure()
            if "MediaType" in df.columns:
                for mtype, color in [("panic", "#FF5A5A"), ("calming", "#BCDC8B"), ("neutral", "#92DDDB")]:
                    mask = df["MediaType"] == mtype
                    if mask.any():
                        fig_media.add_trace(go.Scatter(
                            x=df.loc[mask, "Day"], y=df.loc[mask, "MediaPanicEffect"],
                            mode="markers", name=mtype.capitalize(),
                            marker=dict(color=color, size=6, opacity=0.8)
                        ))
            else:
                fig_media.add_trace(go.Scatter(
                    x=df["Day"], y=df["MediaPanicEffect"],
                    mode="lines", name="Media Panic Effect",
                    line=dict(color="#92DDDB", width=2)
                ))
            fig_media.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
            fig_media.update_layout(
                title="Media / Communication Channel (Agenda-Setting)",
                xaxis_title="Day", yaxis_title="Panic Δ per day",
                legend=dict(orientation="h", y=-0.25),
                height=320, margin=dict(t=40, b=60),
                template="plotly_white",
            )
            st.plotly_chart(fig_media, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 Media Effect Analysis", expanded=False):
                st.markdown("**Media Panic Effect** — daily change in global panic driven by media agenda-setting")
                _render_analysis(df, "MediaPanicEffect", params, decimals=4, higher_is_better=False)
            mtype = params.get("communication_type", "neutral")
            mint = params.get("media_intensity", 0.0)
            st.caption(f"Current setting: **{mtype.capitalize()}** at intensity **{mint:.2f}**")
        else:
            st.warning("MediaPanicEffect column not found. Re-run the simulation.")

    st.divider()

    # ── Row 4: Theory Reference Table ───────────────────────────────────────
    st.subheader("📚 Embedded Behavioural Theory Reference")
    import pandas as pd
    theory_df = pd.DataFrame([
        {
            "Theory": "Prospect Theory",
            "Authors": "Kahneman & Tversky (1979)",
            "Key Parameter": "Loss aversion λ=2.25, α=0.88",
            "Implementation": "KT value function replaces linear price disutility",
            "Policy Relevance": "Price controls, subsidy framing"
        },
        {
            "Theory": "Theory of Planned Behaviour",
            "Authors": "Ajzen (1991)",
            "Key Parameter": "Attitude (0.49), Norm (0.26), PBC (0.39)",
            "Implementation": "TPB intention modulates utility threshold each step",
            "Policy Relevance": "Social norms campaigns, messaging"
        },
        {
            "Theory": "Nudge / Choice Architecture",
            "Authors": "Thaler & Sunstein (2008)",
            "Key Parameter": "Purchase limit (units/visit); Gini equity index",
            "Implementation": "Per-product cap applied at purchase; Gini measured",
            "Policy Relevance": "Rationing fairness, access equity"
        },
        {
            "Theory": "FIES Food Security",
            "Authors": "FAO (2016)",
            "Key Parameter": "8-item scale (0=food-secure, 8=severely food-insecure)",
            "Implementation": "Per-agent daily score (simplified 4-flag proxy); aggregated by income bracket",
            "Policy Relevance": "Vulnerability targeting, food assistance"
        },
        {
            "Theory": "Temporal Discounting / Stockpiling",
            "Authors": "O'Donoghue & Rabin (1999)",
            "Key Parameter": "β (present bias), stockpile_days horizon",
            "Implementation": "β-δ quasi-hyperbolic model; pantry inventory tracking",
            "Policy Relevance": "Stockpile caps, hoarding mitigation"
        },
        {
            "Theory": "Media / Communication Channel",
            "Authors": "McCombs & Shaw (1972)",
            "Key Parameter": "media_intensity (0–1), type: neutral/panic/calming",
            "Implementation": "Daily ±0.10–0.12 global panic adjustment",
            "Policy Relevance": "Crisis communication strategy"
        },
    ])
    st.dataframe(theory_df, use_container_width=True, hide_index=True)


# ===========================================================================
# 11. TAB: EXPORT
# ===========================================================================

def render_export_tab():
    st.header(_t("header_export"))

    sections = {
        "Daily Aggregate (Baseline + Crisis)": "sim_results",
        "Per-Product Stock History":           "sim_stock",
        "Supply Chain Log (Orders + Deliveries)": "sim_scm_log",
        "Food Waste Log":                      "sim_waste",
        "Policy Comparison (Baseline)":        "policy_baseline",
        "Policy Comparison (Policy Scenario)": "policy_scenario",
    }

    any_data = False
    for label, key in sections.items():
        df = st.session_state.get(key)
        if df is not None and not df.empty:
            any_data = True
            with st.expander(f"📊 {label}", expanded=False):
                st.dataframe(df.head(200), use_container_width=True)
                st.download_button(
                    f"📥 Download {label} (CSV)",
                    df.to_csv(index=False).encode("utf-8"),
                    f"{key}.csv",
                    "text/csv",
                    key=f"dl_{key}",
                )

    if not any_data:
        st.info("Run the Interactive Demo simulation first to generate data for export.")

    # --- Full simulation dump ---
    if st.session_state.sim_results is not None:
        st.divider()
        st.markdown("### 📦 Full Simulation Bundle")
        frames = {}
        for label, key in sections.items():
            df = st.session_state.get(key)
            if df is not None and not df.empty:
                frames[label] = df

        if frames:
            buf = io.StringIO()
            for sheet_name, df in frames.items():
                buf.write(f"### {sheet_name}\n")
                df.to_csv(buf, index=False)
                buf.write("\n\n")
            st.download_button(
                "📥 Download Full Bundle (all sheets, CSV)",
                buf.getvalue().encode("utf-8"),
                "GROCERYsim_full_export.csv",
                "text/csv",
                key="dl_full_bundle",
            )


# ===========================================================================
# 11. PDF REPORT (simplified, encoding-safe)
# ===========================================================================

def _to_latin1(text: str) -> str:
    """
    Replace common non-latin-1 Unicode characters with safe ASCII equivalents,
    then hard-encode to latin-1.  fpdf uses latin-1 internally, so any character
    outside that range raises UnicodeEncodeError.
    The final encode("latin-1", "replace") acts as a catch-all safety net for
    any remaining non-latin-1 characters not listed in the table below.
    """
    _REPLACEMENTS = {
        # Dashes & punctuation
        "\u2014": "--",    # em dash
        "\u2013": "-",     # en dash
        "\u2012": "-",     # figure dash
        "\u2015": "--",    # horizontal bar
        "\u2019": "'",     # right single quotation mark
        "\u2018": "'",     # left single quotation mark
        "\u201c": '"',     # left double quotation mark
        "\u201d": '"',     # right double quotation mark
        "\u2026": "...",   # horizontal ellipsis
        "\u2022": "*",     # bullet
        "\u2023": ">",     # triangular bullet
        "\u2020": "+",     # dagger
        "\u2021": "++",    # double dagger
        # Superscripts (latin-1 has only 1,2,3 as \xb9,\xb2,\xb3)
        "\u00b2": "2",     # superscript 2
        "\u00b3": "3",     # superscript 3
        "\u00b9": "1",     # superscript 1
        "\u2070": "0",     # superscript 0
        "\u2074": "4",     # superscript 4
        "\u2075": "5",     # superscript 5
        "\u2076": "6",     # superscript 6
        "\u2077": "7",     # superscript 7
        "\u2078": "8",     # superscript 8
        "\u2079": "9",     # superscript 9
        # Subscripts (none are in latin-1)
        "\u2080": "0",     # subscript 0
        "\u2081": "1",     # subscript 1
        "\u2082": "2",     # subscript 2  ← CO₂
        "\u2083": "3",     # subscript 3
        "\u2084": "4",     # subscript 4
        "\u2085": "5",     # subscript 5
        "\u2086": "6",     # subscript 6
        "\u2087": "7",     # subscript 7
        "\u2088": "8",     # subscript 8
        "\u2089": "9",     # subscript 9
        # Maths & units
        "\u00b0": "deg",   # degree sign
        "\u00d7": "x",     # multiplication sign
        "\u00f7": "/",     # division sign
        "\u00b1": "+/-",   # plus-minus
        "\u2248": "~",     # almost equal
        "\u2260": "!=",    # not equal
        "\u2264": "<=",    # less-than or equal
        "\u2265": ">=",    # greater-than or equal
        "\u03b1": "alpha", # greek alpha
        "\u03b2": "beta",  # greek beta
        "\u03bc": "mu",    # greek mu
        "\u03c3": "sigma", # greek sigma
        # Arrows
        "\u2192": "->",    # right arrow
        "\u2190": "<-",    # left arrow
        "\u2191": "^",     # up arrow
        "\u2193": "v",     # down arrow
        "\u21d2": "=>",    # double right arrow
        # Currency
        "\u20ac": "EUR",   # euro sign (not in ISO-8859-1)
        "\u00a3": "GBP",   # pound sign (IS in latin-1 as \xa3, keep explicit)
        # Misc Latin that fpdf may reject
        "\u00e9": "\xe9",  # é  — keep as proper latin-1 byte
        "\u00e8": "\xe8",  # è
        "\u00e0": "\xe0",  # à
        "\u00fc": "\xfc",  # ü
        "\u00f6": "\xf6",  # ö
        "\u00e4": "\xe4",  # ä
        "\u00e5": "\xe5",  # å  (Finnish!)
        "\u00f8": "\xf8",  # ø
    }
    for char, repl in _REPLACEMENTS.items():
        text = text.replace(char, repl)
    # Final safety net: replace any remaining non-latin-1 characters with '?'
    return text.encode("latin-1", "replace").decode("latin-1")


class _PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 13)
        self.cell(0, 10, _to_latin1("GROCERYsim: Strategic Resilience Report"), 0, 1, "C")
        self.ln(3)

    def footer(self):
        self.set_y(-14)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def section(self, title: str):
        self.set_font("Arial", "B", 11)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 9, _to_latin1(title), 0, 1, "L", True)
        self.ln(3)

    def body(self, text: str):
        self.set_font("Arial", "", 9)
        self.multi_cell(0, 5, _to_latin1(text))
        self.ln(2)

    def add_mpl(self, fig):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, dpi=90, bbox_inches="tight")
            self.image(tmp.name, w=165)
            self.ln(4)


def _make_pdf_report(df: pd.DataFrame, baseline_name: str, n_runs: int) -> bytes:
    """Standard crisis-vs-baseline report (used by Scientific Analysis tab)."""
    pdf = _PDF()
    stats = df.groupby("Scenario")[["Revenue","Waste","LostSales"]].mean()

    base_rev   = stats.loc[baseline_name, "Revenue"]  if baseline_name in stats.index else 0
    cris_rev   = stats.loc["Crisis",      "Revenue"]  if "Crisis"       in stats.index else 0
    rev_drop   = (base_rev - cris_rev) / max(base_rev, 0.01) * 100
    base_waste = stats.loc[baseline_name, "Waste"] if baseline_name in stats.index else 0
    cris_waste = stats.loc["Crisis",      "Waste"] if "Crisis"       in stats.index else 0
    waste_inc  = (cris_waste - base_waste) / max(base_waste, 0.01) * 100

    pdf.add_page()
    pdf.section("1. Executive Summary")
    pdf.body(
        f"Avg daily revenue: EUR {base_rev:,.2f} (Baseline) "
        f"vs EUR {cris_rev:,.2f} (Crisis). "
        f"Revenue contraction: {rev_drop:.1f} %. "
        f"Food waste increased by {waste_inc:.1f} %."
    )

    pdf.add_page()
    pdf.section("2. Cumulative Revenue Comparison")
    df["CumRev"] = df.groupby(["Scenario","Run"])["Revenue"].cumsum()
    fig, ax = plt.subplots(figsize=(9, 4))
    for sc, color in [(baseline_name, "green"), ("Crisis", "red")]:
        sub = df[df["Scenario"] == sc]
        if not sub.empty:
            daily_mean = sub.groupby("Day")["CumRev"].mean()
            ax.plot(daily_mean.index, daily_mean.values, color=color, label=sc)
    ax.set_xlabel("Day"); ax.set_ylabel("Cumulative Revenue (EUR)")
    ax.set_title("Cumulative Revenue"); ax.legend(); ax.grid(alpha=0.3)
    pdf.add_mpl(fig); plt.close(fig)

    pdf.add_page()
    pdf.section("3. Daily Revenue Distribution")
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    for sc, color in [(baseline_name, "lightgreen"), ("Crisis", "salmon")]:
        sub = df[df["Scenario"] == sc]["Revenue"]
        if not sub.empty:
            ax2.hist(sub, bins=30, alpha=0.6, color=color, label=sc)
    ax2.set_xlabel("Revenue (EUR)"); ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Daily Revenue"); ax2.legend()
    pdf.add_mpl(fig2); plt.close(fig2)

    return bytes(pdf.output())


def _make_policy_pdf_brief(
    df_base:    pd.DataFrame,
    df_pol:     pd.DataFrame,
    pol_label:  str,
    narrative:  str,
    policy_cfg: dict,
) -> bytes:
    """
    Generate a policy brief PDF containing:
    • Cover / title page
    • Executive summary (narrative text)
    • KPI comparison table
    • Revenue, CO₂, and welfare time-series charts
    • Income vulnerability table
    • Methods note
    """
    pdf = _PDF()
    days = int(df_base["Day"].max())

    # ---- Cover ----
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, _to_latin1("GROCERYsim ABM -- Policy Impact Brief"), 0, 1, "C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, _to_latin1(f"Policy scenario: {pol_label}"), 0, 1, "C")
    pdf.cell(0, 8, _to_latin1(f"Simulation horizon: {days} days"), 0, 1, "C")
    pdf.ln(6)

    # ---- 1. Executive Summary ----
    pdf.section("1. Executive Summary")
    # Strip markdown bold markers for plain PDF text
    clean_narrative = (narrative
                       .replace("**", "")
                       .replace("*", ""))
    pdf.body(clean_narrative[:3000])   # truncate very long narratives

    # ---- 2. KPI Comparison Table ----
    pdf.add_page()
    pdf.section("2. Key Performance Indicators")
    kpi_defs = [
        ("Revenue/day (EUR)",      "Revenue",              False),
        ("Waste/day (units)",      "Waste",                False),
        ("CO2 total/day (kg)",     "CO2Total",             False),
        ("Import Dep. %",          "ImportDepPct",         False),
        ("Budget Exhaustion %",    "BudgetExhaustionRate", True),
        ("Food Stressed %",        "FoodStressedPct",      True),
        ("Fulfillment %",          "FulfillmentRate",      True),
        ("Mean Fat Purchased %",   "MeanFatPurchased",     False),
    ]
    pdf.set_font("Arial", "B", 9)
    pdf.cell(70, 6, "Metric", 1); pdf.cell(35, 6, "Baseline", 1)
    pdf.cell(35, 6, "Policy", 1); pdf.cell(35, 6, "Delta (%)", 1)
    pdf.ln()
    pdf.set_font("Arial", "", 9)
    for label, col, is_pct in kpi_defs:
        if col not in df_base.columns:
            continue
        b_val = df_base[col].mean() * (100 if is_pct else 1)
        p_val = df_pol [col].mean() * (100 if is_pct else 1)
        d_pct = (p_val - b_val) / max(abs(b_val), 1e-9) * 100
        pdf.cell(70, 6, _to_latin1(label), 1)
        pdf.cell(35, 6, f"{b_val:.2f}", 1)
        pdf.cell(35, 6, f"{p_val:.2f}", 1)
        pdf.cell(35, 6, f"{d_pct:+.1f}%", 1)
        pdf.ln()

    # ---- 3. Revenue chart ----
    pdf.add_page()
    pdf.section("3. Revenue — Baseline vs Policy")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(df_base.groupby("Day")["Revenue"].mean(), color="steelblue",  label="Baseline")
    ax.plot(df_pol .groupby("Day")["Revenue"].mean(), color="firebrick",  label="Policy")
    ax.set_xlabel("Day"); ax.set_ylabel("Revenue (EUR)")
    ax.legend(); ax.grid(alpha=0.3)
    pdf.add_mpl(fig); plt.close(fig)

    # ---- 4. CO2 chart ----
    pdf.section("4. CO2 Footprint — Baseline vs Policy")
    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    ax2.plot(df_base.groupby("Day")["CO2Total"].mean(), color="seagreen",  label="Baseline")
    ax2.plot(df_pol .groupby("Day")["CO2Total"].mean(), color="tomato",    label="Policy")
    ax2.set_xlabel("Day"); ax2.set_ylabel("kg CO2-eq")
    ax2.legend(); ax2.grid(alpha=0.3)
    pdf.add_mpl(fig2); plt.close(fig2)

    # ---- 5. Consumer Welfare chart ----
    pdf.add_page()
    pdf.section("5. Consumer Welfare — Budget Exhaustion & Food Stress")
    fig3, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for label_s, col_s, ax_s in [
        ("Budget Exhaustion %", "BudgetExhaustionRate", axes[0]),
        ("Food Stressed %",     "FoodStressedPct",      axes[1]),
    ]:
        if col_s in df_base.columns:
            ax_s.plot(df_base.groupby("Day")[col_s].mean()*100, color="steelblue", label="Baseline")
            ax_s.plot(df_pol .groupby("Day")[col_s].mean()*100, color="firebrick", label="Policy")
        ax_s.set_title(label_s); ax_s.set_xlabel("Day"); ax_s.legend()
        ax_s.grid(alpha=0.3)
    plt.tight_layout()
    pdf.add_mpl(fig3); plt.close(fig3)

    # ---- 6. Income Vulnerability ----
    pdf.section("6. Income Vulnerability (Budget Exhaustion by Bracket)")
    brackets = ["Low", "Mid", "High"]
    pdf.set_font("Arial", "B", 9)
    pdf.cell(50, 6, "Income Group", 1); pdf.cell(45, 6, "Baseline Exh. %", 1)
    pdf.cell(45, 6, "Policy Exh. %",  1); pdf.cell(35, 6, "Delta (pp)", 1)
    pdf.ln()
    pdf.set_font("Arial", "", 9)
    for br in brackets:
        col = f"BudgetExh_{br}"
        if col not in df_base.columns:
            continue
        b_v = df_base[col].mean() * 100
        p_v = df_pol [col].mean() * 100
        pdf.cell(50, 6, f"{br} income", 1)
        pdf.cell(45, 6, f"{b_v:.1f}%", 1)
        pdf.cell(45, 6, f"{p_v:.1f}%", 1)
        pdf.cell(35, 6, f"{p_v-b_v:+.2f}", 1)
        pdf.ln()

    # ---- 7. Methods ----
    pdf.add_page()
    pdf.section("7. Model Methods Note")
    pdf.body(
        "GROCERYsim ABM v2.0: Mesa-based agent-based model for Finnish dairy retail. "
        "Consumer agents calibrated from DCE preference scores and questionnaire data. "
        "K-Means archetype clustering (k=4); stratified bootstrap population pool. "
        "Utility function: U = origin_bonus + organic_bonus + fat_match - price_disutility. "
        "Agents update preferences via archetype-specific learning rules (rate=0.015/day). "
        "Policy levers: fat tax surcharge, domestic/organic subsidy, domestic supply shock, "
        "nutritional labelling preference boost. CO2 factors: Finnish organic=0.8, "
        "Finnish conventional=1.2, Imported organic=1.5, Imported conventional=2.2 kg CO2-eq/unit. "
        "SecureFood / Horizon Europe -- grant agreement No. 101136583."
    )

    return bytes(pdf.output())


# ===========================================================================
# 11b. POLICY NARRATIVE GENERATOR
# ===========================================================================

def _generate_policy_narrative(
    df_base: pd.DataFrame,
    df_pol:  pd.DataFrame,
    policy_cfg: dict,
    pol_label: str,
) -> str:
    """
    Return a plain-English paragraph (or several) interpreting the KPI deltas
    between the baseline and policy simulation runs.
    """

    def mean_delta(col: str, pct: bool = True) -> tuple[float, float]:
        """Returns (baseline_mean, policy_mean). pct=True means multiply by 100."""
        b = df_base.groupby("Day")[col].mean().mean()
        p = df_pol.groupby("Day")[col].mean().mean()
        if pct:
            return b * 100, p * 100
        return b, p

    def delta_pct(base: float, pol: float) -> float:
        if abs(base) < 1e-9:
            return 0.0
        return (pol - base) / abs(base) * 100

    sentences = []

    # ---- Overview ----
    active = [k for k in ["fat_tax_active","subsidy_active","domestic_shock_active","labelling_active"]
              if policy_cfg.get(k)]
    if not active:
        return "No policy levers were active during this run, so results are identical to the baseline."

    sentences.append(
        f"This simulation evaluated the combined effect of **{pol_label}** compared to a "
        f"no-policy baseline over {int(df_base['Day'].max())} simulated days."
    )

    # ---- Revenue ----
    b_rev, p_rev = mean_delta("Revenue", pct=False)
    rev_d = delta_pct(b_rev, p_rev)
    if abs(rev_d) >= 1.0:
        dir_rev = "increased" if rev_d > 0 else "decreased"
        sentences.append(
            f"Daily store revenue **{dir_rev} by {abs(rev_d):.1f}%** "
            f"(from €{b_rev:.2f} to €{p_rev:.2f} per day on average)."
        )

    # ---- Waste ----
    b_w, p_w = mean_delta("Waste", pct=False)
    w_d = delta_pct(b_w, p_w)
    if abs(w_d) >= 2.0:
        dir_w = "increased" if w_d > 0 else "decreased"
        sentences.append(
            f"Food waste **{dir_w} by {abs(w_d):.1f}%** under the policy scenario "
            f"({b_w:.1f} → {p_w:.1f} units wasted per day)."
        )

    # ---- Fat content ----
    if "fat_tax_active" in active or "labelling_active" in active:
        b_fat, p_fat = mean_delta("MeanFatPurchased", pct=False)
        fat_d = delta_pct(b_fat, p_fat)
        if abs(fat_d) >= 1.0:
            dir_fat = "fell" if fat_d < 0 else "rose"
            trigger = "fat tax" if "fat_tax_active" in active else "nutritional labelling"
            sentences.append(
                f"The average fat content of purchased products **{dir_fat} by {abs(fat_d):.1f}%** "
                f"(from {b_fat:.2f}% to {p_fat:.2f}%), suggesting the **{trigger}** is shifting "
                f"consumers toward lower-fat alternatives."
            )
        else:
            sentences.append(
                "Despite the policy, the average fat content of purchases changed by less than 1%, "
                "indicating consumers have not substantially shifted their fat preferences yet — "
                "a longer simulation horizon or higher tax rate may be needed."
            )

    # ---- CO2 ----
    b_co2, p_co2 = mean_delta("CO2Total", pct=False)
    co2_d = delta_pct(b_co2, p_co2)
    if abs(co2_d) >= 1.0:
        dir_co2 = "reduced" if co2_d < 0 else "increased"
        sentences.append(
            f"The total daily CO₂ footprint **{dir_co2} by {abs(co2_d):.1f}%** "
            f"({b_co2:.1f} → {p_co2:.1f} kg CO₂-eq/day)."
        )

    # ---- Import dependency ----
    b_imp, p_imp = mean_delta("ImportDepPct")
    imp_d = delta_pct(b_imp, p_imp)
    if abs(imp_d) >= 2.0:
        dir_imp = "increased" if imp_d > 0 else "decreased"
        cause = ""
        if "domestic_shock_active" in active and imp_d > 0:
            cause = " — driven by the domestic supply shock forcing greater reliance on imports"
        elif "subsidy_active" in active and policy_cfg.get("subsidy_target") in ("domestic","both") and imp_d < 0:
            cause = " — consistent with the domestic subsidy redirecting demand toward Finnish products"
        sentences.append(
            f"Import dependency **{dir_imp} by {abs(imp_d):.1f}%** "
            f"({b_imp:.1f}% → {p_imp:.1f}% of sales){cause}."
        )

    # ---- Consumer welfare ----
    b_bex, p_bex = mean_delta("BudgetExhaustionRate")
    bex_d = delta_pct(b_bex, p_bex)
    b_stress, p_stress = mean_delta("FoodStressedPct")
    stress_d = delta_pct(b_stress, p_stress)

    welfare_parts = []
    if abs(bex_d) >= 2.0:
        dir_bex = "rose" if bex_d > 0 else "fell"
        welfare_parts.append(
            f"the share of shoppers exhausting their budget **{dir_bex} by {abs(bex_d):.1f}%** "
            f"({b_bex:.1f}% → {p_bex:.1f}% of daily visitors)"
        )
    if abs(stress_d) >= 2.0:
        dir_s = "worsened" if stress_d > 0 else "improved"
        welfare_parts.append(
            f"food stress among low-income households **{dir_s} by {abs(stress_d):.1f}%** "
            f"({b_stress:.1f}% → {p_stress:.1f}%)"
        )
    if welfare_parts:
        sentences.append(
            "On consumer welfare, " + ", and ".join(welfare_parts) + ". "
            + ("This suggests the policy may disproportionately burden lower-income households "
               "and could benefit from a compensating income transfer or targeted exemption."
               if stress_d > 5 else "")
        )

    # ---- Fulfillment ----
    b_ful, p_ful = mean_delta("FulfillmentRate")
    ful_d = delta_pct(b_ful, p_ful)
    if abs(ful_d) >= 1.0:
        dir_ful = "improved" if ful_d > 0 else "declined"
        sentences.append(
            f"Basket fulfillment (items purchased vs. intended) **{dir_ful} by {abs(ful_d):.1f}%** "
            f"({b_ful:.1f}% → {p_ful:.1f}%), "
            + ("indicating better availability under the policy."
               if ful_d > 0 else
               "suggesting the policy may be pricing some consumers out of their planned purchases.")
        )

    # ---- Closing sentence ----
    net_positive = sum([
        rev_d > 0,
        w_d < 0,
        co2_d < 0,
        imp_d < 0 if "subsidy_active" in active else imp_d <= 5,
        bex_d <= 0,
    ])
    if net_positive >= 4:
        sentences.append(
            "**Overall**, the policy scenario shows broadly positive outcomes across economic, "
            "environmental, and welfare dimensions. Stakeholders may consider scaling or extending it."
        )
    elif net_positive <= 1:
        sentences.append(
            "**Overall**, the policy scenario shows mixed-to-negative outcomes. "
            "Consider re-calibrating the policy parameters or combining levers to mitigate trade-offs."
        )
    else:
        sentences.append(
            "**Overall**, the policy involves clear trade-offs: some dimensions improve while others "
            "worsen. Targeted compensating measures (e.g. low-income exemptions) could improve equity."
        )

    return "\n\n".join(sentences)


# ===========================================================================
# 11b. TAB: POLICY ANALYSIS
# ===========================================================================

def render_policy_tab(params: dict):
    st.header(_t("header_policy"))
    st.markdown(
        "Run a **Baseline vs Policy** comparison to quantify how each policy lever "
        "affects revenue, food waste, consumer welfare, and environmental footprint. "
        "All active policies are taken from the **🏛️ Policy Scenarios** sidebar section."
    )

    if st.session_state.config_data is None:
        st.warning("⚠️ Please upload and process data in the **🏠 Data & Population** tab first.")
        return

    policy_cfg = params.get("policy_cfg", {})
    n_active = sum([
        policy_cfg.get("fat_tax_active",         False),
        policy_cfg.get("subsidy_active",          False),
        policy_cfg.get("domestic_shock_active",   False),
        policy_cfg.get("labelling_active",        False),
    ])

    if n_active == 0:
        st.info(
            "ℹ️ No policy levers are currently enabled.  "
            "Turn on at least one lever in the **🏛️ Policy Scenarios** sidebar section, "
            "then click **Run Policy Comparison** below."
        )
    else:
        active_labels = []
        if policy_cfg.get("fat_tax_active"):
            active_labels.append(f"Fat Tax ({int(policy_cfg['fat_tax_rate']*100)}% on ≥{policy_cfg['fat_tax_threshold']}% fat)")
        if policy_cfg.get("subsidy_active"):
            active_labels.append(f"{policy_cfg['subsidy_target'].title()} Subsidy ({int(policy_cfg['subsidy_rate']*100)}% off)")
        if policy_cfg.get("domestic_shock_active"):
            active_labels.append(f"Supply Shock (day {policy_cfg['domestic_shock_day']}, {int(policy_cfg['domestic_shock_severity']*100)}% severity, {policy_cfg['domestic_shock_duration']} days)")
        if policy_cfg.get("labelling_active"):
            active_labels.append(f"Nutritional Labelling (from day {policy_cfg['labelling_day']})")
        policy_label = " + ".join(active_labels)
        st.success(f"**Active policies:** {policy_label}")

    days = params["days"]
    col_run, col_name, col_runs = st.columns([2, 2, 1])
    with col_run:
        run_btn = st.button(_t("btn_run_policy"), type="primary", key="pol_run_btn")
    with col_name:
        scenario_name = st.text_input(
            "Scenario name (for multi-comparison)",
            value=policy_label if n_active > 0 else "Baseline",
            key="pol_scenario_name",
            help="Give this scenario a name so you can add more and compare them side-by-side.",
        )
    with col_runs:
        pol_runs = st.number_input("Runs / scenario", 1, 30, 5, key="pol_mc_runs",
                                   help="Average over N runs to reduce noise")

    col_info, col_clear = st.columns([2, 1])
    stored_count = len(st.session_state.get("policy_scenarios", []))
    if stored_count:
        col_info.caption(
            f"**{stored_count}** scenario(s) saved. "
            "Run more with different sidebar settings and a new name to compare them side-by-side (max 4)."
        )
    with col_clear:
        if st.button("🗑️ Clear Saved Scenarios", key="pol_clear_btn"):
            st.session_state.policy_scenarios = []
            st.session_state.policy_baseline  = None
            st.session_state.policy_scenario  = None
            st.rerun()

    if run_btn:
        if st.session_state.config_data is None:
            st.error("No data loaded.")
            return

        progress = st.progress(0, text="Initialising…")
        total_steps = int(pol_runs) * 2 * days
        step_counter = [0]

        def tick(label=""):
            step_counter[0] += 1
            progress.progress(
                min(step_counter[0] / total_steps, 1.0),
                text=label,
            )

        baseline_records: list[dict] = []
        policy_records:   list[dict] = []

        for run_i in range(int(pol_runs)):
            seed = 100 + run_i * 7

            # ---- Baseline (no policy) ----
            m_base = _make_model(params, False, seed, policy_cfg=None)
            for d in range(1, days + 1):
                m_base.step()
                rec = m_base.daily_records[-1].copy()
                rec["Run"] = run_i
                rec["Scenario"] = "Baseline"
                baseline_records.append(rec)
                if d % 10 == 0:
                    tick(f"Baseline run {run_i+1}/{pol_runs} · day {d}")

            # ---- Policy scenario ----
            m_pol = _make_model(params, False, seed, policy_cfg=policy_cfg)
            for d in range(1, days + 1):
                m_pol.step()
                rec = m_pol.daily_records[-1].copy()
                rec["Run"] = run_i
                rec["Scenario"] = "Policy"
                policy_records.append(rec)
                if d % 10 == 0:
                    tick(f"Policy run {run_i+1}/{pol_runs} · day {d}")

        progress.empty()

        df_base_new = pd.DataFrame(baseline_records)
        df_pol_new  = pd.DataFrame(policy_records)
        used_label  = scenario_name.strip() or (policy_label if n_active > 0 else "Baseline")

        st.session_state.policy_baseline = df_base_new
        st.session_state.policy_scenario = df_pol_new
        st.session_state.policy_label    = used_label

        # Add to multi-scenario store (cap at 4, replace if same name)
        scenarios_store = st.session_state.policy_scenarios or []
        scenarios_store = [s for s in scenarios_store if s["label"] != used_label]
        scenarios_store.append({"label": used_label, "df": df_pol_new, "cfg": policy_cfg.copy()})
        if len(scenarios_store) > 4:
            scenarios_store = scenarios_store[-4:]  # keep most recent 4
        st.session_state.policy_scenarios = scenarios_store
        st.success(f"✅ Scenario **{used_label}** complete! ({len(scenarios_store)} scenario(s) stored)")

    # ---- Display results ----
    if st.session_state.policy_baseline is None:
        st.info("Click **▶️ Run & Compare** to generate charts.")
        return

    df_base = st.session_state.policy_baseline
    df_pol  = st.session_state.policy_scenario
    pol_lbl = st.session_state.policy_label or "Policy"

    # Combine for plotting
    df_all = pd.concat([df_base, df_pol], ignore_index=True)

    # ---- Smooth daily means across runs ----
    def daily_mean(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
        return (
            df[df["Scenario"] == scenario]
            .groupby("Day")
            .mean(numeric_only=True)
            .reset_index()
        )

    base_d = daily_mean(df_all, "Baseline")
    pol_d  = daily_mean(df_all, "Policy")

    # =====================================================================
    # Multi-scenario overlay (if more than 1 scenario stored)
    # =====================================================================
    stored_scenarios = st.session_state.get("policy_scenarios", [])
    if len(stored_scenarios) >= 2:
        st.subheader("📊 Multi-Scenario Comparison")
        st.caption(
            f"{len(stored_scenarios)} scenarios stored — run more with different sidebar settings "
            "and a unique name to add them here. Up to 4 scenarios are kept."
        )
        _MULTI_COLORS = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]

        ms_metric = st.selectbox(
            "Compare by metric:",
            ["Revenue", "Waste", "CO2Total", "ImportDepPct",
             "BudgetExhaustionRate", "FulfillmentRate", "MeanFatPurchased"],
            key="ms_metric_sel",
        )
        _BASELINE_KEY = "Baseline"

        fig_ms = go.Figure()
        # Baseline first
        b_mean = df_base.groupby("Day")[ms_metric].mean().reset_index()
        fig_ms.add_trace(go.Scatter(
            x=b_mean["Day"], y=b_mean[ms_metric],
            name="Baseline (no policy)", mode="lines",
            line=dict(color="#555555", dash="dash"),
        ))
        for i, sc in enumerate(stored_scenarios):
            sc_mean = sc["df"].groupby("Day")[ms_metric].mean().reset_index()
            fig_ms.add_trace(go.Scatter(
                x=sc_mean["Day"], y=sc_mean[ms_metric],
                name=sc["label"], mode="lines",
                line=dict(color=_MULTI_COLORS[i % len(_MULTI_COLORS)]),
            ))
        fig_ms.update_layout(
            title=f"{ms_metric} — Baseline vs All Policy Scenarios",
            xaxis_title="Day", template="plotly_white",
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_ms, use_container_width=True, config=_PLOTLY_CFG)

        # Bar chart: mean value per scenario
        bar_data = [{"Scenario": "Baseline (no policy)",
                     ms_metric: df_base[ms_metric].mean()}]
        for sc in stored_scenarios:
            bar_data.append({"Scenario": sc["label"], ms_metric: sc["df"][ms_metric].mean()})
        df_bar_ms = pd.DataFrame(bar_data)
        fig_bar_ms = px.bar(
            df_bar_ms, x="Scenario", y=ms_metric,
            color="Scenario",
            color_discrete_sequence=["#555555"] + _MULTI_COLORS,
            title=f"Mean {ms_metric} by Scenario",
        )
        fig_bar_ms.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_bar_ms, use_container_width=True, config=_PLOTLY_CFG)
        st.divider()

    # =====================================================================
    # KPI summary cards
    # =====================================================================
    st.subheader("📊 KPI Summary")

    def kpi_delta(base_val, pol_val, higher_is_better=True, fmt=".2f"):
        delta = pol_val - base_val
        pct   = (delta / abs(base_val) * 100) if base_val != 0 else 0.0
        sign  = "+" if delta >= 0 else ""
        color = "normal" if (delta >= 0) == higher_is_better else "inverse"
        return f"{pol_val:{fmt}}", f"{sign}{pct:.1f}%", color

    kpis = [
        ("💰 Revenue/day",   base_d["Revenue"].mean(),   pol_d["Revenue"].mean(),   True,  ".2f"),
        ("🗑️ Waste/day",     base_d["Waste"].mean(),     pol_d["Waste"].mean(),     False, ".1f"),
        ("📦 Sales/day",     base_d["Sales"].mean(),     pol_d["Sales"].mean(),     True,  ".1f"),
        ("🌍 CO₂/day (kg)",  base_d["CO2Total"].mean(),  pol_d["CO2Total"].mean(),  False, ".1f"),
        ("🌐 Import Dep.%",  base_d["ImportDepPct"].mean(), pol_d["ImportDepPct"].mean(), False, ".1f"),
        ("💸 Budget Exh.%",  base_d["BudgetExhaustionRate"].mean()*100,
                             pol_d["BudgetExhaustionRate"].mean()*100, False, ".1f"),
        ("🍔 Mean Fat%",     base_d["MeanFatPurchased"].mean(),
                             pol_d["MeanFatPurchased"].mean(), False, ".2f"),
        ("✅ Fulfillment%",  base_d["FulfillmentRate"].mean()*100,
                             pol_d["FulfillmentRate"].mean()*100, True, ".1f"),
    ]

    cols = st.columns(4)
    for i, (label, base_v, pol_v, hib, fmt) in enumerate(kpis):
        val_str, delta_str, color = kpi_delta(base_v, pol_v, hib, fmt)
        cols[i % 4].metric(
            label=f"{label}",
            value=val_str,
            delta=f"{delta_str} vs baseline",
            delta_color=color,
        )

    st.divider()

    # =====================================================================
    # Auto-generated narrative interpretation
    # =====================================================================
    st.subheader("📝 Policy Impact Summary")
    with st.spinner("Generating narrative…"):
        narrative = _generate_policy_narrative(
            df_base, df_pol,
            policy_cfg=params.get("policy_cfg", {}),
            pol_label=pol_lbl,
        )
    st.markdown(narrative)

    st.divider()

    # =====================================================================
    # Chart section
    # =====================================================================
    chart_tabs = st.tabs([
        "💰 Economic", "🌍 Environmental", "👥 Consumer Welfare",
        "📉 Income Vulnerability", "📊 Detailed Data"
    ])

    # ---- Economic ----
    with chart_tabs[0]:
        col_a, col_b = st.columns(2)

        with col_a:
            fig_rev = go.Figure()
            for scen, color in [("Baseline", "#003399"), ("Policy", "#e74c3c")]:
                d = df_all[df_all["Scenario"] == scen].groupby("Day")["Revenue"].mean().reset_index()
                fig_rev.add_trace(go.Scatter(
                    x=d["Day"], y=d["Revenue"], name=scen,
                    line=dict(color=color), mode="lines"
                ))
            fig_rev.update_layout(
                title="Daily Revenue — Baseline vs Policy",
                xaxis_title="Day", yaxis_title="Revenue (€)",
                template="plotly_white", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_rev, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 Revenue Analysis", expanded=True):
                _render_analysis(df_all, "Revenue", {}, prefix="€", decimals=0,
                                 higher_is_better=True,
                                 baseline_label="Baseline", crisis_label="Policy")

        with col_b:
            fig_waste = go.Figure()
            for scen, color in [("Baseline", "#003399"), ("Policy", "#e74c3c")]:
                d = df_all[df_all["Scenario"] == scen].groupby("Day")["Waste"].mean().reset_index()
                fig_waste.add_trace(go.Scatter(
                    x=d["Day"], y=d["Waste"], name=scen,
                    line=dict(color=color), mode="lines"
                ))
            fig_waste.update_layout(
                title="Daily Waste — Baseline vs Policy",
                xaxis_title="Day", yaxis_title="Wasted Units",
                template="plotly_white", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_waste, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 Waste Analysis", expanded=True):
                _render_analysis(df_all, "Waste", {}, suffix=" units", decimals=1,
                                 higher_is_better=False,
                                 baseline_label="Baseline", crisis_label="Policy")

        # Revenue distribution box-plot
        fig_box = px.box(
            df_all.groupby(["Run","Scenario"])["Revenue"].sum().reset_index(),
            x="Scenario", y="Revenue", color="Scenario",
            color_discrete_map={"Baseline": "#003399", "Policy": "#e74c3c"},
            title="Total Revenue Distribution Across Runs",
        )
        fig_box.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True, config=_PLOTLY_CFG)

    # ---- Environmental ----
    with chart_tabs[1]:
        col_a, col_b = st.columns(2)

        with col_a:
            fig_co2 = go.Figure()
            _CO2_FILL = {"Baseline": "rgba(39,174,96,0.10)", "Policy": "rgba(231,76,60,0.10)"}
            for scen, color in [("Baseline", "#27ae60"), ("Policy", "#e74c3c")]:
                d = df_all[df_all["Scenario"] == scen].groupby("Day")["CO2Total"].mean().reset_index()
                fig_co2.add_trace(go.Scatter(
                    x=d["Day"], y=d["CO2Total"], name=scen,
                    line=dict(color=color), mode="lines", fill="tozeroy",
                    fillcolor=_CO2_FILL[scen],
                ))
            fig_co2.update_layout(
                title="Daily CO₂ Footprint (kg CO₂-eq) — Baseline vs Policy",
                xaxis_title="Day", yaxis_title="kg CO₂-eq",
                template="plotly_white", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_co2, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 CO₂ Analysis", expanded=True):
                _render_analysis(df_all, "CO2Total", {}, suffix=" kg CO₂-eq", decimals=1,
                                 higher_is_better=False,
                                 baseline_label="Baseline", crisis_label="Policy")

        with col_b:
            fig_imp = go.Figure()
            for scen, color in [("Baseline", "#27ae60"), ("Policy", "#e74c3c")]:
                d = df_all[df_all["Scenario"] == scen].groupby("Day")["ImportDepPct"].mean().reset_index()
                fig_imp.add_trace(go.Scatter(
                    x=d["Day"], y=d["ImportDepPct"], name=scen,
                    line=dict(color=color), mode="lines"
                ))
            fig_imp.update_layout(
                title="Import Dependency (% of sales from imported products)",
                xaxis_title="Day", yaxis_title="Import Dependency %",
                template="plotly_white", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_imp, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 Import Dependency Analysis", expanded=True):
                _render_analysis(df_all, "ImportDepPct", {}, suffix="%", decimals=1,
                                 higher_is_better=False,
                                 baseline_label="Baseline", crisis_label="Policy")

        # CO2 breakdown: sales vs waste
        co2_summary = df_all.groupby("Scenario")[["CO2Sales","CO2Waste"]].mean().reset_index()
        fig_co2_break = px.bar(
            co2_summary.melt(id_vars="Scenario", value_vars=["CO2Sales","CO2Waste"],
                             var_name="Type", value_name="kg CO₂-eq/day"),
            x="Scenario", y="kg CO₂-eq/day", color="Type",
            barmode="stack",
            color_discrete_map={"CO2Sales": "#27ae60", "CO2Waste": "#e74c3c"},
            title="Average Daily CO₂ Breakdown: Sales vs Waste",
        )
        fig_co2_break.update_layout(template="plotly_white")
        st.plotly_chart(fig_co2_break, use_container_width=True, config=_PLOTLY_CFG)

    # ---- Consumer Welfare ----
    with chart_tabs[2]:
        col_a, col_b = st.columns(2)

        with col_a:
            fig_bex = go.Figure()
            for scen, color in [("Baseline", "#003399"), ("Policy", "#e74c3c")]:
                d = df_all[df_all["Scenario"] == scen].groupby("Day")["BudgetExhaustionRate"].mean().reset_index()
                fig_bex.add_trace(go.Scatter(
                    x=d["Day"], y=d["BudgetExhaustionRate"] * 100,
                    name=scen, line=dict(color=color), mode="lines"
                ))
            fig_bex.update_layout(
                title="Budget Exhaustion Rate (% consumers) — Baseline vs Policy",
                xaxis_title="Day", yaxis_title="% Consumers",
                template="plotly_white", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_bex, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 Budget Exhaustion Analysis", expanded=True):
                _render_analysis(df_all, "BudgetExhaustionRate", {}, suffix=" (0–1)", decimals=3,
                                 higher_is_better=False,
                                 baseline_label="Baseline", crisis_label="Policy")

        with col_b:
            fig_stress = go.Figure()
            for scen, color in [("Baseline", "#003399"), ("Policy", "#e74c3c")]:
                d = df_all[df_all["Scenario"] == scen].groupby("Day")["FoodStressedPct"].mean().reset_index()
                fig_stress.add_trace(go.Scatter(
                    x=d["Day"], y=d["FoodStressedPct"] * 100,
                    name=scen, line=dict(color=color), mode="lines"
                ))
            fig_stress.update_layout(
                title="Food-Stressed Consumers (low-income + budget exhausted)",
                xaxis_title="Day", yaxis_title="% Consumers",
                template="plotly_white", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_stress, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 Food Stress Analysis", expanded=True):
                _render_analysis(df_all, "FoodStressedPct", {}, suffix=" (0–1)", decimals=3,
                                 higher_is_better=False,
                                 baseline_label="Baseline", crisis_label="Policy")

        col_c, col_d = st.columns(2)

        with col_c:
            fig_fat = go.Figure()
            for scen, color in [("Baseline", "#003399"), ("Policy", "#e74c3c")]:
                d = df_all[df_all["Scenario"] == scen].groupby("Day")["MeanFatPurchased"].mean().reset_index()
                fig_fat.add_trace(go.Scatter(
                    x=d["Day"], y=d["MeanFatPurchased"],
                    name=scen, line=dict(color=color), mode="lines"
                ))
            fig_fat.update_layout(
                title="Mean Fat Content of Purchased Products",
                xaxis_title="Day", yaxis_title="Avg Fat% per unit bought",
                template="plotly_white", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_fat, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 Dietary Analysis", expanded=True):
                _render_analysis(df_all, "MeanFatPurchased", {}, suffix="%", decimals=2,
                                 higher_is_better=False,
                                 baseline_label="Baseline", crisis_label="Policy")

        with col_d:
            fig_ful = go.Figure()
            for scen, color in [("Baseline", "#003399"), ("Policy", "#e74c3c")]:
                d = df_all[df_all["Scenario"] == scen].groupby("Day")["FulfillmentRate"].mean().reset_index()
                fig_ful.add_trace(go.Scatter(
                    x=d["Day"], y=d["FulfillmentRate"] * 100,
                    name=scen, line=dict(color=color), mode="lines"
                ))
            fig_ful.update_layout(
                title="Basket Fulfillment Rate (items purchased / items wanted)",
                xaxis_title="Day", yaxis_title="Fulfillment %",
                yaxis_range=[0, 105],
                template="plotly_white", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_ful, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 Fulfillment Analysis", expanded=True):
                _render_analysis(df_all, "FulfillmentRate", {}, suffix=" (0–1)", decimals=3,
                                 higher_is_better=True,
                                 baseline_label="Baseline", crisis_label="Policy")

    # ---- Income Vulnerability ----
    with chart_tabs[3]:
        st.markdown(
            "How does the policy affect consumers differently across **income brackets**? "
            "Brackets: **Low** (<€1 500/mo), **Mid** (€1 500–3 000/mo), **High** (>€3 000/mo). "
            "Values shown are averages over all simulation days and runs."
        )

        BRACKETS  = ["Low", "Mid", "High"]
        COLORS_B  = {"Baseline": "#003399", "Policy": "#e74c3c"}
        BR_COLORS = {"Low": "#e74c3c", "Mid": "#f39c12", "High": "#27ae60"}

        # ---- Budget exhaustion by bracket ----
        bex_rows = []
        for scen, df_s in [("Baseline", df_base), ("Policy", df_pol)]:
            for br in BRACKETS:
                val = df_s.groupby("Day")[f"BudgetExh_{br}"].mean().mean() * 100
                bex_rows.append({"Scenario": scen, "Bracket": br, "Budget Exhaustion %": val})
        df_bex_br = pd.DataFrame(bex_rows)

        # ---- Fulfillment by bracket ----
        ful_rows = []
        for scen, df_s in [("Baseline", df_base), ("Policy", df_pol)]:
            for br in BRACKETS:
                val = df_s.groupby("Day")[f"Fulfillment_{br}"].mean().mean() * 100
                ful_rows.append({"Scenario": scen, "Bracket": br, "Fulfillment %": val})
        df_ful_br = pd.DataFrame(ful_rows)

        # ---- Mean fat by bracket ----
        fat_rows = []
        for scen, df_s in [("Baseline", df_base), ("Policy", df_pol)]:
            for br in BRACKETS:
                val = df_s.groupby("Day")[f"MeanFat_{br}"].mean().mean()
                fat_rows.append({"Scenario": scen, "Bracket": br, "Mean Fat %": val})
        df_fat_br = pd.DataFrame(fat_rows)

        col_a, col_b = st.columns(2)

        with col_a:
            fig_bex_br = px.bar(
                df_bex_br, x="Bracket", y="Budget Exhaustion %",
                color="Scenario", barmode="group",
                color_discrete_map=COLORS_B,
                title="Budget Exhaustion Rate by Income Bracket",
                category_orders={"Bracket": BRACKETS},
            )
            fig_bex_br.update_layout(template="plotly_white",
                                     legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig_bex_br, use_container_width=True, config=_PLOTLY_CFG)

        with col_b:
            fig_ful_br = px.bar(
                df_ful_br, x="Bracket", y="Fulfillment %",
                color="Scenario", barmode="group",
                color_discrete_map=COLORS_B,
                title="Basket Fulfillment Rate by Income Bracket",
                category_orders={"Bracket": BRACKETS},
            )
            fig_ful_br.update_layout(template="plotly_white",
                                     legend=dict(orientation="h", y=-0.25),
                                     yaxis_range=[0, 105])
            st.plotly_chart(fig_ful_br, use_container_width=True, config=_PLOTLY_CFG)

        # ---- Mean fat grouped bar ----
        fig_fat_br = px.bar(
            df_fat_br, x="Bracket", y="Mean Fat %",
            color="Scenario", barmode="group",
            color_discrete_map=COLORS_B,
            title="Mean Fat Content Purchased by Income Bracket",
            category_orders={"Bracket": BRACKETS},
        )
        fig_fat_br.update_layout(template="plotly_white",
                                 legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_fat_br, use_container_width=True, config=_PLOTLY_CFG)

        # ---- Delta heatmap: policy effect on each bracket ----
        st.markdown("#### Policy Δ by Bracket (Policy − Baseline, percentage points)")
        delta_data = []
        for metric, col_tpl, label in [
            ("BudgetExh_{}", True,  "Budget Exhaustion %"),
            ("Fulfillment_{}", True, "Fulfillment %"),
            ("MeanFat_{}", False,   "Mean Fat %"),
        ]:
            row = {"Metric": label}
            for br in BRACKETS:
                b_val = df_base.groupby("Day")[metric.format(br)].mean().mean()
                p_val = df_pol .groupby("Day")[metric.format(br)].mean().mean()
                mult  = 100 if col_tpl else 1
                row[br] = round((p_val - b_val) * mult, 3)
            delta_data.append(row)
        df_delta = pd.DataFrame(delta_data).set_index("Metric")

        def _color_delta(val):
            # Red = worse (higher budget exh / fat, lower fulfillment); Green = better
            # We invert for Fulfillment
            return "color: #e74c3c" if val > 0 else ("color: #27ae60" if val < 0 else "")

        # applymap → map in pandas ≥ 2.1; fall back gracefully
        _styler = df_delta.style
        try:
            _styler = _styler.map(_color_delta)
        except AttributeError:
            _styler = _styler.applymap(_color_delta)
        st.dataframe(
            _styler.format("{:+.3f}"),
            use_container_width=True,
        )
        st.caption(
            "Red = policy made this metric worse for that income group | "
            "Green = policy improved it | Values are percentage-point changes."
        )

        # ---- Narrative interpretation of vulnerability ----
        vuln_sentences = []
        for br in BRACKETS:
            b_bex_br = df_base.groupby("Day")[f"BudgetExh_{br}"].mean().mean() * 100
            p_bex_br = df_pol .groupby("Day")[f"BudgetExh_{br}"].mean().mean() * 100
            d = p_bex_br - b_bex_br
            if abs(d) >= 1.0:
                dir_s = "increased" if d > 0 else "decreased"
                vuln_sentences.append(
                    f"**{br}-income households**: budget exhaustion {dir_s} by "
                    f"{abs(d):.1f} pp ({b_bex_br:.1f}% → {p_bex_br:.1f}%)."
                )
        if vuln_sentences:
            st.markdown("**Income vulnerability summary:**")
            for s in vuln_sentences:
                st.markdown(f"- {s}")

            # Check for regressive pattern
            low_d = (df_pol.groupby("Day")["BudgetExh_Low"].mean().mean()
                   - df_base.groupby("Day")["BudgetExh_Low"].mean().mean()) * 100
            high_d = (df_pol.groupby("Day")["BudgetExh_High"].mean().mean()
                    - df_base.groupby("Day")["BudgetExh_High"].mean().mean()) * 100
            if low_d > 0 and low_d > high_d + 2:
                st.warning(
                    "⚠️ **Regressive impact detected**: low-income households bear a "
                    f"disproportionately larger increase in budget exhaustion "
                    f"({low_d:+.1f} pp) compared to high-income households ({high_d:+.1f} pp). "
                    "Consider adding a targeted low-income exemption or subsidy."
                )
            elif low_d < 0 and abs(low_d) > abs(high_d) + 2:
                st.success(
                    "✅ **Progressive impact**: low-income households benefit most from the policy "
                    f"({low_d:+.1f} pp budget exhaustion change vs. {high_d:+.1f} pp for high-income)."
                )

    # ---- Detailed Data ----
    with chart_tabs[4]:
        st.markdown("### Mean daily metrics by scenario")
        summary_cols = [
            "Revenue","Waste","Sales","LostSales",
            "CO2Total","ImportDepPct",
            "BudgetExhaustionRate","FoodStressedPct","FulfillmentRate","MeanFatPurchased",
        ]
        summary = df_all.groupby("Scenario")[summary_cols].mean().T.round(4)
        # Compute absolute delta and % change
        if "Baseline" in summary.columns and "Policy" in summary.columns:
            summary["Δ (Policy − Baseline)"] = summary["Policy"] - summary["Baseline"]
            summary["Δ%"] = ((summary["Policy"] - summary["Baseline"]) / summary["Baseline"].abs().clip(lower=1e-9) * 100).round(2)
        st.dataframe(summary, use_container_width=True)

        st.markdown("### Download combined records")
        dl_all = pd.concat([df_base, df_pol], ignore_index=True)
        col_csv, col_pdf = st.columns(2)
        col_csv.download_button(
            "📥 Download Policy Comparison (CSV)",
            dl_all.to_csv(index=False).encode("utf-8"),
            "policy_comparison.csv",
            "text/csv",
            key="dl_policy",
        )
        with col_pdf:
            with st.spinner("Generating PDF policy brief…"):
                narrative_for_pdf = _generate_policy_narrative(
                    df_base, df_pol,
                    policy_cfg=params.get("policy_cfg", {}),
                    pol_label=pol_lbl,
                )
                pdf_bytes = _make_policy_pdf_brief(
                    df_base, df_pol,
                    pol_label=pol_lbl,
                    narrative=narrative_for_pdf,
                    policy_cfg=params.get("policy_cfg", {}),
                )
            st.download_button(
                "📄 Download Policy Brief (PDF)",
                pdf_bytes,
                "policy_brief.pdf",
                "application/pdf",
                key="dl_policy_pdf",
            )


# ===========================================================================
# 11c. TAB: STAKEHOLDER VIEW
# ===========================================================================

def render_stakeholder_tab():
    st.header(_t("header_stakeholder"))
    st.markdown(
        "Three curated dashboards — each filtered to the metrics that matter most "
        "for a specific audience. All data comes from the **Interactive Demo** simulation run."
    )

    if st.session_state.sim_results is None:
        st.warning("⚠️ Run the **🎮 Interactive Demo** simulation first to populate this tab.")
        return

    df        = st.session_state.sim_results
    df_stock  = st.session_state.sim_stock
    df_waste  = st.session_state.sim_waste

    view = st.radio(
        "Select your role:",
        ["🏪 Retailer", "📋 Policy Maker", "🔬 Researcher"],
        horizontal=True,
        key="stakeholder_view",
    )

    scenarios = df["Scenario"].unique().tolist()
    sel_sc    = st.selectbox("Scenario:", scenarios, key="sh_scenario")
    df_sc     = df[df["Scenario"] == sel_sc]

    # -----------------------------------------------------------------------
    if view == "🏪 Retailer":
        st.subheader("🏪 Retailer Dashboard — Operations & Profitability")
        st.markdown(
            "Focus: **revenue, gross margin, waste cost, stockout losses, shelf availability**. "
            "Key question: *Is the store running efficiently and profitably?*"
        )

        total_rev   = df_sc["Revenue"].sum()
        total_waste = df_sc["Waste"].sum()
        total_lost  = df_sc["LostSales"].sum()
        total_sales = df_sc["Sales"].sum()
        waste_pct   = total_waste / max(1, total_sales + total_waste) * 100

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Revenue",    f"€{total_rev:,.0f}")
        k2.metric("Units Sold",       f"{total_sales:,}")
        k3.metric("Waste Units",      f"{total_waste:,}", f"{waste_pct:.1f}% of stock")
        k4.metric("Lost Sales Value", f"€{total_lost:,.0f}")
        k5.metric("Avg Daily Consumers", f"{df_sc['Consumers'].mean():.0f}")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_r = px.area(df_sc, x="Day", y="Revenue",
                            title="Daily Revenue Trend",
                            color_discrete_sequence=["#2E8B57"])
            fig_r.update_layout(template="plotly_white")
            st.plotly_chart(fig_r, use_container_width=True, config=_PLOTLY_CFG)

        with col_b:
            fig_w = px.bar(df_sc, x="Day", y="Waste",
                           title="Daily Waste (Units)",
                           color_discrete_sequence=["#e74c3c"])
            fig_w.update_layout(template="plotly_white")
            st.plotly_chart(fig_w, use_container_width=True, config=_PLOTLY_CFG)

        # Top 10 products by revenue
        if df_stock is not None and not df_stock.empty:
            df_stock_sc = df_stock[df_stock["Scenario"] == sel_sc]
            top_rev = (df_stock_sc.groupby("Product")["Revenue"]
                       .sum().nlargest(10).reset_index())
            fig_top = px.bar(top_rev, x="Revenue", y="Product",
                             orientation="h",
                             title="Top 10 Products by Revenue",
                             color_discrete_sequence=["#003399"])
            fig_top.update_layout(template="plotly_white",
                                  yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_top, use_container_width=True, config=_PLOTLY_CFG)

            # Lost sales by product
            top_lost = (df_stock_sc.groupby("Product")["LostSales"]
                        .sum().nlargest(10).reset_index())
            fig_lost = px.bar(top_lost, x="LostSales", y="Product",
                              orientation="h",
                              title="Top 10 Products by Lost Sales (Stockout + Price Refusal)",
                              color_discrete_sequence=["#dc143c"])
            fig_lost.update_layout(template="plotly_white",
                                   yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_lost, use_container_width=True, config=_PLOTLY_CFG)

    # -----------------------------------------------------------------------
    elif view == "📋 Policy Maker":
        st.subheader("📋 Policy Maker Dashboard — Welfare & Sustainability")
        st.markdown(
            "Focus: **food security, consumer affordability, CO₂ footprint, import dependency**. "
            "Key question: *Is the food system delivering equitable, sustainable outcomes?*"
        )

        last_rec = df_sc.iloc[-1] if not df_sc.empty else {}

        # Pull policy comparison data if available
        pol_base = st.session_state.get("policy_baseline")
        pol_scen = st.session_state.get("policy_scenario")

        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        col_w1.metric("Avg Budget Exhaustion",
                      f"{df_sc['BudgetExhaustionRate'].mean()*100:.1f}%")
        col_w2.metric("Avg Food Stress (low-income)",
                      f"{df_sc['FoodStressedPct'].mean()*100:.1f}%")
        col_w3.metric("Avg CO₂/day (kg)",
                      f"{df_sc['CO2Total'].mean():.1f}")
        col_w4.metric("Avg Import Dependency",
                      f"{df_sc['ImportDepPct'].mean():.1f}%")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_stress = px.area(df_sc, x="Day", y="FoodStressedPct",
                                 title="Food-Stressed Consumers (% daily)",
                                 color_discrete_sequence=["#e67e22"])
            fig_stress.update_layout(template="plotly_white",
                                     yaxis_tickformat=".0%")
            st.plotly_chart(fig_stress, use_container_width=True, config=_PLOTLY_CFG)

        with col_b:
            fig_co2 = px.area(df_sc, x="Day", y="CO2Total",
                              title="Daily CO₂ Footprint (kg CO₂-eq)",
                              color_discrete_sequence=["#27ae60"])
            fig_co2.update_layout(template="plotly_white")
            st.plotly_chart(fig_co2, use_container_width=True, config=_PLOTLY_CFG)

        col_c, col_d = st.columns(2)
        with col_c:
            fig_imp = px.line(df_sc, x="Day", y="ImportDepPct",
                              title="Import Dependency % over Time",
                              color_discrete_sequence=["#2980b9"])
            fig_imp.update_layout(template="plotly_white")
            st.plotly_chart(fig_imp, use_container_width=True, config=_PLOTLY_CFG)

        with col_d:
            fig_fat = px.line(df_sc, x="Day", y="MeanFatPurchased",
                              title="Mean Fat Content Purchased",
                              color_discrete_sequence=["#8e44ad"])
            fig_fat.update_layout(template="plotly_white",
                                  yaxis_title="Avg fat % per unit")
            st.plotly_chart(fig_fat, use_container_width=True, config=_PLOTLY_CFG)

        # Income vulnerability snapshot (from policy comparison if available)
        if pol_base is not None and pol_scen is not None:
            st.subheader("Income Vulnerability Snapshot (Policy vs Baseline)")
            brackets = ["Low", "Mid", "High"]
            vuln_data = []
            for br in brackets:
                b_val = pol_base[f"BudgetExh_{br}"].mean() * 100
                p_val = pol_scen[f"BudgetExh_{br}"].mean() * 100
                vuln_data.append({
                    "Income Group": f"{br} (<€1.5k / €1.5-3k / >€3k)",
                    "Baseline Budget Exh.%": round(b_val, 1),
                    "Policy Budget Exh.%":   round(p_val, 1),
                    "Δ (pp)": round(p_val - b_val, 2),
                })
            st.dataframe(pd.DataFrame(vuln_data), use_container_width=True, hide_index=True)
        else:
            st.info(
                "Run a policy comparison in the **🏛️ Policy Analysis** tab to see "
                "the income vulnerability snapshot here."
            )

    # -----------------------------------------------------------------------
    elif view == "🔬 Researcher":
        st.subheader("🔬 Researcher Dashboard — Full Metrics & Reproducibility")
        st.markdown(
            "Focus: **complete metric set, model configuration, reproducibility information**. "
            "Key question: *Is this simulation valid, reproducible, and scientifically rigorous?*"
        )

        # Reproducibility card
        with st.expander("📋 Model Reproducibility Card", expanded=True):
            cfg = st.session_state.config_data or {}
            stats = cfg.get("stats", {})
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown("**Model configuration**")
                st.json({
                    "mesa_version": "2.3.4",
                    "consumer_model": "utility-based (DCE-calibrated)",
                    "archetype_clustering": "K-Means (k=4)",
                    "population_pool_size": stats.get("pool_size", "N/A"),
                    "real_participants": stats.get("n_real", "N/A"),
                    "bootstrap_method": "stratified by archetype",
                    "shelf_model": "FIFO batches with near-expiry discount",
                    "learning": "adaptive preferences (rate=0.015/day)",
                })
            with col_r2:
                st.markdown("**Simulation parameters**")
                model_cris = st.session_state.get("sim_model_crisis")
                if model_cris:
                    st.json({
                        "days_simulated":    model_cris.current_day,
                        "base_consumers":    model_cris.base_consumers,
                        "reorder_point":     model_cris.reorder_point,
                        "target_stock":      model_cris.target_stock_level,
                        "lead_time_days":    model_cris.lead_time_days,
                        "scenario_start":    model_cris.scenario_start_day,
                        "inflation_pct":     model_cris.inflation_percent,
                        "disruption_days":   model_cris.supply_disruption_days,
                        "panic_sensitivity": model_cris.panic_sensitivity,
                        "fixed_seed":        model_cris.fixed_seed,
                    })
                else:
                    st.info("Run the simulation to populate configuration details.")

        # Full metrics table
        st.markdown("### Full Daily Metrics Table")
        display_cols = [c for c in df_sc.columns if c not in ("Scenario",)]
        st.dataframe(df_sc[display_cols].round(4), use_container_width=True)

        # Citation / methods note
        st.markdown("### Methods Summary (cite-ready)")
        st.markdown(
            """
            > **GROCERYsim ABM v2.0** is a Mesa-based agent-based model of dairy product retail.
            > Consumer agents are calibrated from discrete choice experiment (DCE) responses and
            > questionnaire data collected via a Unity-based task simulation. Agents are clustered
            > into four behavioural archetypes (price_champion, green_buyer, health_optimizer,
            > habitual_buyer) using K-Means (k=4) on eight preference dimensions. A stratified
            > bootstrap generates the synthetic population pool. Purchase decisions follow a
            > utility function: *U = origin_bonus + organic_bonus + fat_match − price_disutility*.
            > Preferences update each day via archetype-specific learning rules (rate = 0.015).
            > Supply chain uses FIFO shelf batches with near-expiry (50% off) discounting and
            > reorder-point replenishment. Policy levers (fat tax, subsidy, supply shock,
            > nutritional labelling) modify prices and delivery volumes. Environmental impact
            > is tracked via product-level CO₂ emission factors. Consumer welfare is measured
            > by budget exhaustion rate, food stress prevalence, and basket fulfillment.
            > SecureFood / Horizon Europe — grant agreement No. 101136583.
            """
        )

        st.download_button(
            "📥 Download Full Metrics CSV",
            df_sc.to_csv(index=False).encode("utf-8"),
            f"grocerysim_{sel_sc.lower()}_full_metrics.csv",
            "text/csv",
            key="sh_dl_full",
        )


# ===========================================================================
# 11d. TAB: SENSITIVITY ANALYSIS
# ===========================================================================

def render_sensitivity_tab(params: dict):
    st.header(_t("header_sensitivity"))
    st.markdown(
        "**One-at-a-time (OAT) parameter sweep** — vary each model parameter across its "
        "range while holding all others at their baseline value. The result shows which "
        "parameters most influence the chosen output metric (tornado chart). "
        "This is a lightweight alternative to Sobol indices for fast, interpretable results."
    )

    if st.session_state.config_data is None:
        st.warning("⚠️ Upload and process data in **🏠 Data & Population** first.")
        return

    # ---- Parameter definitions ----
    PARAM_DEFS = {
        "reorder_pt":    {"label": "Reorder Point",       "min": 0.10, "max": 0.60, "steps": 5, "key": "reorder"},
        "target_stock":  {"label": "Restock Target",      "min": 0.60, "max": 0.99, "steps": 5, "key": "target"},
        "lead_time":     {"label": "Lead Time (days)",     "min": 1,    "max": 10,   "steps": 5, "key": "lead"},
        "base_consumers":{"label": "Base Consumers/day",  "min": 20,   "max": 300,  "steps": 5, "key": "base_con"},
        "panic_sens":    {"label": "Panic Sensitivity",   "min": 0.10, "max": 0.90, "steps": 5, "key": "panic"},
        "inflation":     {"label": "Inflation % (crisis)","min": 0,    "max": 100,  "steps": 5, "key": "inf"},
        "fat_tax_rate":  {"label": "Fat Tax Rate",        "min": 0.0,  "max": 0.5,  "steps": 5, "key": None},
        "shock_severity":{"label": "Shock Severity",     "min": 0.0,  "max": 1.0,  "steps": 5, "key": None},
    }

    OUTPUT_METRICS = {
        "Revenue":               "Avg Daily Revenue (€)",
        "Waste":                 "Avg Daily Waste (units)",
        "CO2Total":              "Avg Daily CO₂ (kg)",
        "ImportDepPct":          "Avg Import Dependency %",
        "BudgetExhaustionRate":  "Avg Budget Exhaustion %",
        "FulfillmentRate":       "Avg Fulfillment %",
        "LostSales":             "Avg Lost Sales",
    }

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        sa_days = st.slider("Days per run", 14, 365, 60, key="sa_days")
    with col_cfg2:
        sa_metric = st.selectbox("Output metric", list(OUTPUT_METRICS.keys()),
                                  format_func=lambda k: OUTPUT_METRICS[k],
                                  key="sa_metric")
    with col_cfg3:
        sa_include_policy = st.checkbox("Include policy params", True, key="sa_pol",
                                        help="Include fat_tax_rate and shock_severity in the sweep")

    selected_params = {k: v for k, v in PARAM_DEFS.items()
                       if v["key"] is not None or sa_include_policy}

    st.markdown(f"Will vary **{len(selected_params)}** parameters × {PARAM_DEFS['reorder_pt']['steps']} levels "
                f"= **{len(selected_params)*5} simulation runs** of {sa_days} days each.")

    run_sa = st.button(_t("btn_run_sensitivity"), type="primary", key="sa_run_btn")

    if "sa_results" not in st.session_state:
        st.session_state.sa_results = None

    if run_sa:
        base_policy_cfg = params.get("policy_cfg", {})
        results = {}   # param_name → list of (param_value, metric_mean)

        total_runs = len(selected_params) * 5
        bar = st.progress(0, text="Starting…")
        run_counter = [0]

        def _run_one(override_params: dict, override_policy: dict = None) -> float:
            p = {**params, **override_params}
            m = _make_model(p, is_crisis=False, seed=42,
                            policy_cfg=override_policy or {})
            for _ in range(sa_days):
                m.step()
            vals = [r.get(sa_metric, 0) for r in m.daily_records]
            return float(np.mean(vals)) if vals else 0.0

        for pname, pdef in selected_params.items():
            lo, hi, steps = pdef["min"], pdef["max"], pdef["steps"]
            levels = np.linspace(lo, hi, steps).tolist()
            results[pname] = []
            for val in levels:
                override = {}
                override_pol = {}
                if pdef["key"]:
                    override[pdef["key"]] = val
                elif pname == "fat_tax_rate":
                    override_pol = {**base_policy_cfg,
                                    "fat_tax_active": True,
                                    "fat_tax_rate": val}
                elif pname == "shock_severity":
                    override_pol = {**base_policy_cfg,
                                    "domestic_shock_active": True,
                                    "domestic_shock_day": 10,
                                    "domestic_shock_duration": sa_days,
                                    "domestic_shock_severity": val}
                metric_val = _run_one(override, override_pol or None)
                results[pname].append((val, metric_val))
                run_counter[0] += 1
                bar.progress(run_counter[0] / total_runs,
                             text=f"Varying {pdef['label']} — {run_counter[0]}/{total_runs}")

        bar.empty()
        st.session_state.sa_results = results
        st.success(f"✅ Sensitivity analysis complete — {total_runs} runs.")

    if st.session_state.sa_results is None:
        st.info("Click **▶️ Run Sensitivity Analysis** to generate results.")
        return

    results = st.session_state.sa_results
    metric_label = OUTPUT_METRICS.get(sa_metric, sa_metric)

    # ---- Compute sensitivity index: (max − min) of metric across levels ----
    sensitivity = {}
    for pname, vals in results.items():
        metric_vals = [v for _, v in vals]
        sensitivity[pname] = max(metric_vals) - min(metric_vals)

    df_tornado = pd.DataFrame([
        {"Parameter": PARAM_DEFS[k]["label"], "Range (max-min)": v}
        for k, v in sorted(sensitivity.items(), key=lambda x: -abs(x[1]))
    ])

    st.subheader(f"🌪️ Tornado Chart — Effect on {metric_label}")
    fig_tor = px.bar(
        df_tornado, x="Range (max-min)", y="Parameter",
        orientation="h",
        title=f"Parameter Sensitivity: Range of {metric_label} across 5 levels",
        color="Range (max-min)",
        color_continuous_scale=["#2980b9", "#e74c3c"],
    )
    fig_tor.update_layout(
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_tor, use_container_width=True, config=_PLOTLY_CFG)

    st.caption(
        "Bar length = max(metric) − min(metric) as each parameter sweeps its range. "
        "Longer bars = higher influence on the output. Parameters at the top are the "
        "key levers for this metric."
    )

    # ---- Individual parameter response curves ----
    st.subheader("📈 Parameter Response Curves")
    n_params = len(results)
    cols_per_row = 3
    param_items = [(k, PARAM_DEFS[k]["label"]) for k in results]
    for row_start in range(0, n_params, cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, (pname, plabel) in enumerate(param_items[row_start:row_start+cols_per_row]):
            x_vals = [v for v, _ in results[pname]]
            y_vals = [m for _, m in results[pname]]
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="lines+markers",
                line=dict(color="#003399"), marker=dict(size=6),
            ))
            fig_curve.update_layout(
                title=plabel, template="plotly_white",
                xaxis_title=plabel, yaxis_title=metric_label,
                margin=dict(t=40, b=30),
            )
            cols[col_idx].plotly_chart(fig_curve, use_container_width=True, config=_PLOTLY_CFG)

    # ---- Download ----
    sa_rows = []
    for pname, vals in results.items():
        for pval, mval in vals:
            sa_rows.append({
                "Parameter": PARAM_DEFS[pname]["label"],
                "ParameterValue": pval,
                metric_label: mval,
            })
    st.download_button(
        "📥 Download Sensitivity Data (CSV)",
        pd.DataFrame(sa_rows).to_csv(index=False).encode("utf-8"),
        "sensitivity_analysis.csv",
        "text/csv",
        key="dl_sa",
    )


# ===========================================================================
# 12. MAIN ENTRY POINT
# ===========================================================================

def main():
    params = build_sidebar_params()
    render_onboarding_tour()

    _title_col, _right_col = st.columns([4, 3])
    with _title_col:
        st.title("🛒 GROCERYsim ABM v2.0")
        st.markdown(_t("subtitle"))
    with _right_col:
        _sf_logo_uri = _logo_uri("SecureFood.png")
        with st.container(border=True):
            st.markdown(
                f"<div style='text-align:center; padding:4px 0 8px 0;'>"
                f"<img src='{_sf_logo_uri}' style='height:48px; width:auto; object-fit:contain;'>"
                f"</div>"
                f"<style>"
                f"[data-testid='stVerticalBlockBorderWrapper'] [data-testid='stButton'] > button,"
                f"[data-testid='stVerticalBlockBorderWrapper'] [data-testid='stDownloadButton'] > button {{"
                f"  height: 42px !important;"
                f"  min-height: 42px !important;"
                f"  white-space: nowrap !important;"
                f"  overflow: hidden !important;"
                f"  text-overflow: ellipsis !important;"
                f"}}"
                f"</style>",
                unsafe_allow_html=True,
            )
            _btn_a, _btn_b = st.columns(2)
            with _btn_a:
                if st.button("🌿 Scenario Simulator", use_container_width=True, key="sf_launch_btn"):
                    st.session_state["page"] = "securefood"
                    st.rerun()
            with _btn_b:
                _pdf_path = os.path.join(_STATIC_DIR, "GROCERYsim_SecureFood_Scenario_Walkthrough_ClimateChange_Dairy.pdf")
                try:
                    with open(_pdf_path, "rb") as _f:
                        _pdf_bytes = _f.read()
                    st.download_button(
                        label=_t("securefood_btn"),
                        data=_pdf_bytes,
                        file_name="GROCERYsim_SecureFood_Scenario_Walkthrough_ClimateChange_Dairy.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="securefood_pdf_btn",
                    )
                except FileNotFoundError:
                    pass
    st.divider()

    tabs = st.tabs(_t("tabs"))

    with tabs[0]:
        render_data_tab()

    with tabs[1]:
        render_demo_tab(params)

    with tabs[2]:
        render_science_tab(params)

    with tabs[3]:
        render_waste_tab()

    with tabs[4]:
        render_product_tab()

    with tabs[5]:
        render_policy_tab(params)

    with tabs[6]:
        render_stakeholder_tab()

    with tabs[7]:
        render_sensitivity_tab(params)

    with tabs[8]:
        render_behaviour_tab(params)

    with tabs[9]:
        render_export_tab()

    render_footer()


if __name__ == "__main__":
    page = st.session_state.get("page", "landing")
    if page == "landing":
        render_landing_page()
    elif page == "case_studies":
        render_case_studies_page()
    elif page == "securefood":
        render_securefood_page()
    else:
        main()
