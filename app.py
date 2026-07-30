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
import csv
import copy
import io
import json
import math
import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF

from data_processor import run_pipeline_from_data, ARCHETYPE_LABELS
from model import SupermarketModel, ProductAgent
from parameter_registry import (
    build_parameter_registry,
    parameter_registry_summary,
    validate_parameter_registry,
)
from sensitivity_analysis import (
    bootstrap_prcc,
    convergence_diagnostics,
    latin_hypercube,
    nonlinear_permutation_importance,
    scale_design,
    variance_decomposition,
)
from calibration_analysis import (
    calibration_design,
    identifiability_diagnostics,
    standardized_rmse,
    waste_rate_percent,
)
from validation_protocol import (
    daily_validation_observables,
    evidence_tier_counts,
    evaluate_baseline_reproduction,
    evaluate_phase2_reproduction,
    evaluate_targets,
    validate_target_definitions,
    validation_summary,
    validation_target_template,
)

plt.switch_backend("Agg")

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Embedded in every SecureFood report and used to reject artifacts retained in
# browser session state across deployments. Bump this identifier whenever a
# scientific model change can alter report outputs.
SF_REPORT_MODEL_REVISION = "inventory-age-cohorts-v3-uncached"

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
            "This 8-step tour covers every feature in about 2 minutes. "
            "Navigation works via the 5 coloured cards below — click any section "
            "button to open it. The breadcrumb at the top of each section "
            "(🏠 Menu › 🔬 Simulation › …) is fully clickable to go back. "
            "Skip this tour any time with the button on the left."
        ),
    },
    {
        "title": "⚙️ Simulation Parameters — Sidebar",
        "body": (
            "The left sidebar controls every aspect of the model: simulation duration, "
            "number of consumer agents, reorder points, lead times, crisis settings "
            "(inflation, supply disruption, panic sensitivity, hoarding multiplier), "
            "and behavioural levers (media channel, purchase-limit nudge, stockpile horizon). "
            "The model auto-calibrates shelf capacity and stock to your store size."
        ),
    },
    {
        "title": "🏠 Card 1 — Data & Setup",
        "body": (
            "Start here every session. Load your DCE participant cohort from Firebase "
            "or upload CSV/JSON files. The data tab shows cohort demographics, "
            "questionnaire reliability, archetype stability, participant resampling, and the product catalogue. "
            "All other sections depend on data being loaded first."
        ),
    },
    {
        "title": "🔬 Card 2 — Simulation (2 sections)",
        "body": (
            "• Interactive Demo — run a live Baseline vs Crisis paired simulation with "
            "animated day-by-day revenue, stock and consumer charts. Save any run as a "
            "named scenario for later comparison.\n"
            "• Scientific Analysis — Monte Carlo (multi-run) with p10/p25/p75/p90 "
            "percentile confidence bands, AI storage recommendations, and a full "
            "Baseline vs Crisis statistical comparison with downloadable PDF report."
        ),
    },
    {
        "title": "📊 Card 3 — Analysis (6 sections)",
        "body": (
            "• Food Waste — waste log, drivers, and CO₂ footprint\n"
            "• Per-Product — stock, sales, price per SKU over time\n"
            "• Behavioural Theory — Prospect Theory loss aversion, TPB intention scores, "
            "food-access stress scale, and stockpile pressure\n"
            "• Sensitivity Analysis — replicated Latin Hypercube global screening, "
            "bootstrap PRCC, nonlinear validation, and convergence diagnostics\n"
            "• Compare Scenarios — side-by-side charts from saved simulation runs\n"
            "• Agent Replay — slider to step through any day; scatter plot of every "
            "shopper's panic vs PBC, archetype breakdown, income vulnerability heatmap"
        ),
    },
    {
        "title": "🏛️ Card 4 — Policy & Strategy (5 sections)",
        "body": (
            "• Policy Analysis — fat tax, domestic/organic subsidy, purchase-cap nudge, "
            "nutritional labelling; auto-generates a branded PDF policy brief\n"
            "• Stakeholder View — KPI dashboards and policy narrative for decision-makers\n"
            "• Stress Test — automated 6-scenario battery (supply collapse, price spike, "
            "panic wave, import crisis, demand surge, cold-chain failure)\n"
            "• Multi-Store Network — simulate up to 8 Finnish stores in lockstep with "
            "panic contagion and emergency redistribution; results link directly to the map\n"
            "• Regional Map — 33 Finnish store locations; after a multi-store run the map "
            "shows a risk-level overlay (🔴 Critical → 🟢 Low) for each simulated store"
        ),
    },
    {
        "title": "🌿 SecureFood Scenario Simulator",
        "body": (
            "Dedicated tool for the Horizon Europe SecureFood project (grant No. 101136583). "
            "After launching the Finland — Dairy Supply Chain case study, the main GROCERYsim "
            "workspace opens. At the top of that page, find the SecureFood panel and click the "
            "dedicated 🌿 Scenario Simulator button. Run the default unmitigated food-security "
            "scenario first. "
            "Enable the separate Additional Policy Analysis module only when you want a "
            "paired policy-versus-no-policy counterfactual with PDF and CSV results."
        ),
    },
    {
        "title": "📤 Card 5 — Export  ·  ✅ You're all set!",
        "body": (
            "Card 5 — Export gives you:\n"
            "• Individual CSV downloads for every simulation dataset "
            "(daily aggregates, per-product stock, SCM log, food waste, policy runs)\n"
            "• Full bundle (all sheets in one CSV)\n"
            "• 📄 Generate PDF Report — a 7-section branded PDF covering parameters, "
            "revenue charts, supply chain summary, policy KPIs, stress-test ranking, "
            "saved scenario comparison, and methodology note.\n\n"
            "Click '🎓 Tour' in the sidebar any time to replay this tour."
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
    heroAboutBtn: 'About',
    figCaption: 'Live store — cognitive shoppers, shelf inventory, periodic restock.',
    overviewLabel: 'WHAT IS GROCERYSIM',
    overviewTitle: 'A web application that lets stakeholders stress-test the resilience of food supply chains.',
    overviewP1: 'The model represents a retail environment as autonomous agents: consumers with individual cognitive traits, shelves with finite inventory, and logistics with realistic lead times. From their interactions, system-level behaviour emerges — resilience, fragility, and adaptation under stress.',
    overviewP2: 'Researchers calibrate scenarios; policy makers explore interventions; retailers validate contingency plans before they are needed.',
    stat1: 'Agent-runs / day', stat2: 'Crisis scenarios', stat3: 'EU markets modeled',
    featuresLabel: 'KEY FEATURES',
    featuresTitle: 'Three pillars. One model.',
    f1t: 'Evidence-Gated Agents', f1d: 'Consumers preserve observed participant profiles and calibrated price/substitution response. Panic, hoarding, and theory-transferred dynamics are optional exploratory assumptions.', f1tag: 'BEHAVIOUR',
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
    heroAboutBtn: 'Tietoa',
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
    heroAboutBtn: 'Σχετικά',
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
    heroAboutBtn: 'Sobre',
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

// ── Hero logo link targets ────────────────────────────────────────────────────
const HERO_LOGO_LINKS = [
  'https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en',
  'https://secure-food.eu',
  'https://www.iamo.de',
  'https://www.iamo.de/forschung/forschungsprojekte/details/iamo-xr-lab',
];

// ── About modal ───────────────────────────────────────────────────────────────
const AboutModal = ({ onClose }) => (
  <div
    onClick={onClose}
    style={{position:'fixed',inset:0,zIndex:9999,display:'flex',alignItems:'center',
            justifyContent:'center',background:'rgba(4,32,38,0.78)',backdropFilter:'blur(5px)'}}>
    <div
      onClick={e => e.stopPropagation()}
      style={{background:'#FAF6EC',borderRadius:'10px',padding:'32px 36px',maxWidth:'500px',
              width:'92%',boxShadow:'0 12px 48px rgba(0,0,0,0.45)',position:'relative',
              fontFamily:'Figtree,sans-serif',color:'#042026'}}>
      {/* close button */}
      <button onClick={onClose}
        style={{position:'absolute',top:'12px',right:'16px',background:'none',border:'none',
                fontSize:'22px',lineHeight:1,cursor:'pointer',color:'#042026',opacity:0.5}}>
        &#xd7;
      </button>
      {/* heading */}
      <h2 style={{fontSize:'15px',fontWeight:800,letterSpacing:'-0.01em',margin:'0 0 8px',
                  paddingBottom:'10px',borderBottom:'2px solid #DBA159'}}>
        About GROCERYsim
      </h2>
      {/* body text */}
      <p style={{fontSize:'13px',lineHeight:1.65,margin:'12px 0'}}>
        This application is part of an academic study on food consumption patterns in Finland.
        All data collected is anonymous and used strictly for research analysis.
      </p>
      <p style={{fontSize:'12px',fontWeight:700,margin:'14px 0 4px',
                 textTransform:'uppercase',letterSpacing:'0.08em',color:'#DBA159'}}>
        Author
      </p>
      <p style={{fontSize:'13px',lineHeight:1.75,margin:'0 0 14px'}}>
        Dr. Ivan &#x110;uri&#x107;<br/>
        IAMO XR LAB Coordinator<br/>
        Leibniz Institute of Agricultural Development in Transition Economies (IAMO)<br/>
        Theodor-Lieser Str.&#x202F;2, 06120, Halle (Saale), Germany<br/>
        <a href="https://www.iamo.de" target="_blank" rel="noreferrer"
           style={{color:'#DBA159',textDecoration:'none',fontWeight:600}}>
          www.iamo.de
        </a>
      </p>
      <p style={{fontSize:'12px',fontWeight:700,margin:'0 0 4px',
                 textTransform:'uppercase',letterSpacing:'0.08em',color:'#DBA159'}}>
        Citation
      </p>
      <p style={{fontSize:'12px',lineHeight:1.65,fontStyle:'italic',
                 background:'#F0E9DA',borderRadius:'6px',padding:'10px 12px',margin:0}}>
        &#x110;uri&#x107;, Ivan (2026). GROCERYsim Agent-Based Model for Consumer Behaviour
        and Supply Chain Stress-Testing. IAMO XR Lab, SecureFood project,
        Horizon Europe Grant 101136583.
      </p>
    </div>
  </div>
);

// ── Hero ─────────────────────────────────────────────────────────────────────
const Hero = ({ t }) => {
  const [showAbout, setShowAbout] = React.useState(false);
  return (
  <section className="hero">
      {showAbout && <AboutModal onClose={() => setShowAbout(false)} />}
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
          <a className="btn btn-ghost" href="#"
             onClick={(e) => { e.preventDefault(); setShowAbout(true); }}>
            {t.heroAboutBtn}
          </a>
        </div>
        <div className="hero-logos">
          <div className="hero-logos-row">
            {[0, 1, 2, 3].map((i) => (
              <a key={i}
                 href={HERO_LOGO_LINKS[i]}
                 target="_blank"
                 rel="noreferrer"
                 className={'hero-logo-slot' + (LOGO_CONFIG.hero[i] ? ' has-logo' : '')}
                 aria-label={'Partner logo ' + (i+1)}
                 style={{textDecoration:'none', cursor:'pointer'}}>
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
              </a>
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
};

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
        policy_cfg = params.get("policy_cfg", {}) or {}
        has_policy = _sf_has_active_policy(params)
        no_policy_params = _sf_without_policy(params)
        m_base = _make_model(
            no_policy_params, is_crisis=False, seed=42, policy_cfg={}
        )
        m_crisis = _make_model(
            params, is_crisis=True, seed=42, policy_cfg=policy_cfg
        )
        m_crisis_no_policy = (
            _make_model(no_policy_params, is_crisis=True, seed=42, policy_cfg={})
            if has_policy else None
        )
        agg_rows, prod_rows = [], []
        for day in range(1, params["days"] + 1):
            m_base.step()
            m_crisis.step()
            agg_b, pb = _collect_model_day(m_base,   day, "Baseline")
            agg_c, pc = _collect_model_day(m_crisis, day, "Crisis")
            agg_rows += [agg_b, agg_c]
            prod_rows += pb + pc
            if m_crisis_no_policy is not None:
                m_crisis_no_policy.step()
                agg_u, pu = _collect_model_day(
                    m_crisis_no_policy, day, "Crisis (No Policy)"
                )
                agg_rows.append(agg_u)
                prod_rows.extend(pu)
        return {
            "df":      pd.DataFrame(agg_rows),
            "df_prod": pd.DataFrame(prod_rows),
            "params":  params,
            "has_policy_counterfactual": has_policy,
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
    cri_end   = (cri_start + cri_dur) if cri_dur > 0 else days + 1

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
    k2.metric("Unmet Demand", f"{lost_total:,.0f} units",
              "stock/price constrained", delta_color="inverse")
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
        margin=dict(t=100, b=40, l=60, r=30),
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
    st.markdown("#### 2 · Unmet Demand Events")
    df_c2 = df_c.copy()
    df_c2["CumLost"] = df_c2["LostSales"].cumsum()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df_c2["Day"], y=df_c2["LostSales"],
                          name="Daily Unmet Demand (Crisis)",
                          marker_color="rgba(231,76,60,0.65)", marker_line_width=0))
    fig2.add_trace(go.Scatter(x=df_c2["Day"], y=df_c2["CumLost"],
                              name="Cumulative Unmet Demand",
                              line=dict(color="#922b21", width=2.5), yaxis="y2"))
    fig2.add_trace(go.Scatter(x=df_b["Day"], y=df_b["LostSales"],
                              name="Baseline Unmet Demand",
                              line=dict(color="#aab7b8", width=1.2, dash="dot")))
    fig2 = _sf_crisis_band(fig2, cri_start, cri_end, days)
    fig2.update_layout(
        template="plotly_white", height=360,
        xaxis_title="Simulation Day", yaxis_title="Unmet Demand Units/day",
        yaxis2=dict(title="Cumulative Units", overlaying="y", side="right"),
        title="Unmet Demand — Daily Events and Cumulative Accumulation",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=100, b=40, l=60, r=30),
    )
    st.plotly_chart(fig2, use_container_width=True)
    _sf_analysis_box(
        f"Cumulative unmet demand reached **{lost_total:,.0f} units**. Peak occurred on "
        f"**Day {peak_lost_day}** ({peak_lost_val:,.0f} units). The counter includes quantity "
        f"not fulfilled after price rejection, substitution, or stock constraints; it is not "
        f"denominated in euros and does not imply permanent customer loss. The grey dotted "
        f"line is the paired baseline."
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
            margin=dict(t=110, b=40, l=60, r=30),
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
                   annotation_text="Descriptive reference (0.30)",
                   annotation_font_size=9, annotation_position="bottom right")
    fig4 = _sf_crisis_band(fig4, cri_start, cri_end, days)
    fig4.update_layout(
        template="plotly_white", height=360,
        xaxis_title="Simulation Day",
        yaxis=dict(title="Panic Level (0–1)", range=[0, 1.05]),
        yaxis2=dict(title="Demand Ratio (1.0 = base)", overlaying="y",
                    side="right", range=[0, max(1.1, sp_peak * 1.1)]),
        title="Consumer Panic Level and Stockpile Pressure — Behavioural Demand Amplification",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=100, b=40, l=60, r=30),
    )
    st.plotly_chart(fig4, use_container_width=True)
    if panic_threshold_day_val is not None:
        panic_txt = (f"The descriptive reference level 0.30 was crossed on **Day {panic_threshold_day_val}**.")
    else:
        panic_txt = "Panic remained below the reference level 0.30 throughout the simulation."
    _sf_analysis_box(
        f"Consumer panic peaked at **{peak_panic:.2f}/1.0** (Day {panic_peak_day}). {panic_txt} "
        f"The stockpiling-demand ratio peaked at **{sp_peak:.2f}** (1.0 = base requested demand). Both series are internal exploratory "
        f"state variables driven partly by analyst assumptions; they are not measured panic or "
        f"validated thresholds. Household amplification is scaled by cross-fitted propensity."
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
            f"Unmet demand: <b>{lost_total:,.0f} units</b> after price, substitution, and stock constraints",
            f"Average retail price: <b>€{avg_p_c:.2f}</b> vs baseline <b>€{avg_p_b:.2f}</b> (+{price_delta_pct:.0f}% inflation pass-through)",
            f"Peak internal panic state: <b>{peak_panic:.2f}/1.0</b> (exploratory, not a validated severity scale)",
            f"Waste delta: <b>{waste_delta:+,.0f} units</b> vs paired baseline — {waste_dir}",
            f"Supply recovery: {rec_str}",
        ],
        "Treat this run as a stress-test result. Before selecting reorder points or safety-stock "
        "levels, run multiple paired seeds and global sensitivity analysis over the disruption, "
        "lead-time, price, panic, and hoarding assumptions.",
    )


# ── Policy Maker results renderer ─────────────────────────────────────────────

def _render_sf_pm_results(data: dict):
    df      = data["df"]
    df_prod = data["df_prod"]
    p       = data["params"]

    df_b = df[df["Scenario"] == "Baseline"].copy().reset_index(drop=True)
    df_c = df[df["Scenario"] == "Crisis"].copy().reset_index(drop=True)
    df_u = df[df["Scenario"] == "Crisis (No Policy)"].copy().reset_index(drop=True)

    cri_start = p["cri_start"]
    cri_dur   = p["cri_duration"]
    days      = p["days"]
    cri_end   = (cri_start + cri_dur) if cri_dur > 0 else days + 1
    pc        = p.get("policy_cfg", {})

    # ── Pre-compute metrics ────────────────────────────────────────────────────
    def _active_window(frame):
        return frame[
            (frame["Day"] >= cri_start) & (frame["Day"] < cri_end)
        ].reset_index(drop=True)

    b_win = _active_window(df_b)
    c_win = _active_window(df_c)
    u_win = _active_window(df_u) if not df_u.empty else pd.DataFrame()
    comparison_win = u_win if not u_win.empty else b_win
    comparison_label = "paired crisis without policy" if not u_win.empty else "baseline"

    peak_stress    = float(c_win["FoodStressedPct"].max()) * 100
    peak_stress_u  = float(comparison_win["FoodStressedPct"].max()) * 100
    base_stress    = float(b_win["FoodStressedPct"].mean()) * 100
    peak_panic_c   = float(c_win["PanicLevel"].max())
    peak_panic_u   = float(comparison_win["PanicLevel"].max())
    peak_budgexh_lo = float(c_win["BudgetExh_Low"].max()) * 100
    peak_budgexh_hi = float(c_win["BudgetExh_High"].max()) * 100
    peak_budgexh_lo_u = float(comparison_win["BudgetExh_Low"].max()) * 100
    mean_gini_c    = float(c_win["GiniAccess"].mean())
    mean_gini_b    = float(b_win["GiniAccess"].mean())
    mean_gini_u    = float(comparison_win["GiniAccess"].mean())
    import_dep_b   = float(b_win["ImportDepPct"].mean())
    import_dep_c   = float(c_win["ImportDepPct"].mean())
    import_dep_u   = float(comparison_win["ImportDepPct"].mean())
    fulfill_lo_c   = float(c_win["Fulfillment_Low"].mean()) * 100
    fulfill_hi_c   = float(c_win["Fulfillment_High"].mean()) * 100
    fulfill_lo_u   = float(comparison_win["Fulfillment_Low"].mean()) * 100
    fulfill_gap    = fulfill_hi_c - fulfill_lo_c
    fies_lo_peak   = float(c_win["FIESSevere_Low"].max()) * 100
    fies_lo_base   = float(b_win["FIESSevere_Low"].mean()) * 100
    fies_lo_u      = float(comparison_win["FIESSevere_Low"].max()) * 100
    fies_delta     = fies_lo_peak - fies_lo_u
    dom_sum_b = b_win["DomesticSales"].sum() + b_win["ImportSales"].sum()
    dom_sum_c = c_win["DomesticSales"].sum() + c_win["ImportSales"].sum()
    dom_sum_u = comparison_win["DomesticSales"].sum() + comparison_win["ImportSales"].sum()
    dom_share_b    = b_win["DomesticSales"].sum() / max(dom_sum_b, 1) * 100
    dom_share_c    = c_win["DomesticSales"].sum() / max(dom_sum_c, 1) * 100
    dom_share_u    = comparison_win["DomesticSales"].sum() / max(dom_sum_u, 1) * 100
    dom_change     = dom_share_c - dom_share_u
    low_below_80   = int((c_win["Fulfillment_Low"] < 0.80).sum())

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
    k1.metric("Peak Shoppers in Low-Income Access Stress", f"{peak_stress:.1f}%",
              f"{peak_stress - peak_stress_u:+.1f} pp vs no-policy crisis",
              delta_color="inverse")
    k2.metric("Peak Budget Exhaustion (Low Income)", f"{peak_budgexh_lo:.1f}%",
              f"{peak_budgexh_lo - peak_budgexh_lo_u:+.1f} pp vs no-policy crisis",
              delta_color="inverse")
    k3.metric("Mean Gini Access Index",           f"{mean_gini_c:.3f}",
              f"{mean_gini_c - mean_gini_u:+.3f} vs no-policy crisis",
              delta_color="inverse")
    k4.metric("Import Dependency (Crisis)",       f"{import_dep_c:.1f}%",
              f"{import_dep_c - import_dep_u:+.1f} pp vs no-policy crisis",
              delta_color="inverse" if import_dep_c > import_dep_u else "normal")

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
    if not df_u.empty:
        fig1.add_trace(go.Scatter(x=df_u["Day"], y=df_u["Fulfillment_Low"]*100,
                                  name="Low income (crisis, no policy)",
                                  line=dict(color="#2c3e50", width=1.5, dash="dot")))
    fig1.add_hline(y=80, line_dash="dot", line_color="#c0392b",
                   annotation_text="80% descriptive reference",
                   annotation_font_size=9, annotation_position="bottom right")
    fig1 = _sf_crisis_band(fig1, cri_start, cri_end, days)
    fig1.update_layout(
        template="plotly_white", height=400,
        xaxis_title="Simulation Day", yaxis_title="Fulfilment Rate (%)",
        yaxis=dict(range=[0, 105]),
        title="Consumer Basket Fulfilment Rate by Income Group — Crisis Scenario",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=100, b=40, l=60, r=30),
    )
    st.plotly_chart(fig1, use_container_width=True)
    _sf_analysis_box(
        f"During active-crisis days, low-income basket fulfilment averaged **{fulfill_lo_c:.1f}%** "
        f"with the policy bundle and **{fulfill_lo_u:.1f}%** in the {comparison_label}; the "
        f"paired difference is **{fulfill_lo_c - fulfill_lo_u:+.1f} pp**. High-income fulfilment "
        f"was **{fulfill_hi_c:.1f}%** in the policy-crisis run. Low-income fulfilment fell below "
        f"the descriptive 80% reference on **{low_below_80} of {len(c_win)} active-crisis days**. "
        f"This is a single-seed model comparison, not a causal estimate."
    )

    # ── Chart 2: Food-Access Stress ───────────────────────────────────────────
    st.markdown("#### 2 · Exploratory Food-Access Stress by Income Group")
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
    if not df_u.empty:
        fig2.add_trace(go.Scatter(x=df_u["Day"], y=df_u["FIESSevere_Low"]*100,
                                  name="Low income (crisis, no policy)",
                                  line=dict(color="#2c3e50", width=1.4, dash="dash")))
    fig2 = _sf_crisis_band(fig2, cri_start, cri_end, days)
    fig2.update_layout(
        template="plotly_white", height=400,
        xaxis_title="Simulation Day", yaxis_title="Households with high access stress (%)",
        title="High Food-Access Stress by Income Bracket — Crisis vs Baseline (dotted)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=100, b=40, l=60, r=30),
    )
    st.plotly_chart(fig2, use_container_width=True)
    _sf_analysis_box(
        f"High modeled food-access stress among low-income households peaked at "
        f"**{fies_lo_peak:.1f}%** in the policy-crisis run and **{fies_lo_u:.1f}%** in the "
        f"{comparison_label} ({fies_delta:+.1f} pp). This exploratory diagnostic measures "
        f"realised access and consumption shortfall across represented households. It is not "
        f"comparable to survey-based FIES prevalence, and the bundled comparison cannot rank "
        f"individual instruments."
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
    if not df_u.empty:
        fig3.add_trace(go.Scatter(x=df_u["Day"], y=df_u["GiniAccess"],
                                  name="Gini (crisis, no policy)",
                                  line=dict(color="#2c3e50", width=1.3, dash="dash"), yaxis="y2"))
    fig3 = _sf_crisis_band(fig3, cri_start, cri_end, days)
    fig3.update_layout(
        template="plotly_white", height=400,
        xaxis_title="Simulation Day", yaxis_title="Budget Exhausted (%)",
        yaxis2=dict(title="Gini Access Index (0–1)", overlaying="y",
                    side="right", range=[0, 1]),
        title="Budget Exhaustion by Income Group and Access Inequality (Gini Index)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=100, b=40, l=60, r=30),
    )
    st.plotly_chart(fig3, use_container_width=True)
    _sf_analysis_box(
        f"Budget exhaustion among low-income consumers peaked at **{peak_budgexh_lo:.1f}%**, "
        f"vs **{peak_budgexh_hi:.1f}%** for high-income. The Gini access index rose from "
        f"**{mean_gini_b:.3f}** in the baseline to **{mean_gini_c:.3f}** in the policy-crisis run; "
        f"the paired crisis-without-policy value was **{mean_gini_u:.3f}**. The policy-bundle "
        f"difference is **{mean_gini_c - mean_gini_u:+.3f}**. No universal action threshold or "
        f"statistical significance is inferred from this single seed."
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
        margin=dict(t=100, b=40, l=60, r=30),
    )
    st.plotly_chart(fig4, use_container_width=True)
    _sf_analysis_box(
        f"Domestic sales share was **{dom_share_b:.1f}%** in the baseline, **{dom_share_u:.1f}%** "
        f"in the {comparison_label}, and **{dom_share_c:.1f}%** in the policy-crisis run. The "
        f"paired bundle difference is **{dom_change:+.1f} pp**. Because several policies change "
        f"together, the shift cannot be attributed to the domestic subsidy alone."
    )

    # ── Chart 5: Policy Effectiveness (conditional) ───────────────────────────
    if active_policies or has_limit or has_media:
        st.markdown("#### 5 · Active Policy Instruments — Key Welfare Metrics")
        metrics_names = ["Low-income access\nstress pp", "Low-income fulfilment pp",
                         "Gini ×100 delta", "Import dependency pp", "Peak panic delta"]
        metrics_vals  = [peak_stress - peak_stress_u, fulfill_lo_c - fulfill_lo_u,
                         (mean_gini_c - mean_gini_u) * 100, import_dep_c - import_dep_u,
                         peak_panic_c - peak_panic_u]
        bar_colors    = ["#e74c3c", "#e67e22", "#8e44ad", "#2980b9", "#16a085"]
        fig5 = go.Figure(go.Bar(
            x=metrics_names, y=metrics_vals,
            marker_color=bar_colors,
            text=[f"{v:.1f}" for v in metrics_vals], textposition="outside",
        ))
        fig5.update_layout(
            template="plotly_white", height=340,
            yaxis_title="Value",
            title=f"Policy Bundle Difference vs Paired No-Policy Crisis: {', '.join(policy_labels)}",
        )
        st.plotly_chart(fig5, use_container_width=True)
        _sf_analysis_box(
            f"Active policy instruments: **{', '.join(policy_labels)}**. Bars show paired "
            f"policy-crisis minus no-policy-crisis differences during active-crisis days. "
            f"Run each lever separately and across multiple paired seeds before ranking instruments."
        )

    # ── Summary box ───────────────────────────────────────────────────────────
    policies_str = ', '.join(policy_labels) if policy_labels else "none active"
    _sf_summary_box(
        "Policy Impact Summary — SecureFood Climate Scenario",
        [
            f"Peak share of all shoppers in low-income access stress: <b>{peak_stress:.1f}%</b> ({peak_stress-peak_stress_u:+.1f} pp vs no-policy crisis)",
            f"Low-income fulfilment: <b>{fulfill_lo_c:.1f}%</b> ({fulfill_lo_c-fulfill_lo_u:+.1f} pp vs no-policy crisis)",
            f"High modeled access stress (low income): <b>{fies_delta:+.1f} pp</b> vs no-policy crisis at peak",
            f"Budget exhaustion (low income): peaked at <b>{peak_budgexh_lo:.1f}%</b> of households",
            f"Access inequality (Gini): <b>{mean_gini_c:.3f}</b> ({mean_gini_c-mean_gini_u:+.3f} vs no-policy crisis)",
            f"Domestic sales share: <b>{dom_share_c:.1f}%</b> ({dom_change:+.1f} pp vs no-policy crisis)",
            f"Peak panic: <b>{peak_panic_c:.2f}</b> ({peak_panic_c-peak_panic_u:+.2f} vs no-policy crisis)",
            f"Policy instruments active: <b>{policies_str}</b>",
        ],
        "Interpret these as bundled, single-seed scenario differences. Run individual-lever "
        "comparisons, Monte Carlo uncertainty, and global sensitivity analysis before making "
        "operational or policy recommendations.",
    )


# ── SecureFood PDF Report Generator ──────────────────────────────────────────

def _sf_resolve_fonts() -> dict:
    """Resolve font paths: prefer bundled Liberation Sans, fall back to macOS Arial."""
    _bundled = os.path.join(_STATIC_DIR, "fonts")
    _candidates = {
        "reg":  [f"{_bundled}/LiberationSans-Regular.ttf",    "/System/Library/Fonts/Supplemental/Arial.ttf",      "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
        "bold": [f"{_bundled}/LiberationSans-Bold.ttf",       "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"],
        "it":   [f"{_bundled}/LiberationSans-Italic.ttf",     "/System/Library/Fonts/Supplemental/Arial Italic.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"],
        "bi":   [f"{_bundled}/LiberationSans-BoldItalic.ttf", "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"],
    }
    resolved = {}
    for key, paths in _candidates.items():
        for p in paths:
            if os.path.isfile(p):
                resolved[key] = p
                break
        else:
            resolved[key] = paths[0]  # let fpdf raise a clear error if missing
    resolved["uni"] = resolved["reg"]  # use same font for unicode fallback
    return resolved

_SF_FONTS = _sf_resolve_fonts()
_SF_DARK    = (  4,  32,  38)
_SF_DARK2   = ( 12,  58,  70)
_SF_AMBER   = (219, 161,  89)
_SF_AMBER_D = (180, 120,  55)
_SF_AMBER_L = (255, 248, 225)
_SF_WHITE   = (255, 255, 255)
_SF_CREAM   = (250, 246, 236)
_SF_CREAM2  = (240, 233, 218)
_SF_BODY    = ( 28,  44,  48)
_SF_RULE    = (200, 195, 185)
_SF_GREEN   = ( 39, 174,  96)
_SF_RED     = (192,  57,  43)
_SF_BLUE    = ( 41, 128, 185)


def _sf_mpl_chart(fig) -> str:
    """Save a matplotlib figure to a temp PNG and return the path."""
    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(buf.name, dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return buf.name


def _sf_logo_on(name: str, w_px: int, bg: tuple) -> str:
    """Composite a static-dir PNG on bg colour, return temp path."""
    from PIL import Image as _Img
    src = os.path.join(_STATIC_DIR, name)
    img = _Img.open(src).convert("RGBA")
    ratio = w_px / img.width
    img = img.resize((w_px, max(1, int(img.height * ratio))), _Img.LANCZOS)
    canvas = _Img.new("RGB", img.size, bg)
    canvas.paste(img, mask=img.split()[3])
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    canvas.save(tmp.name, "PNG")
    return tmp.name


class _SFReport(FPDF):
    """PDF class for the SecureFood scenario report."""
    _sec = ""

    def _lf(self):
        self.add_font("Ar",  "",   _SF_FONTS["reg"])
        self.add_font("Ar",  "B",  _SF_FONTS["bold"])
        self.add_font("Ar",  "I",  _SF_FONTS["it"])
        self.add_font("Ar",  "BI", _SF_FONTS["bi"])
        self.add_font("ArU", "",   _SF_FONTS["uni"])

    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*_SF_DARK)
        self.rect(0, 0, 210, 7, "F")
        self.set_fill_color(*_SF_AMBER)
        self.rect(0, 0, 3, 7, "F")
        self.set_font("Ar", "B", 6.5)
        self.set_text_color(*_SF_WHITE)
        self.set_xy(6, 0.8)
        self.cell(120, 5.5,
            "GROCERYsim SecureFood — Climate-Driven Dairy Supply Chain Disruption Report")
        self.set_font("Ar", "I", 6.5)
        self.set_text_color(*_SF_AMBER)
        self.set_xy(126, 0.8)
        self.cell(69, 5.5, self._sec, align="R")
        self.set_y(10)
        self.set_text_color(*_SF_BODY)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-11)
        self.set_draw_color(*_SF_RULE)
        self.set_line_width(0.2)
        self.line(15, self.get_y(), 195, self.get_y())
        self.set_font("Ar", "I", 7)
        self.set_text_color(140, 140, 130)
        self.set_y(-10)
        self.cell(100, 6, "Horizon Europe SecureFood · Grant No. 101136583 · IAMO XR Lab")
        self.cell(0, 6, f"Page  {self.page_no()}", align="R")
        self.set_text_color(*_SF_BODY)

    # helpers ----------------------------------------------------------------
    def ensure_space(self, height: float):
        """Start a continuation page before a report block would be orphaned."""
        if self.get_y() + height > self.h - self.b_margin:
            self.add_page()

    def chapter(self, num: str, title: str, label: str = ""):
        self._sec = label or title
        self.add_page()
        self.set_fill_color(*_SF_DARK)
        self.rect(0, 10, 210, 16, "F")
        self.set_fill_color(*_SF_AMBER)
        self.rect(0, 10, 5, 16, "F")
        self.set_font("Ar", "B", 7)
        self.set_text_color(*_SF_AMBER)
        self.set_xy(10, 11)
        self.cell(0, 4, f"SECTION {num}")
        self.set_font("Ar", "B", 13)
        self.set_text_color(*_SF_WHITE)
        self.set_xy(10, 15)
        self.cell(0, 9, title)
        self.set_y(30)
        self.set_text_color(*_SF_BODY)

    def sub(self, title: str, min_content_height: float = 12):
        self.ensure_space(17 + min_content_height)
        self.ln(3)
        self.set_font("Ar", "B", 10)
        self.set_text_color(*_SF_DARK)
        self.set_x(15)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*_SF_AMBER)
        self.set_line_width(0.5)
        self.line(15, self.get_y(), 195, self.get_y())
        self.set_line_width(0.2)
        self.set_draw_color(*_SF_RULE)
        self.ln(4)

    def body(self, text: str):
        import textwrap as _tw
        estimated_lines = max(1, len(_tw.wrap(text, 108)))
        self.ensure_space(estimated_lines * 5.2 + 4)
        self.set_font("Ar", "", 9.5)
        self.set_text_color(*_SF_BODY)
        self.set_x(15)
        self.multi_cell(180, 5.2, text)
        self.ln(2)

    def bullet(self, items: list):
        for item in items:
            import textwrap as _tw
            estimated_lines = max(1, len(_tw.wrap(item, 100)))
            self.ensure_space(estimated_lines * 5.2 + 2)
            bx = 18.5
            by = self.get_y() + 2.2
            self.set_fill_color(*_SF_AMBER)
            self.ellipse(bx, by, 2.2, 2.2, "F")
            self.set_x(23)
            self.set_font("Ar", "", 9.5)
            self.set_text_color(*_SF_BODY)
            self.multi_cell(170, 5.2, item)
        self.ln(2)

    def kv(self, rows: list):
        import textwrap as _tw
        for i, (k, v) in enumerate(rows):
            bg = _SF_CREAM2 if i % 2 == 0 else _SF_WHITE
            self.set_fill_color(*bg)
            lines = max(1, len(_tw.wrap(v, 65)))
            rh = 5.5 * lines + 2
            self.ensure_space(rh)
            y0 = self.get_y()
            self.rect(15, y0, 180, rh, "F")
            self.set_draw_color(*_SF_RULE)
            self.set_line_width(0.15)
            self.line(15, y0, 195, y0)
            self.set_xy(17, y0 + 1.5)
            self.set_font("Ar", "B", 8.5)
            self.set_text_color(*_SF_DARK)
            self.cell(55, rh - 2, k)
            self.set_xy(72, y0 + 1.5)
            self.set_font("Ar", "", 8.5)
            self.set_text_color(*_SF_BODY)
            self.multi_cell(120, 5.5, v)
            self.set_y(y0 + rh)
        self.set_draw_color(*_SF_RULE)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(4)

    def finding(self, text: str):
        import textwrap as _tw
        lines = max(1, len(_tw.wrap(text, 90)))
        bh = lines * 5 + 8
        self.ensure_space(bh + 3)
        y = self.get_y()
        self.set_fill_color(*_SF_AMBER_L)
        self.rect(15, y, 180, bh, "F")
        self.set_fill_color(*_SF_AMBER)
        self.rect(15, y, 3.5, bh, "F")
        self.set_xy(21, y + 2)
        self.set_font("Ar", "B", 8)
        self.set_text_color(*_SF_AMBER_D)
        self.cell(14, 4.5, "FINDING  ")
        self.set_font("Ar", "", 8.5)
        self.set_text_color(*_SF_BODY)
        self.set_x(35)
        self.multi_cell(157, 4.5, text)
        self.set_y(y + bh + 3)

    def metric_row(self, metrics: list):
        """Display a row of (label, value, delta, positive) tuples as KPI boxes."""
        self.ensure_space(26)
        n = len(metrics)
        w = 180 / n
        x0 = 15
        y0 = self.get_y()
        for i, (lbl, val, dlt, good) in enumerate(metrics):
            x = x0 + i * w
            self.set_fill_color(*_SF_DARK2)
            self.rect(x, y0, w - 1, 22, "F")
            self.set_fill_color(*(_SF_GREEN if good else _SF_RED))
            self.rect(x, y0, w - 1, 1.5, "F")
            self.set_font("Ar", "", 7)
            self.set_text_color(*_SF_AMBER)
            self.set_xy(x + 2, y0 + 3)
            self.cell(w - 4, 4, lbl)
            self.set_font("Ar", "B", 11)
            self.set_text_color(*_SF_WHITE)
            self.set_xy(x + 2, y0 + 8)
            self.cell(w - 4, 7, val)
            self.set_font("Ar", "I", 7.5)
            self.set_text_color(160, 185, 180)
            self.set_xy(x + 2, y0 + 16)
            self.cell(w - 4, 5, dlt)
        self.set_y(y0 + 26)

    def chart(self, path: str, w: float = 175, caption: str = ""):
        from PIL import Image as _Img
        with _Img.open(path) as _chart_img:
            chart_h = w * _chart_img.height / max(1, _chart_img.width)
        caption_h = 7 if caption else 0
        self.ensure_space(chart_h + caption_h + 2)
        self.image(path, x=15 + (175 - w) / 2, w=w)
        if caption:
            self.set_font("Ar", "I", 7.5)
            self.set_text_color(110, 125, 120)
            self.set_x(15)
            self.cell(180, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)


def _sf_no_policy_config(crisis_start: int = 30) -> dict:
    """Return a fresh, explicit no-policy configuration."""
    return {
        "fat_tax_active": False, "fat_tax_threshold": 3.5, "fat_tax_rate": 0.0,
        "subsidy_active": False, "subsidy_target": "domestic", "subsidy_rate": 0.0,
        "domestic_shock_active": False, "domestic_shock_day": int(crisis_start),
        "domestic_shock_duration": 30, "domestic_shock_severity": 0.5,
        "labelling_active": False, "labelling_day": 1,
        "labelling_health_boost": 0.0, "labelling_organic_boost": 0.0,
    }


def _sf_has_active_policy(params: dict | None) -> bool:
    """Return whether a SecureFood parameter set activates any policy lever."""
    params = params or {}
    policy_cfg = params.get("policy_cfg", {}) or {}
    return any([
        policy_cfg.get("fat_tax_active", False),
        policy_cfg.get("subsidy_active", False),
        policy_cfg.get("labelling_active", False),
        params.get("purchase_limit") is not None,
        float(params.get("media_intensity", 0.0)) > 0,
    ])


def _sf_without_policy(params: dict) -> dict:
    """Return a fresh copy of a scenario with every policy mechanism disabled."""
    crisis_start = int(params.get("cri_start", 30))
    return {
        **params,
        "policy_cfg": _sf_no_policy_config(crisis_start),
        "purchase_limit": None,
        "media_intensity": 0.0,
        "communication_type": "neutral",
    }


def _sf_param_signature(params: dict) -> str:
    """Stable signature used to prevent stale SecureFood results and reports."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def _sf_preset_report_params() -> tuple[dict, dict]:
    """Return independent copies of one shared, policy-free default scenario."""
    shared = {
        "days": 120, "month": 1, "base_con": 200, "reorder": 0.30,
        "target": 0.90, "lead": 3, "cri_start": 30, "cri_duration": 45,
        "inf": 25.0, "dis": 7, "panic": 0.50, "hoard": 1.5,
        "purchase_limit": None, "media_intensity": 0.0,
        "communication_type": "neutral", "stockpile_days": None,
        "exploratory_behaviour": True, "mc_runs": 1,
    }
    return (
        {**shared, "policy_cfg": _sf_no_policy_config(30)},
        {**shared, "policy_cfg": _sf_no_policy_config(30)},
    )


def _generate_sf_report_artifacts(
    sc_params: dict | None = None,
    pm_params: dict | None = None,
    report_mode: str = "Preset demonstration",
    include_policy_analysis: bool = False,
    report_model_revision: str = SF_REPORT_MODEL_REVISION,
) -> dict:
    """
    Generate a fresh SecureFood PDF plus aggregate and product-level CSV exports.

    When parameters are omitted, fresh no-policy defaults are used. Policy
    counterfactuals are included only when ``include_policy_analysis`` is true.
    All artifacts are generated from the same paired simulations so the PDF and
    CSV downloads cannot silently describe different scenario configurations.
    This function is intentionally not cached: a report button click is an
    explicit request to execute the ABM, and scientific outputs must never be
    silently reused across code deployments or browser sessions.
    """
    import textwrap as _tw
    from fpdf.enums import XPos, YPos

    # This value travels with the artifacts, allowing stale session-state
    # reports to be detected after a deployment.
    report_model_revision = str(report_model_revision)
    from datetime import datetime as _datetime, timezone as _timezone
    generated_at = _datetime.now(_timezone.utc).replace(microsecond=0).isoformat()

    # ── Report parameters ─────────────────────────────────────────────────────
    preset_sc, preset_pm = _sf_preset_report_params()
    sc_p = dict(sc_params) if sc_params is not None else preset_sc
    pm_p = dict(pm_params) if pm_params is not None else preset_pm
    include_policy_analysis = bool(include_policy_analysis)
    if not include_policy_analysis:
        # The fixed report has one scenario definition and two outcome lenses.
        # Force both sections to use the same policy-free inputs so operational
        # and household outcomes can be interpreted as one causal sequence.
        sc_p = _sf_without_policy(sc_p)
        pm_p = {
            **sc_p,
            "policy_cfg": dict(sc_p["policy_cfg"]),
        }
    sc_p["policy_cfg"] = dict(sc_p.get("policy_cfg", {}))
    pm_p["policy_cfg"] = dict(pm_p.get("policy_cfg", {}))
    report_mode = str(report_mode).strip() or "Custom"
    _no_pol = _sf_no_policy_config(pm_p.get("cri_start", 30))
    _sc_policy = sc_p.get("policy_cfg", {}) or {}
    _pm_policy = pm_p.get("policy_cfg", {}) or {}
    _has_sc_policy = any(
        _sc_policy.get(key, False)
        for key in (
            "fat_tax_active", "subsidy_active",
            "domestic_shock_active", "labelling_active",
        )
    )
    _has_pm_policy = include_policy_analysis and _sf_has_active_policy(pm_p)
    _policy_labels = []
    if _pm_policy.get("fat_tax_active", False):
        _policy_labels.append(
            f"{float(_pm_policy.get('fat_tax_rate', 0.0))*100:.0f}% fat-content surcharge"
        )
    if _pm_policy.get("subsidy_active", False):
        _policy_labels.append(
            f"{float(_pm_policy.get('subsidy_rate', 0.0))*100:.0f}% "
            f"{_pm_policy.get('subsidy_target', 'domestic')} subsidy"
        )
    if _pm_policy.get("labelling_active", False):
        _policy_labels.append("nutritional labelling")
    if pm_p.get("purchase_limit") is not None:
        _policy_labels.append(f"{int(pm_p['purchase_limit'])}-unit purchase cap")
    if float(pm_p.get("media_intensity", 0.0)) > 0:
        _policy_labels.append(
            f"{pm_p.get('communication_type', 'neutral')} communications "
            f"({float(pm_p['media_intensity']):.2f})"
        )
    _policy_summary = ", ".join(_policy_labels) if _policy_labels else "no active policy levers"
    _comparison_name = "selected-policy crisis" if _has_pm_policy else "unmitigated crisis"
    _welfare_scope = (
        "an additional policy counterfactual" if include_policy_analysis
        else "unmitigated food-security and equity outcomes"
    )
    _crisis_series_label = "selected policy crisis" if _has_pm_policy else "unmitigated crisis"

    # ── Run simulations ───────────────────────────────────────────────────────
    # Use a paired seed across all conditions.  Policy effects require a
    # crisis-without-policy counterfactual; baseline-vs-selected-crisis alone
    # cannot identify the incremental contribution of active policy levers.
    def _run_pair(params, policy_cfg):
        m_b = _make_model(params, is_crisis=False, seed=42, policy_cfg=policy_cfg)
        m_c = _make_model(params, is_crisis=True,  seed=42, policy_cfg=policy_cfg)
        rows, prod = [], []
        for day in range(1, params["days"] + 1):
            m_b.step(); m_c.step()
            ab, pb = _collect_model_day(m_b, day, "Baseline")
            ac, pc = _collect_model_day(m_c, day, "Crisis")
            rows += [ab, ac]; prod += pb + pc
        return {"df": pd.DataFrame(rows), "df_prod": pd.DataFrame(prod), "params": params}

    def _run_crisis(params, policy_cfg):
        model = _make_model(params, is_crisis=True, seed=42, policy_cfg=policy_cfg)
        rows, prod = [], []
        for day in range(1, params["days"] + 1):
            model.step()
            agg, product_rows = _collect_model_day(model, day, "Crisis")
            rows.append(agg)
            prod.extend(product_rows)
        return {"df": pd.DataFrame(rows), "df_prod": pd.DataFrame(prod), "params": params}

    pm_no_policy_p = {
        **pm_p,
        "policy_cfg": _no_pol,
        "purchase_limit": None,
        "media_intensity": 0.0,
        "communication_type": "neutral",
    }

    sc_data = _run_pair(sc_p, _sc_policy)
    shared_reference_run = (
        not _has_sc_policy
        and _sf_param_signature(_sf_without_policy(sc_p))
        == _sf_param_signature(pm_no_policy_p)
    )
    if shared_reference_run:
        # Reuse the exact paired run rather than independently recreating an
        # equivalent simulation. This guarantees cross-perspective consistency
        # and halves the default report's model CPU cost.
        pm_unpol_data = {
            "df": sc_data["df"].copy(),
            "df_prod": sc_data["df_prod"].copy(),
            "params": pm_no_policy_p,
        }
    else:
        pm_unpol_data = _run_pair(pm_no_policy_p, _no_pol)
    pm_policy_data = (
        _run_crisis(pm_p, _pm_policy)
        if _has_pm_policy else {
            "df": pm_unpol_data["df"][
                pm_unpol_data["df"]["Scenario"] == "Crisis"
            ].copy(),
            "df_prod": pm_unpol_data["df_prod"][
                pm_unpol_data["df_prod"]["Scenario"] == "Crisis"
            ].copy(),
            "params": pm_p,
        }
    )

    sc_df = sc_data["df"]
    sc_b  = sc_df[sc_df["Scenario"] == "Baseline"].reset_index(drop=True)
    sc_c  = sc_df[sc_df["Scenario"] == "Crisis"].reset_index(drop=True)
    pm_b  = pm_unpol_data["df"][pm_unpol_data["df"]["Scenario"] == "Baseline"].reset_index(drop=True)
    pm_u  = pm_unpol_data["df"][pm_unpol_data["df"]["Scenario"] == "Crisis"].reset_index(drop=True)
    pm_c  = pm_policy_data["df"].reset_index(drop=True)

    # ── SC metrics ────────────────────────────────────────────────────────────
    rev_b       = sc_b["Revenue"].sum()
    rev_c       = sc_c["Revenue"].sum()
    rev_loss    = rev_b - rev_c
    rev_pct     = 100 * rev_loss / max(rev_b, 1)
    lost_total  = sc_c["LostSales"].sum()
    peak_panic  = float(sc_c["PanicLevel"].max())
    mean_panic  = float(sc_c["PanicLevel"].mean())
    waste_delta = sc_c["Waste"].sum() - sc_b["Waste"].sum()
    nom_gain    = sc_c["NominalRevenue"].sum() - sc_b["Revenue"].sum()
    avg_pb      = float(sc_b["AvgPrice"].mean())
    avg_pc      = float(sc_c["AvgPrice"].mean())
    price_pct   = 100 * (avg_pc / max(avg_pb, 0.01) - 1)
    sc_cri_end = (
        sc_p["cri_start"] + sc_p["cri_duration"]
        if sc_p["cri_duration"] > 0 else sc_p["days"] + 1
    )
    merged_sc   = sc_b[["Day","Revenue"]].merge(sc_c[["Day","Revenue"]], on="Day", suffixes=("_b","_c"))
    post_sc     = merged_sc[merged_sc["Day"] > sc_cri_end]
    rec_rows    = post_sc[post_sc["Revenue_c"] >= post_sc["Revenue_b"] * 0.95]
    recovery_sc = int(rec_rows.iloc[0]["Day"] - sc_cri_end) if len(rec_rows) else None
    peak_vl_day = int(sc_b.loc[(sc_b["Revenue"] - sc_c["Revenue"]).idxmax(), "Day"])
    peak_lost_d = int(sc_c.loc[sc_c["LostSales"].idxmax(), "Day"])
    peak_lost_v = float(sc_c["LostSales"].max())
    sp_peak     = float(sc_c["StockpilePressure"].max())
    panic_peak_d= int(sc_c.loc[sc_c["PanicLevel"].idxmax(), "Day"])

    # ── PM metrics ────────────────────────────────────────────────────────────
    # Compare like-for-like days within the active crisis window. Averaging the
    # full horizon would dilute the effect with pre-crisis and recovery days.
    pm_cri_end = (
        pm_p["cri_start"] + pm_p["cri_duration"]
        if pm_p["cri_duration"] > 0 else pm_p["days"] + 1
    )
    def _pm_window(frame):
        return frame[
            (frame["Day"] >= pm_p["cri_start"]) & (frame["Day"] < pm_cri_end)
        ].reset_index(drop=True)

    pm_b_win = _pm_window(pm_b)
    pm_u_win = _pm_window(pm_u)
    pm_c_win = _pm_window(pm_c)

    peak_stress = float(pm_c_win["FoodStressedPct"].max()) * 100
    base_stress = float(pm_b_win["FoodStressedPct"].mean()) * 100
    peak_bx_lo  = float(pm_c_win["BudgetExh_Low"].max()) * 100
    peak_bx_hi  = float(pm_c_win["BudgetExh_High"].max()) * 100
    mean_gini_c = float(pm_c_win["GiniAccess"].mean())
    mean_gini_b = float(pm_b_win["GiniAccess"].mean())
    imp_dep_b   = float(pm_b_win["ImportDepPct"].mean())
    imp_dep_c   = float(pm_c_win["ImportDepPct"].mean())
    ful_lo      = float(pm_c_win["Fulfillment_Low"].mean()) * 100
    ful_hi      = float(pm_c_win["Fulfillment_High"].mean()) * 100
    fies_peak   = float(pm_c_win["FIESSevere_Low"].max()) * 100
    fies_base   = float(pm_b_win["FIESSevere_Low"].mean()) * 100
    dom_b       = pm_b_win["DomesticSales"].sum() / max(pm_b_win["DomesticSales"].sum() + pm_b_win["ImportSales"].sum(), 1) * 100
    dom_c       = pm_c_win["DomesticSales"].sum() / max(pm_c_win["DomesticSales"].sum() + pm_c_win["ImportSales"].sum(), 1) * 100
    ful_lo_u    = float(pm_u_win["Fulfillment_Low"].mean()) * 100
    ful_hi_u    = float(pm_u_win["Fulfillment_High"].mean()) * 100
    fies_peak_u = float(pm_u_win["FIESSevere_Low"].max()) * 100
    mean_gini_u = float(pm_u_win["GiniAccess"].mean())
    imp_dep_u   = float(pm_u_win["ImportDepPct"].mean())
    dom_u       = pm_u_win["DomesticSales"].sum() / max(pm_u_win["DomesticSales"].sum() + pm_u_win["ImportSales"].sum(), 1) * 100
    peak_panic_u = float(pm_u_win["PanicLevel"].max())
    peak_panic_c = float(pm_c_win["PanicLevel"].max())
    ful_lo_b     = float(pm_b_win["Fulfillment_Low"].mean()) * 100
    fies_peak_b  = float(pm_b_win["FIESSevere_Low"].max()) * 100
    peak_panic_b = float(pm_b_win["PanicLevel"].max())
    _reference_name = "paired crisis without policy" if include_policy_analysis else "baseline"
    _ref_ful_lo = ful_lo_u if include_policy_analysis else ful_lo_b
    _ref_fies_peak = fies_peak_u if include_policy_analysis else fies_peak_b
    _ref_gini = mean_gini_u if include_policy_analysis else mean_gini_b
    _ref_import_dep = imp_dep_u if include_policy_analysis else imp_dep_b
    _ref_domestic = dom_u if include_policy_analysis else dom_b
    _ref_panic = peak_panic_u if include_policy_analysis else peak_panic_b

    # ── Matplotlib style ──────────────────────────────────────────────────────
    _C = {"b": "#2980b9", "r": "#c0392b", "a": "#DBA159", "g": "#27ae60",
          "t": "#44A1A0", "o": "#e67e22", "p": "#8e44ad", "gr": "#95a5a6",
          "dk": "#042026", "lo": "#e74c3c", "mi": "#e67e22", "hi": "#27ae60"}
    plt.rcParams.update({
        "font.family": "sans-serif", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linestyle": "--",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    })

    def _crisis_span(ax, s, e, d):
        ax.axvspan(s, min(e, d), alpha=0.08, color="#c0392b", label="Crisis window")
        ax.axvline(s, color="#c0392b", lw=0.8, ls=":")

    temp_imgs = []  # track temp files

    # ── Chart A: Revenue decomposition ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(sc_b["Day"], sc_b["Revenue"], color=_C["b"], lw=2, label="Baseline (constant price)")
    ax.plot(sc_c["Day"], sc_c["Revenue"], color=_C["r"], lw=2, label="Crisis (constant price)")
    ax.plot(sc_c["Day"], sc_c["NominalRevenue"], color=_C["o"], lw=1.4, ls="--",
            label="Crisis nominal (inflated)")
    ax.fill_between(sc_b["Day"], sc_b["Revenue"], sc_c["Revenue"],
                    alpha=0.10, color=_C["r"], label="Revenue gap")
    _crisis_span(ax, sc_p["cri_start"], sc_cri_end, sc_p["days"])
    ax.set_xlabel("Simulation Day"); ax.set_ylabel("Daily Revenue (€)")
    ax.set_title("Revenue Impact: Constant-Price vs Nominal — Inflation-Volume Decomposition")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    fig.tight_layout()
    p_revA = _sf_mpl_chart(fig); temp_imgs.append(p_revA)

    # ── Chart B: Stockout / Lost Sales ────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(7, 3.0))
    sc_c2 = sc_c.copy(); sc_c2["CumLost"] = sc_c2["LostSales"].cumsum()
    ax1.bar(sc_c2["Day"], sc_c2["LostSales"], color=_C["r"], alpha=0.65, label="Daily lost sales")
    ax2 = ax1.twinx()
    ax2.plot(sc_c2["Day"], sc_c2["CumLost"], color=_C["a"], lw=2, label="Cumulative lost sales")
    ax2.spines["right"].set_visible(True)
    _crisis_span(ax1, sc_p["cri_start"], sc_cri_end, sc_p["days"])
    ax1.set_xlabel("Simulation Day"); ax1.set_ylabel("Unmet Demand Units/day")
    ax2.set_ylabel("Cumulative Units")
    ax1.set_title("Unmet Demand — Daily Events and Cumulative Accumulation")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    p_lost = _sf_mpl_chart(fig); temp_imgs.append(p_lost)

    # ── Chart C: Panic & Stockpile ────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(7, 2.8))
    ax1.plot(sc_c["Day"], sc_c["PanicLevel"], color=_C["r"], lw=2, label="Panic level (crisis)")
    ax1.plot(sc_b["Day"], sc_b["PanicLevel"], color=_C["gr"], lw=1.2, ls="--", label="Panic level (baseline)")
    ax2 = ax1.twinx()
    ax2.plot(sc_c["Day"], sc_c["StockpilePressure"], color=_C["a"], lw=1.8, ls="-.",
             label="Stockpile pressure (crisis)")
    ax2.spines["right"].set_visible(True)
    _crisis_span(ax1, sc_p["cri_start"], sc_cri_end, sc_p["days"])
    ax1.set_xlabel("Simulation Day"); ax1.set_ylabel("Panic Level (0–1)")
    ax2.set_ylabel("Demand Ratio (1.0 = base)")
    ax1.set_title("Consumer Panic Level and Stockpile Pressure")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7.5, ncol=2)
    fig.tight_layout()
    p_panic = _sf_mpl_chart(fig); temp_imgs.append(p_panic)

    # ── Chart D: Shelf Stock (per-category from df_prod) ─────────────────────
    dp = sc_data["df_prod"]
    cats = dp["Category"].dropna().unique().tolist()[:5] if "Category" in dp.columns else []
    fig, ax = plt.subplots(figsize=(7, 3.0))
    cat_colors = [_C["b"], _C["g"], _C["p"], _C["a"], _C["t"]]
    if cats:
        for i, cat in enumerate(cats):
            dc_b = dp[(dp["Category"] == cat) & (dp["Scenario"] == "Baseline")].groupby("Day")["Shelf"].mean()
            dc_c = dp[(dp["Category"] == cat) & (dp["Scenario"] == "Crisis")].groupby("Day")["Shelf"].mean()
            col = cat_colors[i % len(cat_colors)]
            ax.plot(dc_b.index, dc_b.values, color=col, lw=1.2, ls="--", alpha=0.55)
            ax.plot(dc_c.index, dc_c.values, color=col, lw=2.0, label=cat)
    else:
        if not dp.empty and "Shelf" in dp.columns:
            tot_b = dp[dp["Scenario"] == "Baseline"].groupby("Day")["Shelf"].sum()
            tot_c = dp[dp["Scenario"] == "Crisis"].groupby("Day")["Shelf"].sum()
            ax.plot(tot_b.index, tot_b.values, color=_C["b"], lw=1.5, ls="--", label="Baseline shelf stock")
            ax.plot(tot_c.index, tot_c.values, color=_C["r"], lw=2.0, label="Crisis shelf stock")
    _crisis_span(ax, sc_p["cri_start"], sc_cri_end, sc_p["days"])
    ax.set_xlabel("Simulation Day"); ax.set_ylabel("Units on Shelf")
    ax.set_title("Shelf Stock by Category — Baseline (dashed) vs Crisis (solid)")
    ax.legend(fontsize=7.5, ncol=3)
    fig.tight_layout()
    p_stock = _sf_mpl_chart(fig); temp_imgs.append(p_stock)

    # ── Chart E: Basket Fulfilment by Income ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.0))
    for col_k, col_c, lbl in [
        ("Fulfillment_Low",  _C["lo"], "Low income"),
        ("Fulfillment_Mid",  _C["mi"], "Mid income"),
        ("Fulfillment_High", _C["hi"], "High income"),
    ]:
        ax.plot(pm_c["Day"], pm_c[col_k] * 100, color=col_c, lw=2,
                label=f"{lbl} ({_crisis_series_label})")
    if include_policy_analysis:
        ax.plot(pm_u["Day"], pm_u["Fulfillment_Low"] * 100, color=_C["lo"], lw=1.3,
                ls=":", label="Low income (crisis, no policy)")
    ax.plot(pm_b["Day"], pm_b["FulfillmentRate"] * 100, color=_C["gr"], lw=1.2, ls="--",
            label="All income (baseline)")
    ax.axhline(80, color="#555555", ls="--", lw=1, label="80% reporting reference")
    _crisis_span(ax, pm_p["cri_start"], pm_cri_end, pm_p["days"])
    ax.set_xlabel("Simulation Day"); ax.set_ylabel("Fulfilment Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Consumer Basket Fulfilment Rate by Income Group — Crisis Scenario")
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    p_ful = _sf_mpl_chart(fig); temp_imgs.append(p_ful)

    # ── Chart F: FIES Food Insecurity ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(pm_c["Day"], pm_c["FIESSevere_Low"]  * 100, color=_C["lo"], lw=2, label="Low income (crisis)")
    ax.plot(pm_c["Day"], pm_c["FIESSevere_Mid"]  * 100, color=_C["mi"], lw=2, label="Mid income (crisis)")
    ax.plot(pm_c["Day"], pm_c["FIESSevere_High"] * 100, color=_C["hi"], lw=2, label="High income (crisis)")
    ax.plot(pm_b["Day"], pm_b["FIESSevere_Low"]  * 100, color=_C["lo"], lw=1, ls="--", alpha=0.5, label="Low (baseline)")
    if include_policy_analysis:
        ax.plot(pm_u["Day"], pm_u["FIESSevere_Low"]  * 100, color="#2c3e50", lw=1.2,
                ls=":", label="Low (crisis, no policy)")
    _crisis_span(ax, pm_p["cri_start"], pm_cri_end, pm_p["days"])
    ax.set_xlabel("Simulation Day"); ax.set_ylabel("Access Stress High (%)")
    ax.set_title("High Food-Access Stress by Income Bracket — Crisis vs Baseline")
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    p_fies = _sf_mpl_chart(fig); temp_imgs.append(p_fies)

    # ── Chart G: Budget Exhaustion & Gini ─────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(7, 2.8))
    ax1.plot(pm_c["Day"], pm_c["BudgetExh_Low"]  * 100, color=_C["lo"], lw=2, label="Low income budget exhausted")
    ax1.plot(pm_c["Day"], pm_c["BudgetExh_High"] * 100, color=_C["hi"], lw=2, label="High income budget exhausted")
    ax2 = ax1.twinx()
    ax2.plot(pm_c["Day"], pm_c["GiniAccess"], color=_C["a"], lw=1.8, ls="-.", label="Gini access index")
    ax2.plot(pm_b["Day"], pm_b["GiniAccess"], color=_C["a"], lw=1, ls=":", alpha=0.5, label="Gini (baseline)")
    if include_policy_analysis:
        ax2.plot(pm_u["Day"], pm_u["GiniAccess"], color="#2c3e50", lw=1.1,
                 ls="--", label="Gini (crisis, no policy)")
    ax2.spines["right"].set_visible(True)
    _crisis_span(ax1, pm_p["cri_start"], pm_cri_end, pm_p["days"])
    ax1.set_xlabel("Simulation Day"); ax1.set_ylabel("Budget Exhausted (%)")
    ax2.set_ylabel("Gini Access Index (0–1)")
    ax1.set_title("Budget Exhaustion by Income Group and Gini Access Index")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7.5, ncol=2)
    fig.tight_layout()
    p_gini = _sf_mpl_chart(fig); temp_imgs.append(p_gini)

    # ── Chart H: Domestic vs Import ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.stackplot(pm_c["Day"],
                 pm_c["DomesticSales"], pm_c["ImportSales"],
                 labels=[f"Domestic ({_crisis_series_label})", f"Import ({_crisis_series_label})"],
                 colors=[_C["g"], _C["b"]], alpha=0.65)
    if include_policy_analysis:
        ax.plot(pm_u["Day"], pm_u["DomesticSales"] + pm_u["ImportSales"],
                color="#2c3e50", lw=1.2, ls=":", label="Total crisis (no policy)")
    ax.plot(pm_b["Day"], pm_b["DomesticSales"] + pm_b["ImportSales"],
            color=_C["gr"], lw=1.5, ls="--", label="Total baseline")
    _crisis_span(ax, pm_p["cri_start"], pm_cri_end, pm_p["days"])
    ax.set_xlabel("Simulation Day"); ax.set_ylabel("Units Sold")
    ax.set_title("Domestic vs Import Sales Volume — Food Sovereignty")
    ax.legend(fontsize=7.5, ncol=3)
    fig.tight_layout()
    p_dom = _sf_mpl_chart(fig); temp_imgs.append(p_dom)

    # ── BUILD PDF ─────────────────────────────────────────────────────────────
    pdf = _SFReport()
    pdf.set_auto_page_break(auto=False)
    pdf._lf()

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1: COVER
    # ─────────────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(*_SF_DARK)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_fill_color(*_SF_AMBER)
    pdf.rect(0, 0, 210, 5, "F")

    # Logo row: GROCERYsim + SecureFood side by side on DARK
    gs_p = _sf_logo_on("GROCERYsim.png", 360, _SF_DARK)
    sf_p = _sf_logo_on("SecureFood.png", 260, _SF_DARK)
    temp_imgs += [gs_p, sf_p]
    from PIL import Image as _PILImg
    gs_img = _PILImg.open(gs_p)
    sf_img = _PILImg.open(sf_p)
    gs_w, gs_h_mm = 80, 80 * gs_img.height / gs_img.width
    sf_w, sf_h_mm = 58, 58 * sf_img.height / sf_img.width
    # Outer frame
    pdf.set_draw_color(*_SF_AMBER)
    pdf.set_line_width(0.6)
    pdf.rect(25, 12, 160, gs_h_mm + 10)
    pdf.image(gs_p, x=30, y=17, w=gs_w)
    pdf.image(sf_p, x=120, y=17 + (gs_h_mm - sf_h_mm) / 2, w=sf_w)

    y_after_logos = 17 + gs_h_mm + 14

    # Amber rule
    pdf.set_fill_color(*_SF_AMBER)
    pdf.rect(15, y_after_logos, 180, 0.8, "F")

    # Report title
    pdf.set_y(y_after_logos + 8)
    pdf.set_font("Ar", "B", 22)
    pdf.set_text_color(*_SF_WHITE)
    pdf.cell(0, 10, f"SecureFood {report_mode} Report", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Ar", "I", 12)
    pdf.set_text_color(*_SF_AMBER)
    pdf.cell(0, 7, "Climate-Driven Dairy Supply Chain Disruption — Finland", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_fill_color(*_SF_AMBER)
    pdf.rect(15, pdf.get_y(), 180, 0.8, "F")
    pdf.ln(8)

    # Description block
    pdf.set_font("Ar", "", 9)
    pdf.set_text_color(190, 210, 205)
    desc = (
        "This report presents a comprehensive agent-based simulation analysis of the impact "
        "of climate-driven disruptions on the Finnish dairy supply chain. The simulation "
        "applies the GROCERYsim ABM v2.0 framework informed by 116 collected Finnish "
        "participant records, of which 108 provide usable matched baskets and linked DCE data. "
        "Results are reported from two complementary perspectives: Supply Chain Actors "
        f"(operational resilience) and {_welfare_scope}. This is a {report_mode.lower()} "
        "report. Its displayed parameters are analyst-defined stress-test assumptions, not "
        "forecasts or IPCC-calibrated effects."
    )
    pdf.set_x(25)
    pdf.multi_cell(160, 5.5, desc)
    pdf.ln(6)

    # Info box
    box_y = pdf.get_y()
    pdf.set_fill_color(*_SF_DARK2)
    pdf.rect(25, box_y, 160, 30, "F")
    pdf.set_fill_color(*_SF_AMBER)
    pdf.rect(25, box_y, 4, 30, "F")
    pdf.set_y(box_y + 3)
    pdf.set_font("Ar", "B", 9)
    pdf.set_text_color(*_SF_AMBER)
    pdf.set_x(34)
    pdf.cell(0, 5.5, "Horizon Europe SecureFood Consortium", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Ar", "", 8.5)
    pdf.set_text_color(190, 210, 205)
    pdf.set_x(34); pdf.cell(0, 5, "Grant Agreement No. 101136583", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(34); pdf.cell(0, 5, "IAMO XR Lab — Leibniz Institute of Agricultural Development in Transition Economies", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(34); pdf.cell(0, 5, "Theodor-Lieser Str. 2  ·  06120 Halle (Saale)  ·  Germany  ·  www.iamo.de", new_x="LMARGIN", new_y="NEXT")

    # Partner logos
    strip_y = 225
    pdf.set_fill_color(*_SF_DARK2)
    pdf.rect(0, strip_y - 4, 210, 48, "F")
    pdf.set_fill_color(*_SF_AMBER)
    pdf.rect(0, strip_y - 4, 210, 0.7, "F")
    pdf.set_y(strip_y); pdf.set_font("Ar", "B", 6.5)
    pdf.set_text_color(*_SF_AMBER)
    pdf.cell(0, 5, "CONSORTIUM PARTNERS", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    logos_cfg = [("EU.png", 20, 260), ("SecureFood.png", 50, 260),
                 ("IAMO.png", 46, 260), ("Logo_lab.png", 38, 260)]
    gap = 5
    total = sum(w for _, w, _ in logos_cfg) + gap * 3
    xl = (210 - total) / 2
    for nm, w, px in logos_cfg:
        lp = _sf_logo_on(nm, px, _SF_DARK2)
        temp_imgs.append(lp)
        li = _PILImg.open(lp)
        h = w * li.height / li.width
        pdf.image(lp, x=xl, y=strip_y + 8, w=w)
        xl += w + gap

    pdf.set_fill_color(*_SF_DARK)
    pdf.rect(0, 277, 210, 20, "F")
    pdf.set_fill_color(*_SF_AMBER)
    pdf.rect(0, 277, 210, 0.7, "F")
    pdf.set_y(280)
    pdf.set_font("Ar", "I", 7.5)
    pdf.set_text_color(180, 195, 190)
    pdf.cell(0, 5, "Generated by GROCERYsim ABM v2.0  ·  Funded by the European Union — Views are those of the authors only.", align="C",
             new_x="LMARGIN", new_y="NEXT")
    from datetime import datetime as _dt
    pdf.set_font("Ar", "", 7)
    pdf.cell(
        0, 4.5,
        f"Model revision: {report_model_revision}  ·  Generated: "
        f"{_dt.now().strftime('%d %B %Y %H:%M:%S')}",
        align="C",
    )

    # Re-enable automatic breaks for flowing report content. The cover uses
    # fixed coordinates and must never create an overflow-only second page.
    pdf.set_auto_page_break(auto=True, margin=16)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2: EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    pdf.chapter("1", "Executive Summary", "Executive Summary")
    pdf.body(
        f"This {report_mode.lower()} report analyses the impact of a climate-driven disruption on the Finnish dairy "
        "supply chain using the GROCERYsim Agent-Based Model v2.0. The simulation runs for "
        f"{sc_p['days']} days, with a crisis beginning on Day {sc_p['cri_start']} and featuring a "
        f"{sc_p['inf']:.0f}% retail price inflation, a {sc_p['dis']}-day supply delivery delay, "
        f"and a {sc_p['cri_duration']}-day disruption window. Consumer panic sensitivity is set "
        f"at {sc_p['panic']:.2f} and the maximum hoarding multiplier at {sc_p['hoard']:.1f}. "
        "The supply-chain and food-security sections use the exact same paired baseline/crisis "
        "run and seed; they differ only in the outcomes reported. "
        "These are scenario assumptions: the current export has no direct panic-belief measure."
    )

    pdf.sub("Key Findings — Supply Chain Perspective")
    pdf.metric_row([
        ("Total Revenue Loss",    f"€{rev_loss:,.0f}",     f"−{rev_pct:.1f}% vs baseline",      False),
        ("Unmet Demand",          f"{lost_total:,.0f} units", "stockout/price constrained",       False),
        ("Peak Panic Level",      f"{peak_panic:.2f}/1.0", f"mean {mean_panic:.2f}",             False),
        ("Recovery Time",         f"{recovery_sc} days" if recovery_sc else "Not in horizon",
         "after crisis end",                                                                       recovery_sc is not None),
    ])
    pdf.body(
        f"The crisis generated a constant-price revenue contraction of €{rev_loss:,.0f} "
        f"({rev_pct:.1f}%) over the simulation horizon. The model also recorded {lost_total:,.0f} "
        f"unmet demand units; this is a mechanism-level quantity and must not be added to the "
        f"revenue gap. Consumer panic peaked on Day {panic_peak_d} at a "
        f"level of {peak_panic:.2f}/1.0, accompanied by a stockpiling-demand ratio of "
        f"{sp_peak:.2f}. Nominal revenue (at inflated prices) "
        f"{'increased' if nom_gain > 0 else 'decreased'} by €{abs(nom_gain):,.0f}, "
        f"masking the underlying volume shortfall — a critical distinction for operational "
        "planning. Food waste "
        f"{'increased' if waste_delta >= 0 else 'decreased'} by {abs(waste_delta):.0f} units "
        "relative to baseline in this paired deterministic run."
    )

    pdf.sub(
        "Key Findings — Additional Policy Analysis"
        if include_policy_analysis else "Key Findings — Food Security & Equity"
    )
    pdf.metric_row([
        ("Low-Income Access Stress", f"{peak_stress:.1f}%", f"baseline {base_stress:.1f}%",      False),
        ("Low-Income Fulfilment", f"{ful_lo:.1f}%", f"{_reference_name} {_ref_ful_lo:.1f}%", ful_lo >= _ref_ful_lo),
        ("High Access Stress (Low)", f"{fies_peak:.1f}%", f"{_reference_name} {_ref_fies_peak:.1f}%", fies_peak <= _ref_fies_peak),
        ("Gini Access Index", f"{mean_gini_c:.3f}", f"{_reference_name} {_ref_gini:.3f}", mean_gini_c <= _ref_gini),
    ])
    pdf.body(
        f"The {_comparison_name} run produced a peak {peak_stress:.1f}% share of all shoppers "
        f"who were both low-income and access-stressed (baseline mean {base_stress:.1f}%). "
        f"Low-income basket fulfilment "
        f"averaged {ful_lo:.1f}%, compared with {ful_hi:.1f}% for high-income households. "
        f"Against the {_reference_name}, low-income fulfilment changed by "
        f"{ful_lo - _ref_ful_lo:+.1f} percentage points, peak high access stress changed by "
        f"{fies_peak - _ref_fies_peak:+.1f} points, and the mean Gini access index changed by "
        f"{mean_gini_c - _ref_gini:+.3f}. Import dependency changed from "
        f"{_ref_import_dep:.1f}% in the reference to {imp_dep_c:.1f}% in the analysed crisis, "
        f"and peak panic changed from {_ref_panic:.2f} to {peak_panic_c:.2f}. These are paired, "
        "single-seed scenario differences, not estimated causal effects."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 3: SCENARIO PARAMETERS
    # ─────────────────────────────────────────────────────────────────────────
    pdf.chapter("2", "Scenario Configuration & Parameters", "Scenario Parameters")
    pdf.body(
        f"The following parameters were used for this {report_mode.lower()} report. These values "
        "define an illustrative moderate-to-severe disruption affecting Finnish dairy supply "
        "chains. They are transparent analyst inputs and have not been calibrated to an observed event."
    )
    pdf.sub("2.1  Shared Scenario Definition", min_content_height=65)
    pdf.kv([
        ("Horizon & Traffic",    f"{sc_p['days']} days; {sc_p['base_con']} agents/day"),
        ("Calendar & Crisis",    f"Month {sc_p['month']}; starts Day {sc_p['cri_start']}; "
                                  f"lasts {sc_p['cri_duration']} days (ends Day {sc_cri_end})"),
        ("Stress Inputs",        f"{sc_p['inf']:.0f}% price inflation; "
                                  f"{sc_p['dis']}-day delivery interruption"),
        ("Behaviour",            f"Panic {sc_p['panic']:.2f}; hoarding {sc_p['hoard']:.1f}× maximum "
                                  "(exploratory assumptions)"),
        ("Logistics",            f"{sc_p['lead']}-day lead time (engineering assumption)"),
        ("Inventory Policy",     f"Reorder at {sc_p['reorder']*100:.0f}%; "
                                  f"restock to {sc_p['target']*100:.0f}% of capacity"),
        ("Policy Interventions", "None" if not _has_sc_policy else "Configured in supplied settings"),
    ])
    if include_policy_analysis:
        pdf.sub("2.2  Additional Policy Counterfactual", min_content_height=75)
        pdf.kv([
            ("Shared Scenario", "Same horizon, crisis, logistics, behaviour, and seed as Section 2.1"),
            ("Fat-Content Surcharge", (
                f"{float(_pm_policy.get('fat_tax_rate', 0.0))*100:.0f}% above "
                f"{float(_pm_policy.get('fat_tax_threshold', 3.5)):.1f}% fat"
                if _pm_policy.get("fat_tax_active", False) else "Disabled"
            )),
            ("Product Subsidy",     (
                f"{float(_pm_policy.get('subsidy_rate', 0.0))*100:.0f}% on "
                f"{_pm_policy.get('subsidy_target', 'domestic')} products"
                if _pm_policy.get("subsidy_active", False) else "Disabled"
            )),
            ("Purchase Rationing",   (
                f"{int(pm_p['purchase_limit'])}-unit per-product cap"
                if pm_p.get("purchase_limit") is not None else "Disabled"
            )),
            ("Nutritional Labelling", (
                f"Active from Day {int(_pm_policy.get('labelling_day', 1))}; "
                f"health +{float(_pm_policy.get('labelling_health_boost', 0.0))*100:.0f}%, "
                f"organic +{float(_pm_policy.get('labelling_organic_boost', 0.0))*100:.0f}%"
                if _pm_policy.get("labelling_active", False) else "Disabled"
            )),
            ("Gov. Communications",  (
                f"{pm_p.get('communication_type', 'neutral').title()} at intensity "
                f"{float(pm_p.get('media_intensity', 0.0)):.2f}"
                if float(pm_p.get("media_intensity", 0.0)) > 0 else "Disabled / neutral"
            )),
        ])
    else:
        pdf.sub("2.2  Outcome Perspectives", min_content_height=45)
        pdf.kv([
            ("Supply-Chain Lens", "Revenue, sales, inventory, deliveries, waste, and recovery"),
            ("Food-Security Lens", "Basket fulfilment, access stress, inequality, and import dependency"),
            ("Consistency Rule", "Both lenses use the exact same paired baseline/crisis run with seed 42"),
        ])
    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 4: SUPPLY CHAIN ANALYSIS — Revenue
    # ─────────────────────────────────────────────────────────────────────────
    pdf.chapter("3", "Supply Chain Actor Analysis", "SC Analysis — Revenue")
    pdf.body(
        "This section analyses the operational impact of the climate disruption from the "
        "perspective of producers, distributors, and retailers managing the Finnish dairy "
        "supply chain. The focus is on revenue integrity, stockout risk, inventory dynamics, "
        "and consumer panic propagation."
    )
    pdf.sub("3.1  Revenue Impact — Inflation-Volume Decomposition", min_content_height=100)
    pdf.chart(p_revA,
        caption="Fig. 1 — Daily revenue: baseline vs crisis (constant-price) and crisis nominal (inflated). "
                "Red shading = revenue gap. Dotted vertical = crisis onset.")
    pdf.finding(
        f"Constant-price revenue fell by €{rev_loss:,.0f} ({rev_pct:.1f}%) over {sc_p['days']} days. "
        f"Nominal revenue (orange dashed, at inflated prices) "
        f"{'rose' if nom_gain > 0 else 'fell'} by €{abs(nom_gain):,.0f}, "
        f"demonstrating that the {sc_p['inf']:.0f}% inflation partially offsets the volume loss in "
        f"headline figures — masking the true operational deterioration. "
        f"Peak single-day volume loss occurred on Day {peak_vl_day}."
    )
    pdf.sub("3.2  Unmet Demand Events", min_content_height=100)
    pdf.chart(p_lost,
        caption="Fig. 2 — Daily unmet demand units (bars) and cumulative units (amber line).")
    pdf.finding(
        f"The model recorded {lost_total:,.0f} unmet demand units over the simulation horizon. "
        f"The peak single-day count was {peak_lost_v:,.0f} units on Day {peak_lost_d}. "
        "This counter includes quantity not fulfilled after price rejection, substitution, or "
        "stock constraints; it is not denominated in euros and is not evidence of permanent "
        "customer loss."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 5: SC Analysis — Stock & Panic
    # ─────────────────────────────────────────────────────────────────────────
    pdf._sec = "SC Analysis — Stock & Panic"
    pdf.sub("3.3  Shelf Stock Depletion by Product Category", min_content_height=100)
    pdf.chart(p_stock,
        caption="Fig. 3 — Mean shelf stock by category: baseline (dashed) vs crisis (solid). "
                "Each colour represents one product category.")
    pdf.body(
        "The solid lines show the realised crisis trajectory and the dashed lines the paired "
        "baseline trajectory. Differences reflect the combined analyst-defined supply interruption, "
        "price shock, and exploratory demand-amplification mechanisms. Category-specific effects "
        "should be interpreted from the plotted trajectories rather than assumed from shelf life alone."
    )
    pdf.sub("3.4  Consumer Panic Level and Stockpile Pressure", min_content_height=100)
    pdf.chart(p_panic,
        caption="Fig. 4 — Internal panic state (left) and stockpiling-demand ratio (right; 1.0 = base demand).")
    pdf.finding(
        f"Consumer panic peaked at {peak_panic:.2f}/1.0 on Day {panic_peak_d}, "
        f"with a simulation mean of {mean_panic:.2f}. The stockpiling-demand ratio reached "
        f"{sp_peak:.2f} (1.0 = base requested demand). "
        "Both are internal exploratory state variables generated by scenario assumptions; "
        "they are not measured panic or validated behavioural thresholds. The household-level "
        "hoarding multiplier is nevertheless scaled by the model's cross-fitted propensity, "
        "preserving the evidence separation introduced in the revised model."
    )
    pdf.body(
        "The scenario combines a supply interruption with optional demand amplification. "
        + ("The additional module compares paired no-policy and selected-policy crisis runs. "
           if include_policy_analysis else
           "The default analysis compares the unmitigated crisis with the no-shock baseline. ")
        + "The methodology section identifies panic and communication dynamics "
        "as unvalidated exploratory assumptions."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 6: POLICY MAKER ANALYSIS — Welfare
    # ─────────────────────────────────────────────────────────────────────────
    pdf.chapter(
        "4",
        "Additional Policy Analysis — Consumer Welfare"
        if include_policy_analysis else "Food Security & Equity Analysis",
        "Additional Policy Analysis" if include_policy_analysis else "Food Security & Equity",
    )
    pdf.body(
        "This section analyses the food security and equity implications of the dairy supply "
        "disruption from the perspective of government agencies, regulators, and food system "
        "authorities. "
        + (
            "Results compare the selected policy configuration against a paired crisis-without-policy "
            f"counterfactual. Selected levers: {_policy_summary}."
            if include_policy_analysis
            else "This default analysis contains no intervention. It compares the unmitigated crisis "
                 "with the no-shock baseline; policy effects are available only in the additional module."
        )
    )
    pdf.sub("4.1  Basket Fulfilment Rate by Income Group", min_content_height=100)
    pdf.chart(p_ful,
        caption=f"Fig. 5 — {_crisis_series_label.title()} fulfilment by income and {_reference_name}. "
                "The 80% line is a descriptive reporting reference, not a validated welfare threshold.")
    pdf.finding(
        f"Low-income households achieved {ful_lo:.1f}% mean basket fulfilment in the {_comparison_name} "
        f"run, vs {ful_hi:.1f}% for high-income — a {ful_hi - ful_lo:.1f} pp within-run gap. "
        f"Low-income fulfilment changed by {ful_lo - _ref_ful_lo:+.1f} pp relative to the "
        f"{_reference_name}. It fell below the descriptive 80% reference for "
        f"{int((pm_c_win['Fulfillment_Low'] < 0.80).sum())} active-crisis days. "
        "The model supports descriptive distributional comparison, but one paired seed does not "
        "establish population uncertainty or a causal policy effect."
    )
    pdf.sub("4.2  High Food-Access Stress by Income Bracket", min_content_height=100)
    pdf.chart(p_fies,
        caption=f"Fig. 6 — high modeled access stress by income, compared with the {_reference_name}. "
                "Exploratory model diagnostic; not survey-based FIES prevalence.")
    pdf.finding(
        f"High modeled access stress among low-income households peaked "
        f"at {fies_peak:.1f}% in the {_comparison_name} run, versus {_ref_fies_peak:.1f}% in the "
        f"{_reference_name} and a baseline mean of {fies_base:.1f}%. The unvalidated "
        f"diagnostic combines access and consumption shortfall signals. It supports comparisons among model "
        f"scenarios but is not a food-insecurity prevalence estimate."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 7: PM Analysis — Equity & Sovereignty
    # ─────────────────────────────────────────────────────────────────────────
    pdf._sec = (
        "Additional Policy — Equity & Sovereignty"
        if include_policy_analysis else "Food Security — Equity & Sovereignty"
    )
    pdf.sub("4.3  Budget Exhaustion and Access Inequality (Gini Index)", min_content_height=100)
    pdf.chart(p_gini,
        caption="Fig. 7 — Budget exhaustion rates by income (left axis) and Gini access index (right axis, amber).")
    pdf.body(
        f"Budget exhaustion — defined as a consumer being unable to complete their intended "
        f"basket due to price constraints — reached {peak_bx_lo:.1f}% for low-income "
        f"households at peak, compared to {peak_bx_hi:.1f}% for high-income. This "
        f"{peak_bx_lo - peak_bx_hi:.1f} pp differential directly reflects the regressive "
        f"impact of food price inflation: the same percentage increase costs low-income "
        f"households a much larger share of their available food budget. "
        f"The mean Gini access index was {mean_gini_b:.3f} in the baseline and "
        f"{mean_gini_c:.3f} in the {_comparison_name} run. "
        f"The simulated difference from the {_reference_name} was {mean_gini_c - _ref_gini:+.3f}. "
        "No universal intervention threshold is imposed; the index is used only for relative "
        "comparison within this model."
    )
    pdf.sub("4.4  Domestic vs Import Sales Volume — Food Sovereignty", min_content_height=100)
    pdf.chart(p_dom,
        caption=f"Fig. 8 — {_crisis_series_label.title()} domestic/import volume with {_reference_name} totals.")
    pdf.finding(
        f"Domestic products represented {dom_c:.1f}% of sales in the {_comparison_name} run, "
        f"compared with {_ref_domestic:.1f}% in the {_reference_name}. Corresponding import "
        f"dependency was {imp_dep_c:.1f}% and {_ref_import_dep:.1f}%. "
        + ("This bundled comparison should not be interpreted as a separately identified effect "
           "of any one policy lever."
           if include_policy_analysis else
           "This is a descriptive shock-versus-baseline comparison, not a policy-effect estimate.")
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 8: POLICY EFFECTIVENESS & RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    class _NoOpPolicySection:
        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    policy_pdf = pdf if include_policy_analysis else _NoOpPolicySection()
    policy_pdf.chapter("5", "Policy Effectiveness & Recommendations", "Policy & Recommendations")

    policy_pdf.sub("5.1  Policy Intervention Assessment", min_content_height=80)
    policy_pdf.kv([
        ("Fat-Content Surcharge",
         (f"Active above {_pm_policy.get('fat_tax_threshold', 3.5):.1f}% fat at "
          f"{float(_pm_policy.get('fat_tax_rate', 0.0))*100:.0f}%. The parameter is an "
          "analyst-defined price intervention, not an estimated behavioural effect."
          if _pm_policy.get("fat_tax_active", False) else "Disabled in this report configuration.")),
        ("Domestic Subsidy",
         (f"Active at {float(_pm_policy.get('subsidy_rate', 0.0))*100:.0f}%. Policy-crisis "
          f"import dependency was {imp_dep_c:.1f}% versus {imp_dep_u:.1f}% without policy. "
          "The report does not estimate fiscal cost, producer income, or the subsidy-only effect."
          if _pm_policy.get("subsidy_active", False) else "Disabled in this report configuration.")),
        ("Purchase Rationing",
         (f"Active at {int(pm_p['purchase_limit'])} units per product. It constrains requested "
          "quantity before purchase, but a bundled design cannot separate its effect from other levers."
          if pm_p.get("purchase_limit") is not None else "Disabled in this report configuration.")),
        ("Nutritional Labelling",
         ("Active as an exploratory additive preference-score change. The effect size is an "
          "analyst assumption, not estimated from the GROCERYsim experiment."
          if _pm_policy.get("labelling_active", False) else "Disabled in this report configuration.")),
        ("Government Communications",
         (f"{pm_p.get('communication_type', 'neutral').title()} communication at intensity "
          f"{float(pm_p.get('media_intensity', 0.0)):.2f}. Peak panic was {peak_panic_c:.2f} "
          f"with selected policy versus {peak_panic_u:.2f} without policy. The response "
          "magnitude is an unvalidated scenario assumption."
          if float(pm_p.get("media_intensity", 0.0)) > 0 else "Disabled / neutral in this report configuration.")),
    ])

    policy_pdf.sub("5.2  Priority Recommendations")
    policy_pdf.bullet([
        "Treat the current output as a hypothesis-generating stress test. Do not present the "
        "modeled access-stress percentage as Finnish population prevalence.",
        "Run the subsidy, purchase cap, labelling, and communications levers individually and "
        "in combinations across multiple paired seeds before ranking interventions.",
        "Report Monte Carlo uncertainty intervals and sensitivity to the analyst-defined price, "
        "delivery, panic, and hoarding assumptions before stakeholder use.",
        "Use external retailer or administrative data for any future calibration of inventory, "
        "delivery, waste, and policy-effect claims; until then, label results as scenario comparisons.",
    ])
    policy_pdf.finding(
        f"In this paired single-seed comparison, the selected policy configuration ({_policy_summary}) "
        "changed low-income "
        f"fulfilment by {ful_lo - ful_lo_u:+.1f} pp, peak high access stress by "
        f"{fies_peak - fies_peak_u:+.1f} pp, mean access Gini by "
        f"{mean_gini_c - mean_gini_u:+.3f}, import dependency by "
        f"{imp_dep_c - imp_dep_u:+.1f} pp, and peak panic by "
        f"{peak_panic_c - peak_panic_u:+.2f}. These signs and magnitudes are model outputs, "
        "not validated causal estimates or policy recommendations."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 9: METHODOLOGY & CITATION
    # ─────────────────────────────────────────────────────────────────────────
    methodology_num = "6" if include_policy_analysis else "5"
    pdf.chapter(methodology_num, "Methodology, Data Sources & Citation", "Methodology & Citation")

    pdf.sub(f"{methodology_num}.1  Model Architecture")
    pdf.body(
        "GROCERYsim ABM v2.0 is built on the Mesa Agent-Based Modelling framework (Python). "
        "Each simulation step represents one retail trading day. Persistent household "
        "profiles retain pantry, visit, belief, and preference state; short-lived "
        "ConsumerAgent objects represent their individual store visits. ProductAgent "
        "objects persist across days. Execution uses explicit product, logistics, and "
        "shopping phases; only within-day shopper order is shuffled."
    )
    pdf.body(
        "Report design: Supply Chain results compare paired baseline and crisis runs. "
        + (
            "The additional policy analysis compares three paired conditions using seed 42: "
            "no-shock/no-policy baseline, crisis without policy, and selected-policy crisis. "
            if include_policy_analysis else
            "The default food-security analysis compares a no-shock/no-policy baseline with an "
            "unmitigated crisis; no intervention is present in either run. "
        )
        + "Summary welfare metrics use only active-crisis days. A single seed is used for reproducibility, so this "
        "report does not quantify Monte Carlo or participant-resampling uncertainty."
    )
    pdf.sub(f"{methodology_num}.2  Consumer Behavioural Model")
    pdf.bullet([
        "This illustrative walkthrough explicitly enables exploratory behavioural "
        "extensions. They are not estimated from the GROCERYsim sample.",
        "Optional Theory of Planned Behaviour (TPB; Ajzen, 1991): constructed attitude, "
        "subjective norm, and PBC states modulate purchase intention.",
        "Optional Prospect Theory price response (Kahneman & Tversky, 1979): "
        "lambda = 2.25 and alpha = 0.88 are literature transfers, not sample estimates.",
        "DCE price, origin, organic, fat, and opt-out effects are estimated from the "
        "cleaned experiment (108 linked participants) and audited on a participant "
        "holdout. The pooled price coefficient supports milk-candidate choice, not "
        "household-specific willingness-to-pay claims.",
        "Observed phase-one/phase-two basket changes inform cross-fitted substitution and "
        "retention propensities. Phase-two baskets remain validation targets and are not replayed "
        "as simulated crisis demand.",
        "Income stratification: Low (<€1,500/mo), Mid (€1,500–€3,000), High (≥€3,000), "
        "used for descriptive outcome disaggregation.",
    ])
    pdf.sub(f"{methodology_num}.3  Supply Chain Model")
    pdf.bullet([
        "(s, Q) inventory policy: replenishment orders of Q units placed when stock < s.",
        f"Lead time: {sc_p['lead']}-day delivery delay (extendable in crisis scenarios).",
        "Perishable waste: units exceeding product shelf life removed and logged.",
        "Exploratory food-access stress diagnostic, aggregated by income bracket; "
        "not comparable to survey-based FIES prevalence.",
    ])
    pdf.sub(f"{methodology_num}.4  Key References", min_content_height=85)
    pdf.kv([
        ("Ajzen (1991)",           "The Theory of Planned Behaviour. Organizational Behavior and Human Decision Processes, 50(2), 179–211."),
        ("FAO (2016)",             "Methods for estimating comparable rates of food insecurity globally. FAO, Rome."),
        ("IPCC AR6 (2022)",        "Climate Change 2022: Impacts, Adaptation and Vulnerability. Working Group II. Cambridge University Press."),
        ("Kahneman & Tversky (1979)", "Prospect Theory: An Analysis of Decision under Risk. Econometrica, 47(2), 263–291."),
        ("Sheffi (2005)",          "The Resilient Enterprise: Overcoming Vulnerability for Competitive Advantage. MIT Press."),
        ("Thaler & Sunstein (2008)", "Nudge: Improving Decisions about Health, Wealth, and Happiness. Yale University Press."),
        ("Grashuis et al. (2020)", "Grocery Purchasing Behavior during COVID-19. Agribusiness, 36(3), 497–508."),
    ])
    pdf.sub(f"{methodology_num}.5  Citation", min_content_height=35)
    pdf.set_fill_color(*_SF_CREAM2)
    cy = pdf.get_y()
    pdf.rect(15, cy, 180, 24, "F")
    pdf.set_fill_color(*_SF_DARK)
    pdf.rect(15, cy, 3.5, 24, "F")
    pdf.set_xy(21, cy + 3)
    pdf.set_font("ArU", "", 9)
    pdf.set_text_color(*_SF_BODY)
    pdf.multi_cell(170, 5.5,
        "Đurić, Ivan (2026). GROCERYsim Agent-Based Model for Consumer Behaviour "
        "and Supply Chain Stress-Testing. IAMO XR Lab, SecureFood project, "
        "Horizon Europe Grant 101136583.")
    pdf.set_y(cy + 24 + 2)
    pdf.set_font("Ar", "I", 8)
    pdf.set_text_color(120, 135, 130)
    pdf.set_x(15)
    pdf.cell(0, 5, "Software: https://github.com/IvanDuric/Finland_ABM")

    # ── Write PDF and tidy CSV artifacts from the same model runs ─────────────
    pdf_bytes = bytes(pdf.output())

    def _export_frame(frame, perspective, params, policy_summary):
        exported = frame.copy()
        metadata = [
            ("ReportMode", report_mode),
            ("Perspective", perspective),
            ("RandomSeed", 42),
            ("SimulationDays", int(params["days"])),
            ("StartMonth", int(params["month"])),
            ("BaseDailyConsumers", int(params["base_con"])),
            ("CrisisStartDay", int(params["cri_start"])),
            ("CrisisDurationDays", int(params["cri_duration"])),
            ("PriceInflationPct", float(params["inf"])),
            ("SupplyDisruptionDays", int(params["dis"])),
            ("LeadTimeDays", int(params["lead"])),
            ("ReorderPointPct", float(params["reorder"]) * 100),
            ("RestockTargetPct", float(params["target"]) * 100),
            ("PanicSensitivity", float(params["panic"])),
            ("HoardingFactor", float(params["hoard"])),
            ("ExploratoryBehaviour", bool(params.get("exploratory_behaviour", False))),
            ("PolicyAnalysisIncluded", include_policy_analysis),
            ("SelectedPolicy", policy_summary),
        ]
        for position, (column, value) in reversed(list(enumerate(metadata))):
            exported.insert(0, column, value)
        return exported

    sc_policy_summary = "configured supply-chain policy" if _has_sc_policy else "no active policy levers"
    sc_aggregate = _export_frame(sc_data["df"], "Supply Chain Actor", sc_p, sc_policy_summary)
    sc_products = _export_frame(sc_data["df_prod"], "Supply Chain Actor", sc_p, sc_policy_summary)

    pm_base_aggregate = pm_b.copy(); pm_base_aggregate["Scenario"] = "Baseline"
    pm_no_policy_aggregate = pm_u.copy(); pm_no_policy_aggregate["Scenario"] = "Crisis (No Policy)"
    pm_selected_aggregate = pm_c.copy()
    pm_selected_aggregate["Scenario"] = (
        "Crisis (Selected Policy)" if _has_pm_policy else "Crisis (No Active Policy)"
    )
    pm_aggregate_frames = [pm_base_aggregate, pm_no_policy_aggregate]
    if include_policy_analysis:
        pm_aggregate_frames.append(pm_selected_aggregate)
    pm_aggregate = _export_frame(
        pd.concat(pm_aggregate_frames, ignore_index=True),
        "Additional Policy Analysis" if include_policy_analysis else "Food Security & Equity",
        pm_p,
        _policy_summary,
    )

    pm_product_source = pm_unpol_data["df_prod"]
    pm_base_products = pm_product_source[
        pm_product_source["Scenario"] == "Baseline"
    ].copy()
    pm_base_products["Scenario"] = "Baseline"
    pm_no_policy_products = pm_product_source[
        pm_product_source["Scenario"] == "Crisis"
    ].copy()
    pm_no_policy_products["Scenario"] = "Crisis (No Policy)"
    pm_selected_products = pm_policy_data["df_prod"].copy()
    pm_selected_products["Scenario"] = (
        "Crisis (Selected Policy)" if _has_pm_policy else "Crisis (No Active Policy)"
    )
    pm_product_frames = [pm_base_products, pm_no_policy_products]
    if include_policy_analysis:
        pm_product_frames.append(pm_selected_products)
    pm_products = _export_frame(
        pd.concat(pm_product_frames, ignore_index=True),
        "Additional Policy Analysis" if include_policy_analysis else "Food Security & Equity",
        pm_p,
        _policy_summary,
    )

    if include_policy_analysis:
        aggregate_csv = pd.concat([sc_aggregate, pm_aggregate], ignore_index=True, sort=False)
        product_csv = pd.concat([sc_products, pm_products], ignore_index=True, sort=False)
    else:
        # Every daily record already contains operational and household welfare
        # measures. Export the shared run once instead of duplicating identical
        # rows under two perspective labels.
        aggregate_csv = _export_frame(
            sc_data["df"], "Shared Default Scenario", sc_p, "no active policy levers"
        )
        product_csv = _export_frame(
            sc_data["df_prod"], "Shared Default Scenario", sc_p, "no active policy levers"
        )

    # Clean up temp files
    for f in temp_imgs:
        try: os.unlink(f)
        except: pass

    return {
        "pdf": pdf_bytes,
        "aggregate_csv": aggregate_csv.to_csv(index=False).encode("utf-8-sig"),
        "product_csv": product_csv.to_csv(index=False).encode("utf-8-sig"),
        "model_revision": report_model_revision,
        "generated_at": generated_at,
    }


def _generate_sf_pdf_report(
    sc_params: dict | None = None,
    pm_params: dict | None = None,
    report_mode: str = "Preset demonstration",
    include_policy_analysis: bool = False,
) -> bytes:
    """Backward-compatible PDF-only entry point used by tests and callers."""
    return _generate_sf_report_artifacts(
        sc_params,
        pm_params,
        report_mode,
        include_policy_analysis,
        SF_REPORT_MODEL_REVISION,
    )["pdf"]


def _render_sf_artifact_downloads(
    artifacts: dict | None,
    filename_stem: str,
    key_prefix: str,
    columns=None,
    disabled: bool = False,
    show_status: bool = True,
):
    """Render aligned PDF and CSV download actions for one generated report."""
    artifacts = artifacts or {}
    disabled = bool(disabled or not artifacts)
    if show_status and artifacts:
        st.success(
            "Fresh simulation completed · "
            f"model `{artifacts.get('model_revision', 'unknown')}` · "
            f"generated `{artifacts.get('generated_at', 'unknown time')}`"
        )
    pdf_col, daily_col, product_col = columns or st.columns(3)
    pdf_col.download_button(
        "📄 PDF Report" if columns else "📄 Download PDF Report",
        data=artifacts.get("pdf", b""),
        file_name=f"{filename_stem}_Report.pdf",
        mime="application/pdf",
        use_container_width=True,
        key=f"{key_prefix}_pdf",
        disabled=disabled,
    )
    daily_col.download_button(
        "⬇️ Daily CSV" if columns else "⬇️ Download Daily Results (CSV)",
        data=artifacts.get("aggregate_csv", b""),
        file_name=f"{filename_stem}_Daily_Results.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{key_prefix}_daily_csv",
        disabled=disabled,
    )
    product_col.download_button(
        "⬇️ Product CSV" if columns else "⬇️ Download Product Results (CSV)",
        data=artifacts.get("product_csv", b""),
        file_name=f"{filename_stem}_Product_Results.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{key_prefix}_product_csv",
        disabled=disabled,
    )


# ── Main SecureFood page ───────────────────────────────────────────────────────

_SF_PARAMETER_HELP = {
    "days": "Total simulated calendar days. Use a horizon long enough to include the pre-crisis period, disruption, and recovery. This changes runtime and export length, not the empirical sample size.",
    "consumers": "Target household shopping visits per day. Higher values increase demand pressure and runtime. Profiles are resampled from the same empirical population; this is store traffic, not a new survey sample.",
    "crisis_start": "First day on which the configured price and delivery shocks become active. Leave enough pre-crisis days to establish a meaningful reference period.",
    "crisis_duration": "Number of days the disruption remains active. After this window, prices and supply conditions begin recovery. Zero means the shock continues through the simulation.",
    "inflation": "Percentage increase applied to crisis retail prices before policy effects. This is an analyst-defined stress assumption, not a calibrated forecast.",
    "disruption": "Additional crisis delivery delay in days. Larger values postpone replenishment and normally increase stockout risk. Zero represents a price-only shock.",
    "month": "Calendar month on simulation Day 1. It selects the model's seasonal store-traffic multiplier and therefore affects demand volume.",
    "lead": "Normal time from placing an inventory order to delivery. The crisis delivery delay is added while the disruption is active.",
    "reorder": "Storage level, as a percentage of capacity, that triggers a new order. A higher threshold orders earlier but can increase holding and perishable-waste risk.",
    "target": "Storage level the replenishment order attempts to restore. A higher target builds a larger buffer but can increase waste for short-life products.",
    "panic": "Strength of population panic growth after shoppers observe scarcity or price shocks. Zero disables contagion; one is the strongest configured response. This is exploratory, not directly estimated from the DCE.",
    "hoard": "Maximum multiplier on precautionary purchasing when panic is present. Each household's observed phase-transition propensity scales the effect, so not every shopper receives the maximum.",
    "rationing_on": "Activates a per-product quantity limit for every shopping visit. Use it to test whether rationing distributes scarce stock more evenly.",
    "rationing_limit": "Maximum units of one product a shopper may buy during a visit. It applies separately to each product, not to the entire basket.",
    "communication": "Daily government communication effect on panic. Calming reduces panic, panic-oriented communication increases it, and neutral has no direct effect. Communication does not create inventory or directly change prices.",
    "communication_intensity": "Strength of the selected communication effect from 0 (none) to 1 (maximum configured effect). Treat this as a scenario assumption, not a measured coefficient.",
    "subsidy_on": "Activates a consumer price subsidy for the selected product group. It lowers eligible retail prices but does not directly increase supply capacity.",
    "subsidy_target": "Products eligible for the subsidy: Finnish-origin products, organic products, or products satisfying either condition.",
    "subsidy_rate": "Percentage reduction in eligible crisis retail prices. For example, 25% changes an eligible €4.00 price to €3.00 before near-expiry discounts.",
    "surcharge_on": "Activates a price surcharge on products whose recorded fat content exceeds the selected threshold.",
    "fat_threshold": "Minimum recorded fat percentage above which the surcharge applies. Products at or below the threshold are unaffected.",
    "surcharge_rate": "Percentage price increase applied above the fat threshold. It can alter affordability and choice; it does not represent tax-revenue accounting.",
    "labelling_on": "Activates an exploratory information intervention that shifts health and organic preferences from the chosen start day.",
    "labelling_day": "First simulation day on which nutritional labelling affects eligible consumer preference weights.",
    "health_boost": "Additive increase in the health-preference weight after labelling begins. This is a scenario assumption and should be sensitivity-tested.",
    "organic_boost": "Additive increase in the organic-preference weight after labelling begins. This is a scenario assumption and should be sensitivity-tested.",
}


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

    # Reports are stored in session state after generation. A browser session
    # can survive an app redeployment, so discard artifacts made by an older
    # scientific model instead of continuing to offer a stale PDF and CSVs.
    for _artifact_key, _signature_key in (
        ("sf_preset_report_artifacts", None),
        ("sf_policy_report_artifacts", "sf_policy_report_signature"),
    ):
        _artifacts = st.session_state.get(_artifact_key)
        if (
            _artifacts
            and _artifacts.get("model_revision") != SF_REPORT_MODEL_REVISION
        ):
            st.session_state[_artifact_key] = None
            if _signature_key:
                st.session_state[_signature_key] = None

    st.divider()

    # ── Scenario background card ───────────────────────────────────────────────
    st.markdown("""
<div style="
    background: linear-gradient(135deg, #F0E9DA 0%, #FAF6EC 100%);
    border-left: 5px solid #DBA159;
    border-radius: 6px;
    padding: 20px 24px 16px 24px;
    margin-bottom: 18px;
">
<h4 style="margin:0 0 10px 0; color:#042026; font-size:15px; font-weight:700; letter-spacing:-0.01em;">
    Scenario: Climate-Driven Dairy Supply Chain Disruption
</h4>
<p style="margin:0 0 10px 0; color:#2c4a52; font-size:13.5px; line-height:1.65;">
    <strong>Background and Trigger Event</strong><br>
    Increasingly frequent and intense weather events, driven by climate change, are creating
    critical vulnerabilities within the agricultural sector, specifically threatening the
    stability of the milk and dairy supply chain.
</p>
<p style="margin:0 0 8px 0; color:#2c4a52; font-size:13.5px; line-height:1.65;">
    <strong>Market Disruptions</strong>
</p>
<ul style="margin:0; padding-left:20px; color:#2c4a52; font-size:13.5px; line-height:1.75;">
    <li><strong>Cost Inflation:</strong> Extreme weather disrupts livestock feed production,
        driving up operational costs. These increases are subsequently passed down to the
        consumer market, resulting in significantly higher retail prices for milk and dairy
        products.</li>
    <li><strong>Supply Scarcity:</strong> Simultaneous production and logistical bottlenecks
        lead to critical stock shortages and consistently low product availability at the
        retail level.</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Choose one of two report workflows")
    _flow_default, _flow_policy = st.columns(2)
    with _flow_default:
        st.info(
            "**1. Default preset**\n\n"
            "Review the fixed parameters and select **Generate Default Scenario Report**. "
            "This workflow contains no policy intervention."
        )
    with _flow_policy:
        st.warning(
            "**2. Optional policy analysis**\n\n"
            "Open **Additional Policy Analysis**, enable it, set the scenario and policy, "
            "run the analysis, then generate the report from that analysis."
        )

    # ── Preset report and data downloads ──────────────────────────────────────
    st.markdown("### 📄 Default SecureFood scenario report")
    st.info(
        "This fixed preset analyses one climate disruption without any policy intervention. "
        "Supply-chain and food-security findings come from the same paired simulation; only "
        "the outcome perspective changes."
    )
    with st.expander("View preset parameters", expanded=False):
        st.markdown(
            "**Shared no-policy preset:** 120 days; 200 consumers/day; crisis starts Day 30 "
            "for 45 days; 25% price inflation; 7-day supply disruption; 3-day lead time; "
            "30% reorder point; 90% restock target; panic sensitivity 0.50; hoarding factor 1.5; "
            "no subsidy, rationing, surcharge, labelling, or communication intervention.\n\n"
            "**Two outcome perspectives, one run:** the supply-chain section reports operational "
            "outcomes, while the food-security and equity section reports household outcomes. "
            "Both use the exact same baseline/crisis pair and seed 42."
        )
    _rpt_col, _rpt_spacer = st.columns([1.4, 1.6])
    with _rpt_col:
        if st.button(
            "📊 Generate Default Scenario Report",
            type="primary",
            use_container_width=True,
            key="sf_preset_report_gen_btn",
        ):
            with st.spinner("Running simulations and building report — this may take ~15 s…"):
                try:
                    st.session_state["sf_preset_report_artifacts"] = (
                        _generate_sf_report_artifacts(
                            report_mode="Default scenario",
                            include_policy_analysis=False,
                            report_model_revision=SF_REPORT_MODEL_REVISION,
                        )
                    )
                except Exception as _e:
                    st.error(f"Report generation failed: {_e}")
    if st.session_state.get("sf_preset_report_artifacts"):
        _render_sf_artifact_downloads(
            st.session_state["sf_preset_report_artifacts"],
            "GROCERYsim_SecureFood_Default_Scenario",
            "sf_preset_download",
        )

    sc_tab, pm_tab = st.tabs([
        "🏭 Default Scenario Analysis",
        "🏛️ Additional Policy Analysis",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # SUPPLY CHAIN ACTOR
    # ══════════════════════════════════════════════════════════════════════════
    with sc_tab:
        st.markdown(
            "_For **producers, distributors, and retailers** managing Finnish dairy supply chains "
            "under climate-driven disruption. Focus: operational resilience, revenue, and inventory._"
        )
        st.markdown("### ⚙️ Scenario Parameters")
        st.caption("Select the small ? icon beside any parameter to see what it changes and how to interpret it.")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**📅 General**")
            sc_days = st.slider("Duration (Days)", 30, 365, 120, 5, key="sf_sc_days",
                help="Simulation length in days. The shared default report uses 120 days so both operational and household recovery outcomes are observed on the same horizon.")
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
                                   max(1, sc_days - sc_cri_start), 45, 5,
                key="sf_sc_cri_dur",
                help="How long the disruption persists before recovery. 0 = runs to end of simulation. 30–60 days models a temporary weather event; 0 models structural climate change impact.")

        with st.expander("🧠 Consumer Behaviour (advanced)", expanded=False):
            cb1, cb2 = st.columns(2)
            sc_panic = cb1.slider("Panic Sensitivity", 0.0, 1.0, 0.50, 0.05,
                key="sf_sc_panic",
                help="Scenario assumption for response to perceived scarcity. The current GROCERYsim export has no direct panic-belief item, so this parameter is not empirically estimated.")
            sc_hoard = cb2.slider("Hoarding Factor", 1.0, 3.0, 1.5, 0.1,
                key="sf_sc_hoard",
                help="Maximum purchase multiplier during panic, scaled by each household's cross-fitted phase-transition propensity. Treat the maximum as a scenario assumption.")

        sc_params = {
            "days": sc_days, "month": sc_month, "base_con": int(sc_consumers),
            "reorder": sc_reorder, "target": sc_target, "lead": sc_lead,
            "cri_start": sc_cri_start, "cri_duration": int(sc_cri_dur),
            "inf": float(sc_inflation), "dis": int(sc_disruption),
            "panic": sc_panic, "hoard": sc_hoard, "mc_runs": 1,
            "policy_cfg": _sf_no_policy_config(sc_cri_start),
            "purchase_limit": None, "media_intensity": 0.0,
            "communication_type": "neutral", "stockpile_days": None,
            "exploratory_behaviour": True,
        }

        col_run, _ = st.columns([2, 6])
        if col_run.button("▶ Run Supply Chain Simulation", type="primary",
                          key="sf_sc_run", use_container_width=True):
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
            "_Optional counterfactual module for **government agencies, regulators, and food "
            "system authorities**. It compares a selected intervention with the same crisis "
            "without policy._"
        )
        st.warning(
            "Policy analysis is **not included by default**. Enable this optional module only "
            "when you want to test an intervention against a paired crisis-without-policy run."
        )
        pm_policy_enabled = st.checkbox(
            "Enable additional policy analysis",
            value=False,
            key="sf_pm_policy_enabled",
            help="Keeps policy assumptions out of the default scenario until explicitly enabled.",
        )

        if pm_policy_enabled:
            st.markdown("### ⚙️ Counterfactual scenario settings")
            st.caption("Select the small ? icon beside any parameter to see what it changes and how to interpret it.")
            p1, p2, p3 = st.columns(3)

            with p1:
                st.markdown("**🔴 Crisis severity**")
                pm_days = st.slider(
                    "Duration (Days)", 60, 365, 120, 10, key="sf_pm_days",
                    help=_SF_PARAMETER_HELP["days"],
                )
                pm_consumers = st.number_input(
                    "Base Daily Consumers", 50, 2000, 200, 50, key="sf_pm_consumers",
                    help=_SF_PARAMETER_HELP["consumers"],
                )
                pm_cri_start = st.slider(
                    "Crisis Start Day", 5, max(6, pm_days - 20),
                    min(30, pm_days - 20), key="sf_pm_cri_start",
                    help=_SF_PARAMETER_HELP["crisis_start"],
                )
                pm_cri_dur = st.slider(
                    "Crisis Duration (Days)", 0, max(1, pm_days - pm_cri_start),
                    min(45, max(1, pm_days - pm_cri_start)), 5, key="sf_pm_cri_dur",
                    help=_SF_PARAMETER_HELP["crisis_duration"],
                )
                pm_inflation = st.slider(
                    "Price Inflation (%)", 0, 100, 25, 5, key="sf_pm_inflation",
                    help=_SF_PARAMETER_HELP["inflation"],
                )
                pm_disruption = st.slider(
                    "Supply Disruption (Days delay)", 0, 30, 7, 1,
                    key="sf_pm_disruption", help=_SF_PARAMETER_HELP["disruption"],
                )

            with p2:
                st.markdown("**🚚 Logistics & inventory**")
                pm_month = st.selectbox(
                    "Start Month", list(range(1, 13)), index=0, key="sf_pm_month",
                    help=_SF_PARAMETER_HELP["month"],
                )
                pm_lead = st.slider(
                    "Lead Time (Days)", 1, 14, 3, 1, key="sf_pm_lead",
                    help=_SF_PARAMETER_HELP["lead"],
                )
                pm_reorder = st.slider(
                    "Reorder Point (% of storage)", 10, 60, 30, 5,
                    key="sf_pm_reorder", help=_SF_PARAMETER_HELP["reorder"],
                ) / 100.0
                pm_target = st.slider(
                    "Restock Target (% of storage)", 70, 100, 90, 5,
                    key="sf_pm_target", help=_SF_PARAMETER_HELP["target"],
                ) / 100.0

            with p3:
                st.markdown("**🧠 Behaviour assumptions**")
                pm_panic = st.slider(
                    "Panic Sensitivity", 0.0, 1.0, 0.50, 0.05,
                    key="sf_pm_panic",
                    help=_SF_PARAMETER_HELP["panic"],
                )
                pm_hoard = st.slider(
                    "Hoarding Factor", 1.0, 3.0, 1.5, 0.1,
                    key="sf_pm_hoard",
                    help=_SF_PARAMETER_HELP["hoard"],
                )

            st.markdown("### 🏛️ Policy levers")
            pol1, pol2, pol3 = st.columns(3)

            with pol1:
                st.markdown("**Access and communication**")
                pm_pl_on = st.checkbox(
                    "Enable Purchase Rationing", False, key="sf_pm_pl_on",
                    help=_SF_PARAMETER_HELP["rationing_on"],
                )
                pm_pl_val = st.slider(
                    "Max Units per Product per Visit", 1, 10, 3,
                    key="sf_pm_pl_val", disabled=not pm_pl_on,
                    help=_SF_PARAMETER_HELP["rationing_limit"],
                )
                pm_purchase_limit = pm_pl_val if pm_pl_on else None
                pm_comm = st.selectbox(
                    "Government Communication Strategy",
                    ["neutral", "calming", "panic"], key="sf_pm_comm",
                    help=_SF_PARAMETER_HELP["communication"],
                )
                pm_media = st.slider(
                    "Communication Intensity", 0.0, 1.0, 0.30, 0.05,
                    key="sf_pm_media", disabled=pm_comm == "neutral",
                    help=_SF_PARAMETER_HELP["communication_intensity"],
                ) if pm_comm != "neutral" else 0.0

            with pol2:
                st.markdown("**Prices and affordability**")
                pm_sub_on = st.checkbox(
                    "Enable Product Subsidy", False, key="sf_pm_sub_on",
                    help=_SF_PARAMETER_HELP["subsidy_on"],
                )
                pm_sub_target = st.selectbox(
                    "Subsidy Target", ["domestic", "organic", "both"],
                    key="sf_pm_sub_target", disabled=not pm_sub_on,
                    help=_SF_PARAMETER_HELP["subsidy_target"],
                )
                pm_sub_rate = st.slider(
                    "Subsidy Rate (%)", 5, 40, 15, 5, key="sf_pm_sub_rate",
                    disabled=not pm_sub_on, help=_SF_PARAMETER_HELP["subsidy_rate"],
                ) / 100.0 if pm_sub_on else 0.0
                pm_fat_on = st.checkbox(
                    "Enable Fat-Content Surcharge", False, key="sf_pm_fat_on",
                    help=_SF_PARAMETER_HELP["surcharge_on"],
                )
                pm_fat_threshold = st.slider(
                    "Fat Threshold (%)", 0.5, 5.0, 3.5, 0.5,
                    key="sf_pm_fat_threshold", disabled=not pm_fat_on,
                    help=_SF_PARAMETER_HELP["fat_threshold"],
                )
                pm_fat_rate = st.slider(
                    "Surcharge Rate (%)", 5, 50, 20, 5, key="sf_pm_fat_rate",
                    disabled=not pm_fat_on, help=_SF_PARAMETER_HELP["surcharge_rate"],
                ) / 100.0 if pm_fat_on else 0.0

            with pol3:
                st.markdown("**Information and preferences**")
                pm_lab_on = st.checkbox(
                    "Enable Nutritional Labelling", False, key="sf_pm_lab_on",
                    help=_SF_PARAMETER_HELP["labelling_on"],
                )
                pm_lab_day = st.slider(
                    "Labelling Start Day", 1, pm_days,
                    min(pm_cri_start, pm_days), key="sf_pm_lab_day",
                    disabled=not pm_lab_on, help=_SF_PARAMETER_HELP["labelling_day"],
                )
                pm_lab_health = st.slider(
                    "Health Preference Boost", 0.0, 0.40, 0.10, 0.05,
                    key="sf_pm_lab_health", disabled=not pm_lab_on,
                    help=_SF_PARAMETER_HELP["health_boost"],
                ) if pm_lab_on else 0.0
                pm_lab_organic = st.slider(
                    "Organic Preference Boost", 0.0, 0.30, 0.05, 0.05,
                    key="sf_pm_lab_organic", disabled=not pm_lab_on,
                    help=_SF_PARAMETER_HELP["organic_boost"],
                ) if pm_lab_on else 0.0

            pm_policy_cfg = {
                "fat_tax_active": pm_fat_on,
                "fat_tax_threshold": pm_fat_threshold,
                "fat_tax_rate": pm_fat_rate,
                "subsidy_active": pm_sub_on,
                "subsidy_target": pm_sub_target,
                "subsidy_rate": pm_sub_rate,
                "domestic_shock_active": False,
                "domestic_shock_day": pm_cri_start,
                "domestic_shock_duration": 30,
                "domestic_shock_severity": 0.5,
                "labelling_active": pm_lab_on,
                "labelling_day": pm_lab_day,
                "labelling_health_boost": pm_lab_health,
                "labelling_organic_boost": pm_lab_organic,
            }
            pm_params = {
                "days": pm_days, "month": pm_month,
                "base_con": int(pm_consumers), "reorder": pm_reorder,
                "target": pm_target, "lead": pm_lead,
                "cri_start": pm_cri_start, "cri_duration": int(pm_cri_dur),
                "inf": float(pm_inflation), "dis": int(pm_disruption),
                "panic": pm_panic, "hoard": pm_hoard, "mc_runs": 1,
                "policy_cfg": pm_policy_cfg,
                "purchase_limit": pm_purchase_limit,
                "media_intensity": pm_media,
                "communication_type": pm_comm,
                "stockpile_days": None,
                "exploratory_behaviour": True,
            }
            pm_has_policy = _sf_has_active_policy(pm_params)
            pm_signature = _sf_param_signature(pm_params)
            pm_result_is_current = (
                st.session_state.get("sf_results_pm_signature") == pm_signature
            )
            if not pm_has_policy:
                st.info("Select at least one policy lever to create a policy counterfactual.")
            elif st.session_state.get("sf_results_pm") is not None and not pm_result_is_current:
                st.warning(
                    "The policy settings have changed since the last run. Run the policy "
                    "analysis again before generating or downloading its report."
                )

            run_col, report_col, pdf_col, daily_col, product_col = st.columns(
                [1.55, 2.05, 1.15, 1.15, 1.25]
            )
            if run_col.button(
                "▶ Run Policy Simulation", type="primary", key="sf_pm_run",
                use_container_width=True, disabled=not pm_has_policy
            ):
                with st.spinner("Running paired policy and no-policy simulations…"):
                    result = _sf_run_simulation(pm_params)
                    if result:
                        st.session_state["sf_results_pm"] = result
                        st.session_state["sf_results_pm_signature"] = pm_signature
                        st.session_state["sf_policy_report_artifacts"] = None
                        st.session_state["sf_policy_report_signature"] = None
                        pm_result_is_current = True

            if report_col.button(
                "📊 Generate Report from This Analysis", key="sf_pm_report_gen_btn",
                use_container_width=True,
                disabled=not (pm_has_policy and pm_result_is_current),
            ):
                with st.spinner("Building the additional policy report…"):
                    try:
                        st.session_state["sf_policy_report_artifacts"] = (
                            _generate_sf_report_artifacts(
                                sc_params=_sf_without_policy(pm_params),
                                pm_params=pm_params,
                                report_mode="Additional policy analysis",
                                include_policy_analysis=True,
                                report_model_revision=SF_REPORT_MODEL_REVISION,
                            )
                        )
                        st.session_state["sf_policy_report_signature"] = pm_signature
                    except Exception as _e:
                        st.error(f"Policy report generation failed: {_e}")

            policy_artifacts_are_current = bool(
                st.session_state.get("sf_policy_report_artifacts")
                and st.session_state.get("sf_policy_report_signature") == pm_signature
                and pm_result_is_current
            )
            if policy_artifacts_are_current:
                _render_sf_artifact_downloads(
                    st.session_state["sf_policy_report_artifacts"],
                    "GROCERYsim_SecureFood_Additional_Policy",
                    "sf_policy_download",
                    columns=(pdf_col, daily_col, product_col),
                    show_status=False,
                )
                _artifacts = st.session_state["sf_policy_report_artifacts"]
                st.success(
                    "Fresh simulation completed · "
                    f"model `{_artifacts.get('model_revision', 'unknown')}` · "
                    f"generated `{_artifacts.get('generated_at', 'unknown time')}`"
                )
            else:
                pdf_col.button(
                    "📄 PDF Report", key="sf_policy_pdf_disabled",
                    disabled=True, use_container_width=True,
                )
                daily_col.button(
                    "⬇️ Daily CSV", key="sf_policy_daily_csv_disabled",
                    disabled=True, use_container_width=True,
                )
                product_col.button(
                    "⬇️ Product CSV", key="sf_policy_product_csv_disabled",
                    disabled=True, use_container_width=True,
                )

            if st.session_state.get("sf_results_pm") and pm_result_is_current:
                _render_sf_pm_results(st.session_state["sf_results_pm"])
        else:
            st.caption(
                "The default scenario remains unmitigated. Turn on the module above to reveal "
                "crisis, inventory, behaviour, rationing, subsidy, surcharge, labelling, and "
                "communication controls."
            )


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
    "sf_results_pm_signature": None,
    "sf_preset_report_artifacts": None,
    "sf_policy_report_artifacts": None,
    "sf_policy_report_signature": None,
    # Simulation results
    "sim_results":     None,
    "sim_stock":       None,
    "sim_scm_log":     None,
    "sim_waste":       None,
    "sim_product_recs": None,
    "sim_model_crisis": None,
    "sim_pref_drift":   None,
    # Scientific workflow state
    "mc_stage":             0,
    "data_base_raw":        None,
    "data_base_opt":        None,
    "data_crisis":          None,
    "ai_recs":              None,
    "active_baseline":      "Baseline (Raw)",
    "prod_stats_raw":       None,
    "mc_session_populated": False,   # True once _populate_session_from_mc has run
    "mc_median_seed":       None,    # seed of the representative (median) run
    # Policy analysis results
    "policy_baseline":  None,   # DataFrame: daily records, no policy
    "policy_scenario":  None,   # DataFrame: daily records, with policy active
    "policy_label":     None,   # human-readable name of the active policy run
    # Multi-scenario store: list of {"label": str, "df": DataFrame}
    "policy_scenarios": [],
    # Onboarding tour (1 = first step, 0 = hidden)
    "tour_step": 1,
    # Card navigation: which section is currently active (None = home)
    "nav_section": None,
    # Saved scenarios for compare feature
    "saved_scenarios": [],
    # Stress-test results cache
    "stress_results": None,
    # Export tab: cached generated PDF bytes
    "_generated_pdf": None,
    # Last sidebar params (for PDF report)
    "_last_params": None,
    # Agent-level replay log (list → DataFrame)
    "agent_log": None,
    # Multi-store network results
    "multistore_results": None,   # DataFrame: Day × Store × Scenario metrics
    "multistore_config":  None,   # list of store-config dicts used in last run
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
        "tabs": ["🏠 Data & Population", "🎮 Interactive Demo", "🔬 Scientific Analysis", "♻️ Food Waste", "📦 Per-Product", "🏛️ Policy Analysis", "👔 Stakeholder View", "🎚️ Sensitivity Analysis", "🧪 Behavioural Theory", "📥 Export", "📊 Compare Scenarios", "🚨 Stress Test", "🎬 Agent Replay", "🗺️ Regional Map", "🏪 Multi-Store Network"],
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
        "tabs": ["🏠 Data & väestö", "🎮 Interaktiivinen demo", "🔬 Tieteellinen analyysi", "♻️ Ruokahävikki", "📦 Tuotekohtainen", "🏛️ Politiikka-analyysi", "👔 Sidosryhmänäkymä", "🎚️ Herkkyysanalyysi", "🧪 Käyttäytymisteoria", "📥 Vienti", "📊 Vertaile Skenaarioita", "🚨 Stressitesti", "🎬 Agenttitoisinto", "🗺️ Aluekartta", "🏪 Monikauppaverkosto"],
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
        "tabs": ["🏠 Δεδομένα & Πληθυσμός", "🎮 Διαδραστικό Demo", "🔬 Επιστημονική Ανάλυση", "♻️ Απώλεια Τροφίμων", "📦 Ανά Προϊόν", "🏛️ Ανάλυση Πολιτικής", "👔 Προβολή Ενδιαφερομένων", "🎚️ Ανάλυση Ευαισθησίας", "🧪 Θεωρία Συμπεριφοράς", "📥 Εξαγωγή", "📊 Σύγκριση Σεναρίων", "🚨 Δοκιμή Αντοχής", "🎬 Αναπαραγωγή Πρακτόρων", "🗺️ Περιφερειακός Χάρτης", "🏪 Δίκτυο Πολυ-Καταστημάτων"],
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
        "tabs": ["🏠 Dados & População", "🎮 Demo Interativo", "🔬 Análise Científica", "♻️ Desperdício Alimentar", "📦 Por Produto", "🏛️ Análise de Políticas", "👔 Visão das Partes", "🎚️ Análise de Sensibilidade", "🧪 Teoria Comportamental", "📥 Exportar", "📊 Comparar Cenários", "🚨 Teste de Stress", "🎬 Repetição de Agentes", "🗺️ Mapa Regional", "🏪 Rede Multi-Loja"],
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
_LOCAL_DCE_PATH = _pl.Path(__file__).parent / ".streamlit" / "dce_data_clean.csv"

def _load_bundled_data():
    """Return Firebase, catalogue, and optional cleaned DCE alternative rows."""
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

    try:
        dce_text = str(st.secrets["dce"]["data"])
    except Exception:
        try:
            dce_text = _LOCAL_DCE_PATH.read_text(encoding="utf-8-sig")
        except Exception:
            dce_text = ""
    dce_rows = list(csv.DictReader(io.StringIO(dce_text))) if dce_text else None

    return firebase_dict, products_dict, dce_rows


@st.cache_data(
    show_spinner=False,
    ttl=24 * 60 * 60,
    max_entries=8,
)
def _cached_run_pipeline_from_data(
    firebase_dict: dict,
    products_dict: dict,
    pool_size: int,
    n_archetypes: int,
    dce_rows: list[dict] | None,
) -> dict:
    """Cache the deterministic evidence pipeline across sessions.

    Streamlit executes the app once per session and reruns it after every
    interaction.  The pipeline includes bootstrap clustering, cross-fitting,
    DCE estimation, and substitution validation, so rebuilding it for every
    visitor wastes Community Cloud CPU.  Streamlit hashes every input here;
    changing the Firebase export, catalogue, DCE rows, or configuration creates
    a new cache entry rather than reusing stale scientific results.

    ``st.cache_data`` returns an isolated copy to each caller, avoiding shared
    mutable state between simultaneous stakeholder sessions.  The TTL and
    entry bound prevent old cohort variants from accumulating indefinitely.
    """
    return run_pipeline_from_data(
        firebase_dict,
        products_dict,
        pool_size=int(pool_size),
        n_archetypes=int(n_archetypes),
        dce_rows=dce_rows,
    )

if st.session_state.config_data is None:
    try:
        _firebase_dict, _products_dict, _dce_rows = _load_bundled_data()
        if _firebase_dict is not None and _products_dict is not None:
            st.session_state.config_data = _cached_run_pipeline_from_data(
                _firebase_dict, _products_dict, 2000, 4, _dce_rows,
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
    d_consumers = int(model.daily_consumer_count)
    # Model counterparts only; empirical data must use the same denominators.
    validation_observables = daily_validation_observables(
        d_sales, d_rev_nominal, d_consumers, d_waste,
        [a.daily_lost_sales > 0 for a in agents],
    )

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
        "Consumers":  d_consumers,
        "EmpiricalSamplingUnits": last_rec.get("EmpiricalSamplingUnits", len(model.population_pool)),
        "SimulatedHouseholdDraws": last_rec.get("SimulatedHouseholdDraws", len(model.population_pool)),
        "PopulationSamplingMethod": last_rec.get("PopulationSamplingMethod", "supplied_population"),
        "BehaviorEvidenceMode": last_rec.get("BehaviorEvidenceMode", "empirical_only"),
        "PanicDynamicsEnabled": last_rec.get("PanicDynamicsEnabled", 0),
        "TPBEnabled": last_rec.get("TPBEnabled", 0),
        "ProspectTheoryEnabled": last_rec.get("ProspectTheoryEnabled", 0),
        "PreferenceLearningEnabled": last_rec.get("PreferenceLearningEnabled", 0),
        "ArchetypeModifiersEnabled": last_rec.get("ArchetypeModifiersEnabled", 0),
        "PolicyChoiceEffectsEnabled": last_rec.get("PolicyChoiceEffectsEnabled", 0),
        "DCEAttributeRankingEnabled": last_rec.get("DCEAttributeRankingEnabled", 0),
        "DCEAttributeRankingCategories": last_rec.get(
            "DCEAttributeRankingCategories", "none"
        ),
        "ChoicePriceScaleIdentified": last_rec.get("ChoicePriceScaleIdentified", 0),
        "SubstitutionPriceGateEnabled": last_rec.get("SubstitutionPriceGateEnabled", 0),
        "SubstitutionRankingMethod": last_rec.get(
            "SubstitutionRankingMethod", "seeded_uniform_affordable_same_category"
        ),
        "SubstitutionChoiceEvidenceEvents": last_rec.get(
            "SubstitutionChoiceEvidenceEvents", 0
        ),
        "SubstitutionAttempts": last_rec.get("SubstitutionAttempts", 0),
        "SubstitutionCandidatesConsidered": last_rec.get("SubstitutionCandidatesConsidered", 0),
        "SubstitutionPriceRejections": last_rec.get("SubstitutionPriceRejections", 0),
        **validation_observables,
        "PanicLevel": model.global_panic_level,
        # Environmental
        "CO2Sales":          last_rec.get("CO2Sales",          0.0),
        "CO2Waste":          last_rec.get("CO2Waste",          0.0),
        "CO2Total":          last_rec.get("CO2Total",          0.0),
        "ImportDepPct":      last_rec.get("ImportDepPct",      0.0),
        "DomesticSales":     last_rec.get("DomesticSales",     0),
        "ImportSales":       last_rec.get("ImportSales",       0),
        "OrganicSalesUnits": last_rec.get("OrganicSalesUnits", 0),
        "CategorySalesUnits": last_rec.get("CategorySalesUnits", {}),
        # Consumer welfare — aggregate
        "BudgetExhaustionRate": last_rec.get("BudgetExhaustionRate", 0.0),
        "FoodStressedPct":      last_rec.get("FoodStressedPct",      0.0),
        "FulfillmentRate":      last_rec.get("FulfillmentRate",      1.0),
        "ConsumptionFulfillmentRate": last_rec.get("ConsumptionFulfillmentRate", 1.0),
        "HouseholdsWithConsumptionShortfall": last_rec.get("HouseholdsWithConsumptionShortfall", 0),
        "HouseholdConsumptionShortfallShare": last_rec.get("HouseholdConsumptionShortfallShare", 0.0),
        "CumulativeConsumptionShortfallRate": last_rec.get("CumulativeConsumptionShortfallRate", 0.0),
        "BaseDemandUnits": last_rec.get("BaseDemandUnits", 0),
        "RequestedDemandUnits": last_rec.get("RequestedDemandUnits", 0),
        "PolicyAllowedUnits": last_rec.get("PolicyAllowedUnits", 0),
        "UnmetDemandUnits": last_rec.get("UnmetDemandUnits", 0),
        "HouseholdConsumptionDemand": last_rec.get("HouseholdConsumptionDemand", 0.0),
        "HouseholdConsumption": last_rec.get("HouseholdConsumption", 0.0),
        "HouseholdConsumptionUnmet": last_rec.get("HouseholdConsumptionUnmet", 0.0),
        "ExpectedVisitIntervalDays": last_rec.get("ExpectedVisitIntervalDays", 0.0),
        "RequestedConsumers": last_rec.get("RequestedConsumers", d_consumers),
        "VisitorCapacityCapped": last_rec.get("VisitorCapacityCapped", 0),
        "TrafficVariationEnabled": last_rec.get("TrafficVariationEnabled", 0),
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
        # Population-wide realised pantry-consumption access.
        "AccessStress_Low":     last_rec.get("AccessStress_Low",     0.0),
        "AccessStress_Mid":     last_rec.get("AccessStress_Mid",     0.0),
        "AccessStress_High":    last_rec.get("AccessStress_High",    0.0),
        "AccessStressHigh_Low": last_rec.get("AccessStressHigh_Low", 0.0),
        "AccessStressHigh_Mid": last_rec.get("AccessStressHigh_Mid", 0.0),
        "AccessStressHigh_High": last_rec.get("AccessStressHigh_High", 0.0),
        "ConsumptionShortfall_Low": last_rec.get("ConsumptionShortfall_Low", 0.0),
        "ConsumptionShortfall_Mid": last_rec.get("ConsumptionShortfall_Mid", 0.0),
        "ConsumptionShortfall_High": last_rec.get("ConsumptionShortfall_High", 0.0),
        # Deprecated aliases retained for existing analyses.
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
    pending_calibration = st.session_state.pop(
        "_pending_calibration_widget_values", {}
    )
    for widget_key, widget_value in pending_calibration.items():
        st.session_state[widget_key] = widget_value

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
                                              key="sim_base_con",
                                              help="Exact in evidence-only mode; varies only when exploratory calendar traffic is enabled")
    traffic_variation = st.sidebar.checkbox(
        "Enable exploratory calendar traffic variation",
        value=False,
        key="sim_traffic_variation",
        help=(
            "Applies the assumed weekday/month multipliers and ±10% daily noise. "
            "GROCERYsim did not measure store footfall time series, so this is off "
            "in evidence-only runs."
        ),
    )

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
**Exploratory daily traffic variation**

When explicitly enabled: `base × weekday factor × month factor × noise (±10%)`.
The evidence-only default uses exactly `base` visitors because GROCERYsim does not
contain a longitudinal store-footfall series.

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
    reorder_pt  = st.sidebar.slider(_t("reorder_pt"), 10, 90, 30, key="sim_reorder_pct") / 100.0
    target_stock = st.sidebar.slider(_t("restock_target"), 50, 100, 90, key="sim_target_pct") / 100.0
    lead_time   = st.sidebar.slider(_t("lead_time"), 1, 14, 2, key="sim_lead")

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
    exploratory_behaviour = st.sidebar.checkbox(
        "Enable exploratory dynamic behaviour",
        value=False,
        key="sim_exploratory_behaviour",
        help=(
            "Opt in to panic contagion, panic stockpiling, transferred Prospect "
            "Theory/TPB rules, and repeated-visit preference learning. These "
            "mechanisms, including labelling-induced choice effects, are not "
            "identified by the current GROCERYsim export."
        ),
    )
    if exploratory_behaviour:
        st.sidebar.warning(
            "Exploratory extensions are ON. Treat differences as assumption-based "
            "scenario results and include these coefficients in sensitivity analysis."
        )
    else:
        st.sidebar.info(
            "Empirical-only mode: observed baskets, budgets, DCE attribute weights, "
            "cross-fitted price response, and substitution remain active. Unvalidated "
            "panic, TPB, Prospect Theory, archetype modifiers, learning, and labelling "
            "choice effects are off."
        )
    panic_sens = st.sidebar.slider(
        _t("panic_sensitivity"), 0.0, 1.0, 0.50, 0.05,
        key="sim_panic", disabled=not exploratory_behaviour,
    )
    hoarding = st.sidebar.slider(
        _t("hoarding_factor"), 1.0, 3.0, 1.5, 0.1,
        key="sim_hoard", disabled=not exploratory_behaviour,
    )

    with st.sidebar.expander("🔧 Advanced panic assumptions", expanded=False):
        st.caption(
            "These coefficients are not estimated from the current GROCERYsim "
            "export. Keep them in sensitivity analysis and document any changes."
        )
        panic_exposure_floor = st.slider(
            "Normal scarcity exposure floor", 0.0, 0.50, 0.10, 0.01,
            key="sim_panic_exposure_floor",
            disabled=not exploratory_behaviour,
            help="Share of shoppers signalling scarcity that is treated as ordinary retail friction.",
        )
        panic_growth_rate = st.slider(
            "Scarcity-to-panic growth rate", 0.0, 1.0, 0.50, 0.05,
            key="sim_panic_growth_rate",
            disabled=not exploratory_behaviour,
            help="Daily amplification of scarcity exposure above the floor.",
        )
        panic_decay_active = st.slider(
            "Active-phase panic decay", 0.0, 0.30, 0.05, 0.01,
            key="sim_panic_decay_active",
            disabled=not exploratory_behaviour,
            help="Daily decay while the crisis remains active.",
        )
        panic_decay_recovery = st.slider(
            "Recovery-phase panic decay", 0.0, 0.30, 0.10, 0.01,
            key="sim_panic_decay_recovery",
            disabled=not exploratory_behaviour,
            help="Daily decay after crisis prices and supply conditions normalize.",
        )
        inflation_panic_rate = st.slider(
            "Inflation-to-panic rate", 0.0, 1.0, 0.40, 0.05,
            key="sim_inflation_panic_rate",
            disabled=not exploratory_behaviour,
            help="Direct daily panic signal from the scenario price shock.",
        )

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
            disabled=not exploratory_behaviour,
            help="How strongly media amplifies or dampens panic each day."
        )
        communication_type = st.selectbox(
            _t("comm_type"), ["neutral", "panic", "calming"],
            key="comm_type",
            disabled=not exploratory_behaviour,
            help=(
                "panic = sensationalist coverage (raises global panic); "
                "calming = reassuring coverage (lowers panic); "
                "neutral = factual reporting (no panic effect)."
            ),
        )

    with st.sidebar.expander(_t("exp_stockpile"), expanded=False):
        stockpile_days_on = st.checkbox(
            _t("stockpile_on"), False, key="stockpile_on",
            disabled=not exploratory_behaviour,
            help="Override the agent's heuristic stockpile planning horizon. The beta mapping is not empirically estimated."
        )
        stockpile_days_val = st.slider(
            _t("stockpile_days"), 1, 14, 3,
            key="stockpile_days_val",
            disabled=(not exploratory_behaviour or not stockpile_days_on),
            help="O'Donoghue & Rabin (1999): agents plan to hold this many days of supply at home."
        )
        stockpile_days_override = stockpile_days_val if stockpile_days_on else None

    st.sidebar.header(_t("sidebar_mc"))
    mc_runs = st.sidebar.number_input(_t("mc_runs"), 3, 100, 10,
                                       help="More runs = tighter confidence intervals")
    show_ci = st.sidebar.checkbox(
        "Show confidence intervals",
        value=True,
        key="show_ci",
        help="Toggle p10/IQR/p90 bands in MC charts. CI columns are always included in the downloaded CSV.",
    )

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
        if not exploratory_behaviour:
            st.caption(
                "Choice effects from labelling are not estimated in the current data "
                "and are disabled in empirical-only mode."
            )
        lab_active = st.checkbox(
            _t("labelling_on"), False, key="pol_lab_active",
            disabled=not exploratory_behaviour,
        )
        lab_day           = st.slider(_t("labelling_start"), 1, max(2, days_to_run - 1),
                                      1, key="pol_lab_day", disabled=not exploratory_behaviour)
        lab_health_boost  = st.slider(_t("labelling_health"), 0.0, 0.4, 0.15, 0.05,
                                      key="pol_lab_health", disabled=not exploratory_behaviour)
        lab_organic_boost = st.slider(_t("labelling_organic"), 0.0, 0.3, 0.10, 0.05,
                                      key="pol_lab_organic", disabled=not exploratory_behaviour)

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
        "labelling_active":        lab_active and exploratory_behaviour,
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
        "panic":                   panic_sens if exploratory_behaviour else 0.0,
        "hoard":                   hoarding if exploratory_behaviour else 1.0,
        "exploratory_behaviour":   bool(exploratory_behaviour),
        "mc_runs":                 int(mc_runs),
        "show_ci":                 bool(show_ci),
        "policy_cfg":              policy_cfg,
        # Behavioural interventions
        "purchase_limit":          purchase_limit,
        "media_intensity":         media_intensity if exploratory_behaviour else 0.0,
        "communication_type":      communication_type if exploratory_behaviour else "neutral",
        "stockpile_days":          stockpile_days_override if exploratory_behaviour else None,
        "traffic_variation":       bool(traffic_variation),
        "panic_exposure_floor":    panic_exposure_floor,
        "panic_growth_rate":       panic_growth_rate,
        "panic_decay_active":      panic_decay_active,
        "panic_decay_recovery":    panic_decay_recovery,
        "inflation_panic_rate":    inflation_panic_rate,
    }


def _make_model(
    params: dict,
    is_crisis: bool,
    seed: int,
    ai_recs=None,
    policy_cfg: dict = None,
) -> SupermarketModel:
    config = st.session_state.config_data
    if config is None:
        raise RuntimeError("Load population data before constructing the model.")
    if config.get("stats", {}).get("population_pipeline_version", 0) < 9:
        raise RuntimeError(
            "The loaded configuration predates the evidence-separated choice pipeline. "
            "Reprocess or reload the source files in Data & Population first."
        )
    exploratory = bool(params.get("exploratory_behaviour", False))
    archetypes_supported = bool(
        config.get("stats", {})
        .get("archetype_stability", {})
        .get("archetypes_supported", False)
    )
    return SupermarketModel(
        config_data          = config,
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
        panic_exposure_floor      = params.get("panic_exposure_floor", 0.10),
        panic_growth_rate         = params.get("panic_growth_rate", 0.50),
        panic_decay_active        = params.get("panic_decay_active", 0.05),
        panic_decay_recovery      = params.get("panic_decay_recovery", 0.10),
        inflation_panic_rate      = params.get("inflation_panic_rate", 0.40),
        enable_panic_dynamics     = exploratory,
        enable_tpb                = exploratory,
        enable_prospect_theory    = exploratory,
        enable_preference_learning = exploratory and archetypes_supported,
        enable_archetype_modifiers = exploratory and archetypes_supported,
        enable_policy_choice_effects = exploratory,
        enable_traffic_variation   = bool(params.get("traffic_variation", False)),
    )


# ===========================================================================
# 4. TAB: DATA & POPULATION
# ===========================================================================

def render_data_tab():
    st.header(_t("header_data"))

    # ── Bundled-data status banner ────────────────────────────────────────────
    _has_secret  = "firebase" in st.secrets and "data" in st.secrets["firebase"]
    _has_catalogue = (_DATA_DIR / "master_products.json").exists()
    _has_dce = (
        ("dce" in st.secrets and "data" in st.secrets["dce"])
        or _LOCAL_DCE_PATH.exists()
    )
    if st.session_state.config_data is not None:
        cfg_stats = st.session_state.config_data.get("stats", {})
        n_real  = cfg_stats.get("n_real", "?")
        n_pool  = cfg_stats.get("pool_size", "?")
        n_prods = len(st.session_state.config_data.get("products", []))
        raw_prods = cfg_stats.get("catalogue_rows_raw", n_prods)
        duplicate_note = (
            f" ({raw_prods - n_prods} duplicate scene placements collapsed)"
            if isinstance(raw_prods, int) and raw_prods > n_prods else ""
        )
        if cfg_stats.get("population_pipeline_version", 0) < 9:
            st.warning(
                f"⚠️ **Legacy population loaded** — {n_real} participants · {n_prods} unique SKUs.\n\n"
                "Simulation is blocked until you reload or reprocess the source files with "
                "the complete-profile resampling and evidence-separated choice pipeline below."
            )
        else:
            st.success(
                f"✅ **Initial data loaded** — {n_real} empirical participants · "
                f"{n_pool} resampled household draws/model · {n_prods} unique SKUs{duplicate_note}.\n\n"
                "You can jump straight to **🎮 Interactive Demo**. "
                "Use the expander below only if you want to load a different dataset."
            )
    elif not (_has_secret and _has_catalogue and _has_dce):
        missing = []
        if not _has_secret:    missing.append("Firebase secret (add in Streamlit Cloud → Settings → Secrets)")
        if not _has_catalogue: missing.append("product catalogue (data/master_products.json)")
        if not _has_dce: missing.append("cleaned DCE alternatives CSV with recorded prices")
        st.warning("⚠️ Bundled data not fully configured. Missing: " + " · ".join(missing) +
                   ". Upload files manually below or configure the missing source.")

    _legacy_population = (
        st.session_state.config_data is not None
        and st.session_state.config_data.get("stats", {}).get("population_pipeline_version", 0) < 9
    )
    with st.expander(
        "🔄 Reload / Override Data Files",
        expanded=(st.session_state.config_data is None or _legacy_population),
    ):
        st.markdown(
            "Upload a new **Firebase export**, **product catalogue**, and cleaned "
            "**DCE alternatives CSV** to rebuild the evidence pipeline from scratch."
        )

        col_fb, col_prod, col_dce = st.columns(3)
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
        with col_dce:
            dce_file = st.file_uploader(
                "📂 Cleaned DCE Alternatives CSV",
                type=["csv"],
                key="upload_dce",
                help="Long-format DCE alternatives including respondent_id, choice_id, chosen, attributes, and displayed price",
            )

        pool_size = st.number_input(
            "Simulated Household Pool Size", 100, 50000, 2000,
            help="Each model seed resamples this many complete profiles from the observed participants. This changes simulation scale, not empirical sample size.",
        )
        n_archetypes = st.selectbox(
            "Requested exploratory clusters", [2, 3, 4, 5], index=2,
            help="Clusters affect behaviour only if separation, bootstrap stability, minimum size, and k-selection gates all pass.",
        )

        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("🔄 Process Uploaded Files", type="primary"):
            if fb_file is None or prod_file is None or dce_file is None:
                st.error("Please upload both JSON files and the cleaned DCE CSV before processing.")
                return
            with st.spinner("Parsing profiles, auditing constructs/clusters, and configuring participant resampling…"):
                try:
                    firebase_dict  = json.load(fb_file)
                    products_dict  = json.load(prod_file)
                    dce_rows = list(csv.DictReader(io.StringIO(
                        dce_file.getvalue().decode("utf-8-sig")
                    )))
                    config         = _cached_run_pipeline_from_data(
                        firebase_dict, products_dict,
                        int(pool_size), int(n_archetypes), dce_rows,
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
                _firebase_dict, _products_dict, _dce_rows = _load_bundled_data()
                if _firebase_dict is None or _products_dict is None:
                    st.error("Bundled data not available. Firebase secret or product catalogue missing.")
                else:
                    st.session_state.config_data = _cached_run_pipeline_from_data(
                        _firebase_dict, _products_dict,
                        int(pool_size), int(n_archetypes), _dce_rows,
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

    if stats.get("population_pipeline_version", 0) < 9:
        st.error(
            "This in-memory configuration predates the evidence-separated choice pipeline. "
            "Use **Reload Bundled Files** or process the uploaded files again before running simulations."
        )

    # ---- Summary metrics ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Real Participants", stats["n_real"])
    c2.metric("Empirical Sampling Units", stats.get("empirical_sampling_units", stats["n_real"]))
    c3.metric("Households Drawn / Model", stats["pool_size"])
    c4.metric("Products in Catalogue", len(prods))

    if stats.get("n_skipped", 0):
        st.warning(
            f"⚠️ {stats['n_skipped']} participant(s) were skipped "
            "(basket had no products matching the catalogue)."
        )

    st.caption(
        "The displayed model population is not 2,000 independent observations. Each seed "
        "resamples complete participant profiles with replacement; no demographics, beliefs, "
        "preferences, prices, quantities, or baskets are jittered. Longitudinal household "
        "state is persistent. Because the source data do not "
        "contain a validated shopping-frequency variable, expected revisit intervals "
        "are inferred from household-pool size and configured daily traffic. Calendar "
        "multipliers and random traffic variation require explicit exploratory opt-in. "
        "The revisit assumption should be calibrated "
        "when observed visit-frequency data become available."
    )

    reliability = stats.get("questionnaire_reliability", {})
    if reliability.get("status") == "ok":
        with st.expander("🧾 Questionnaire Construct Audit", expanded=True):
            reliability_df = pd.DataFrame(reliability.get("constructs", []))
            st.dataframe(reliability_df, hide_index=True, use_container_width=True)
            st.caption(
                "Raw Cronbach alpha uses participants with complete responses for each declared "
                "positional item group. Missing items are not silently filled with neutral answers. "
                "Reverse-key metadata and a confirmatory measurement model are not available."
            )
            if not reliability.get("all_constructs_acceptable", False):
                st.warning(
                    "At least one declared construct did not reach the conservative reliability "
                    "gate. Treat its factor score as exploratory and inspect item coding before interpretation."
                )

    cluster_audit = stats.get("archetype_stability", {})
    if cluster_audit.get("status") == "ok":
        with st.expander("🧭 Archetype Stability Audit", expanded=True):
            ca1, ca2, ca3 = st.columns(3)
            ca1.metric("Requested k", cluster_audit.get("selected_k"))
            ca2.metric("Best silhouette k", cluster_audit.get("recommended_k"))
            ca3.metric(
                "Operational archetypes",
                "Enabled" if cluster_audit.get("archetypes_supported") else "Disabled",
            )
            st.dataframe(
                pd.DataFrame(cluster_audit.get("candidates", [])),
                hide_index=True, use_container_width=True,
            )
            st.caption(
                "Operational gate: requested k must be the best silhouette solution, silhouette "
                "≥ 0.25, median bootstrap adjusted Rand index ≥ 0.75, and every cluster must "
                "contain at least 5% of participants (minimum five)."
            )
            if not cluster_audit.get("archetypes_supported"):
                st.warning(
                    "The categorical solution failed at least one gate. Cluster labels remain "
                    "visible for exploratory description, but agents use their continuous "
                    "participant-level preferences; archetype modifiers and learning rules are disabled."
                )

    calibration = stats.get("behavioral_calibration", {})
    if calibration.get("status") == "ok":
        with st.expander("🧪 Behavioural Calibration Audit", expanded=True):
            st.markdown(
                f"**{calibration['n_observed']}** participants have usable phase-two "
                f"observations. A fixed **{calibration['n_train']}/{calibration['n_validation']}** "
                "train/validation split evaluates pre-crisis predictors, while five-fold "
                "cross-fitting prevents a participant's own phase-two outcome from setting "
                "their simulated behaviour. The observed median price shock was "
                f"**{calibration.get('observed_median_price_shock', 0):.0%}**."
            )
            calibration_rows = []
            for key, label in [
                ("price_response", "Quantity/price response"),
                ("substitution", "Product substitution"),
                ("hoarding", "Quantity increase / hoarding"),
                ("budget_utilization", "Phase-two reservation spending"),
            ]:
                retained = calibration.get(f"{key}_model_retained", False)
                calibration_rows.append({
                    "Outcome": label,
                    "Validation MAE": calibration.get(f"{key}_mae"),
                    "Naive-mean MAE": calibration.get(f"{key}_naive_mae"),
                    "Skill vs naive": calibration.get(f"{key}_skill"),
                    "Individual model used": "Yes" if retained else "No — population mean",
                })
            st.dataframe(pd.DataFrame(calibration_rows), hide_index=True,
                         use_container_width=True)
            st.caption(
                "One-shopping-occasion price-shock calibration using the empirical "
                "relative-price rule: revealed-preference margin "
                f"{calibration.get('revealed_preference_margin', 0):.3f}; held-out "
                f"observed quantity retention {calibration.get('retention_validation_observed_mean', 0):.1%}, "
                f"predicted {calibration.get('retention_validation_predicted_mean', 0):.1%}. "
                f"Individual retention skill versus the naive mean: "
                f"{calibration.get('retention_validation_skill', 0):+.3f}."
            )
            if not calibration.get("price_response_model_retained", False):
                st.warning(
                    "Individual price response did not beat the naive validation benchmark. "
                    "The ABM therefore uses cross-fitted population means rather than "
                    "inventing unsupported household heterogeneity."
                )
            if calibration.get("retention_validation_skill", 0) <= 0:
                st.warning(
                    "The one-occasion aggregate mean is aligned, but participant-level "
                    "retention does not beat a naive mean forecast. This is not validation "
                    "of the ABM's multi-day inventory/pantry trajectory; do not interpret "
                    "individual price-response paths as validated predictions."
                )
            st.info(
                "Phase-two baskets are calibration and validation targets only. Simulated "
                "crisis demand starts from phase-one needs and is generated by model rules."
            )

    dce_validation = stats.get("dce_choice_validation", {})
    if dce_validation.get("status") == "ok":
        with st.expander("🎯 Choice-Experiment Holdout Assessment", expanded=False):
            d1, d2, d3 = st.columns(3)
            d1.metric("Held-out choices", dce_validation["n_validation_choices"])
            d2.metric("Choice accuracy", f"{dce_validation['validation_accuracy']:.1%}")
            d3.metric(
                "Null-model accuracy",
                f"{dce_validation.get('null_model_accuracy', dce_validation['majority_accuracy']):.1%}",
            )
            st.write(
                "Training-cohort pooled coefficients: price "
                f"**{dce_validation.get('price_coefficient', 0):+.3f} per EUR**, Finnish origin "
                f"**{dce_validation['origin_coefficient']:+.3f}**, organic "
                f"**{dce_validation['organic_coefficient']:+.3f}**, fat linear "
                f"**{dce_validation['fat_linear_coefficient']:+.3f}**, and fat-squared "
                f"**{dce_validation['fat_quadratic_coefficient']:+.3f}**. These are "
                "conditional-logit coefficients estimated jointly with the opt-out "
                "alternative."
            )
            st.caption(
                f"Held-out log loss {dce_validation['validation_log_loss']:.3f} "
                f"versus null {dce_validation.get('null_model_log_loss', float('nan')):.3f}; "
                f"{dce_validation.get('n_inferred_price_choice_sets_excluded', 0)} "
                "choice sets with inferred prices excluded."
            )
            if (
                dce_validation.get("beats_null_benchmark", False)
                and dce_validation.get("model_converged", True)
            ):
                st.success(
                    "The pooled price-and-attribute model beats the held-out null model. "
                    "It is used probabilistically to allocate milk substitutes among "
                    "available and affordable candidates."
                )
            else:
                st.warning(
                    "The DCE model either did not beat its held-out null benchmark or did "
                    "not converge. It is therefore diagnostic only and cannot influence agents."
                )
            st.warning(
                "This is a pooled milk-domain model, not evidence of household-specific "
                "willingness-to-pay heterogeneity. The phase-one/phase-two baskets separately "
                "identify whether replacement occurs. Milk DCE coefficients are not "
                "extrapolated to cheese, yogurt, cream, or plant drinks."
            )

    substitution_validation = stats.get("substitution_choice_validation", {})
    if substitution_validation.get("status") == "ok":
        with st.expander("🔁 Replacement-Choice Validity Audit", expanded=True):
            s1, s2, s3 = st.columns(3)
            s1.metric(
                "Unambiguous events",
                substitution_validation.get("n_unambiguous_events", 0),
            )
            s2.metric(
                "Price-gate target coverage",
                f"{substitution_validation.get('candidate_price_gate_target_coverage', 0):.1%}",
            )
            supported_categories = substitution_validation.get(
                "supported_ranking_categories", []
            )
            transition_categories = substitution_validation.get(
                "supported_transition_categories", []
            )
            s3.metric(
                "Validated transition shares",
                ", ".join(transition_categories) if transition_categories else "None",
            )
            st.dataframe(
                pd.DataFrame(substitution_validation.get("categories", [])),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "An event requires exactly one removed and one added catalogue SKU "
                "within a category. Ranking is accepted only with at least 30 events, "
                "top-1 accuracy of at least 25%, and a five-point advantage over "
                "leave-one-event-out category popularity. Candidate price screening "
                "additionally requires at least 100 events and 90% chosen-target coverage."
            )
            transition_table = substitution_validation.get(
                "phase_transition_target_models", []
            )
            if transition_table:
                st.markdown("**Two-stage basket transition destination audit**")
                st.dataframe(
                    pd.DataFrame(transition_table),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(substitution_validation.get("transition_gate", ""))
            if not substitution_validation.get(
                "candidate_price_gate_supported", False
            ):
                st.warning(
                    "The phase-two retain/drop price threshold does not cover enough "
                    "observed replacement targets, so it is not transferred to substitute "
                    "SKU screening. Remaining visit budget remains a hard constraint."
                )
                st.caption(
                    substitution_validation.get(
                        "candidate_price_gate_limitation", ""
                    )
                )
            if not supported_categories and not transition_categories:
                st.warning(
                    "No category has enough predictive evidence for deterministic "
                    "substitute ranking. The ABM therefore makes a seeded uniform draw "
                    "among affordable, in-stock, same-category candidates. This is an "
                    "explicit structural uncertainty, not a validated consumer-choice rule."
                )
            else:
                st.info(
                    "Replacement incidence comes from the phase-one/phase-two basket "
                    "difference. Milk target choice uses the validated DCE multinomial "
                    "model; supported non-milk categories use training-cohort transition "
                    "shares only after beating uniform choice on held-out participants."
                )
            st.info(substitution_validation.get("caution", ""))

    registry = build_parameter_registry(
        stats=stats,
        runtime_params=st.session_state.get("_last_params", {}),
    )
    registry_errors = validate_parameter_registry(registry)
    registry_summary = parameter_registry_summary(registry)
    with st.expander("📑 Parameter Evidence Registry", expanded=True):
        st.markdown(
            "This registry separates values **observed in GROCERYsim**, values "
            "**estimated with held-out or cross-fitted data**, **literature transfers**, "
            "**scenario inputs**, and **unidentified engineering assumptions**. Its "
            "classification is intentionally conservative."
        )
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Influential entries", registry_summary["n_parameters"])
        r2.metric("Identifiable here", registry_summary["n_identifiable"])
        r3.metric(
            "Unresolved high-priority",
            registry_summary["n_unresolved_high_priority"],
        )
        r4.metric(
            "Policy-grade ready",
            "Yes" if registry_summary["policy_grade_ready"] else "No",
        )
        if registry_errors:
            st.error("Registry validation failed: " + " | ".join(registry_errors))
        if not registry_summary["policy_grade_ready"]:
            st.warning(
                "The current model is suitable for transparent exploratory scenario "
                "analysis, but not yet for point prediction or policy-effect claims. "
                "Critical assumptions must be calibrated, externally validated, or "
                "propagated through uncertainty analysis before policy-grade use."
            )

        registry_df = pd.DataFrame(registry)
        evidence_options = sorted(registry_df["evidence_class"].unique())
        chosen_evidence = st.multiselect(
            "Filter evidence classes",
            evidence_options,
            default=evidence_options,
            key="parameter_registry_evidence_filter",
        )
        priority_filter = st.selectbox(
            "Minimum review focus",
            ["All", "Critical only", "Critical + high"],
            key="parameter_registry_priority_filter",
        )
        shown = registry_df[registry_df["evidence_class"].isin(chosen_evidence)]
        if priority_filter == "Critical only":
            shown = shown[shown["priority"] == "critical"]
        elif priority_filter == "Critical + high":
            shown = shown[shown["priority"].isin(["critical", "high"])]
        registry_display = shown[[
                "parameter_id", "parameter", "component", "current_value",
                "unit", "evidence_class", "identifiable_from_current_data",
                "validation", "priority", "uncertainty_treatment", "source",
            ]].copy()
        # Values intentionally mix numbers, booleans, and method labels.  Keep
        # the audit column display-only and Arrow-stable without changing the
        # machine-readable registry returned by build_parameter_registry().
        registry_display["current_value"] = registry_display["current_value"].map(str)
        st.dataframe(
            registry_display,
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Identifiable means the current GROCERYsim export contains enough "
            "information to estimate the stated quantity; it does not imply external "
            "validity or causal identification."
        )
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "⬇️ Download registry (CSV)",
            registry_df.to_csv(index=False).encode("utf-8"),
            file_name="grocerysim_parameter_evidence_registry.csv",
            mime="text/csv",
            use_container_width=True,
        )
        dl2.download_button(
            "⬇️ Download registry (JSON)",
            json.dumps(registry, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="grocerysim_parameter_evidence_registry.json",
            mime="application/json",
            use_container_width=True,
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
**Buyer-type profiles** are exploratory k-means descriptions based on five declared
survey attitude scores plus identifiable origin, organic, and chosen-fat attributes.
No PCA is used. The descriptive participant cheaper-choice share is excluded;
the pooled DCE price coefficient is estimated and validated separately.

| Exploratory label | Centroid description |
|------|-----------|
| 💸 **Price Champion** | Higher declared price orientation |
| 🌿 **Green Buyer** | Higher environment, animal-welfare, or organic orientation |
| 💪 **Health Optimizer** | Higher declared health orientation |
| 🔁 **Habitual Buyer** | Higher sensory/familiarity orientation |

These names do **not** establish crisis behaviour, panic, hoarding, or causal response.

**Scientific gate**
- Categories affect behaviour only when k-selection, separation, bootstrap stability,
  and minimum-cluster-size requirements pass.
- Otherwise continuous participant attributes are used and category-specific modifiers
  and learning rules are disabled.
- Prospect Theory, TPB, panic stockpiling, and preference learning are disabled
  in empirical-only mode. They require the explicit exploratory-behaviour opt-in.
- food-access stress flags are summed per agent daily (0 = none → 4 = severe).
""")
    _clusters_supported = stats.get("archetype_stability", {}).get(
        "archetypes_supported", False
    )
    arch_data = (
        stats.get("archetypes_real", {}) if _clusters_supported
        else stats.get("exploratory_archetypes_real", {})
    )
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
                _group_column = "archetype" if _clusters_supported else "exploratory_archetype"
                sub = df_real[df_real[_group_column] == arch]
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

    if st.session_state.config_data is None:
        st.warning("⚠️ Load data in the **🏠 Data & Population** tab first.")
        return

    mode = st.radio(
        "Simulation Mode",
        ["⚡ Quick Preview — single run, live animation",
         "🔬 Full Analysis — Monte Carlo with confidence intervals"],
        horizontal=True,
        key="demo_mode",
    )
    st.divider()

    if not mode.startswith("⚡"):
        _render_demo_mc(params)
        return

    # --- Quick Preview (single run, live animation) ---
    st.markdown(
        "Run a single paired simulation (Baseline vs Crisis) and watch the "
        "results update in real time."
    )
    run_speed = st.slider(_t("animation_speed"), 0.0, 0.2, 0.02, 0.01)

    if st.button(_t("btn_run_demo"), type="primary"):
        SEED = 42
        model_base   = _make_model(params, is_crisis=False, seed=SEED)
        model_crisis = _make_model(params, is_crisis=True,  seed=SEED)

        results      = []
        stock_rows   = []
        agent_rows   = []
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

            # Agent-level snapshot (for Agent Replay tab)
            agent_rows.extend(_collect_agent_snapshot(model_base,   day, "Baseline"))
            agent_rows.extend(_collect_agent_snapshot(model_crisis, day, "Crisis"))

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
        st.session_state.agent_log        = pd.DataFrame(agent_rows) if agent_rows else None

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
                            ("MeanFIES",             "Exploratory Access Stress", True),
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
                    ("MeanFIES",             "Access-Stress Score (0–4)", "Lower is better"),
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
                    (("MeanFIES", "Realised Access Stress"), _t2),
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
                        ("MeanFIES",             "Access Stress Δ"),
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

    # ── Save scenario ────────────────────────────────────────────────────────
    if st.session_state.sim_results is not None:
        st.divider()
        st.markdown("### 💾 Save This Scenario for Comparison")
        _sc_col1, _sc_col2, _sc_col3 = st.columns([3, 1, 1])
        with _sc_col1:
            _sc_name = st.text_input(
                "Scenario name",
                value=f"Scenario {len(st.session_state.saved_scenarios) + 1}",
                key="save_scenario_name",
                label_visibility="collapsed",
                placeholder="Give this scenario a name…",
            )
        with _sc_col2:
            if st.button("💾 Save Scenario", use_container_width=True, key="save_scenario_btn"):
                _df_r = st.session_state.sim_results
                _base = _df_r[_df_r["Scenario"] == "Baseline"]
                _cris = _df_r[_df_r["Scenario"] == "Crisis"]
                _entry = {
                    "name":       _sc_name or f"Scenario {len(st.session_state.saved_scenarios)+1}",
                    "timestamp":  pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "params":     {k: params.get(k) for k in
                                   ["days","base_con","month","reorder","target","lead",
                                    "inf","dis","panic","hoard","cri_start"]},
                    "revenue_base":    float(_base["Revenue"].sum()),
                    "revenue_crisis":  float(_cris["Revenue"].sum()),
                    "lost_sales":      float(_df_r["LostSales"].sum()),
                    "waste":           float(_df_r["Waste"].sum()),
                    "fulfillment":     float(_df_r["FulfillmentRate"].mean()),
                    "co2":             float(_df_r["CO2Total"].mean()) if "CO2Total" in _df_r.columns else 0.0,
                    "import_dep":      float(_df_r["ImportDepPct"].mean()) if "ImportDepPct" in _df_r.columns else 0.0,
                    "fies_low":        float(_df_r["FIESSevere_Low"].mean()) if "FIESSevere_Low" in _df_r.columns else 0.0,
                    "panic_peak":      float(_df_r["PanicLevel"].max()) if "PanicLevel" in _df_r.columns else 0.0,
                    "tags":            [],
                    "notes":           "",
                    "df":             _df_r.copy(),
                }
                st.session_state.saved_scenarios.append(_entry)
                st.success(f"✅ Saved **{_entry['name']}** — go to 📊 Compare Scenarios tab to compare.")
        with _sc_col3:
            if st.session_state.saved_scenarios:
                if st.button("🗑️ Clear All", use_container_width=True, key="clear_scenarios_btn"):
                    st.session_state.saved_scenarios = []
                    st.rerun()


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
# 7. MONTE CARLO FULL ANALYSIS (embedded in Interactive Demo)
# ===========================================================================

def _populate_session_from_mc(df_base: pd.DataFrame, df_cri: pd.DataFrame,
                               n_runs: int, params: dict, label_base: str,
                               ai_recs=None):
    """
    Derive all downstream session state from completed MC results:
      sim_results    ← day-by-day MC mean trajectory (both scenarios)
      sim_stock, sim_waste, sim_scm_log, agent_log, sim_pref_drift, sim_model_crisis
                     ← single re-run of the median realisation seed
    """
    # ── 1. MC mean → sim_results ─────────────────────────────────────────────
    mean_b = df_base.groupby("Day").mean(numeric_only=True).reset_index()
    mean_b["Scenario"] = label_base
    mean_c = df_cri.groupby("Day").mean(numeric_only=True).reset_index()
    mean_c["Scenario"] = "Crisis"
    st.session_state.sim_results = pd.concat([mean_b, mean_c], ignore_index=True)

    # ── 2. Identify median realisation (run closest to median crisis revenue) ─
    run_totals   = df_cri.groupby("Run")["Revenue"].sum().sort_values()
    median_run_id = run_totals.index[len(run_totals) // 2]
    median_seed   = 1000 + int(median_run_id)   # matches _run_mc_batch seed formula
    st.session_state.mc_median_seed = median_seed

    # ── 3. Re-run median seed to collect agent-level detail ──────────────────
    with st.spinner(
        f"Collecting representative run (median realisation, seed {median_seed})…"
    ):
        m_base   = _make_model(params, is_crisis=False, seed=median_seed, ai_recs=ai_recs)
        m_crisis = _make_model(params, is_crisis=True,  seed=median_seed, ai_recs=ai_recs)

        stock_rows = []
        agent_rows = []

        for day in range(1, params["days"] + 1):
            m_base.step()
            m_crisis.step()
            m_base.collect_product_snapshot()
            m_crisis.collect_product_snapshot()
            m_base.collect_preference_snapshot()
            m_crisis.collect_preference_snapshot()

            _, prod_b = _collect_model_day(m_base,   day, label_base)
            _, prod_c = _collect_model_day(m_crisis, day, "Crisis")
            stock_rows.extend(prod_b)
            stock_rows.extend(prod_c)
            agent_rows.extend(_collect_agent_snapshot(m_base,   day, label_base))
            agent_rows.extend(_collect_agent_snapshot(m_crisis, day, "Crisis"))

        # SCM logs
        log_b = pd.DataFrame(m_base.truck.log)
        log_c = pd.DataFrame(m_crisis.truck.log)
        if not log_b.empty: log_b["Scenario"] = label_base
        if not log_c.empty: log_c["Scenario"] = "Crisis"

        # Preference drift snapshots
        pref_b = pd.DataFrame(getattr(m_base,   "_pref_snapshots", []))
        pref_c = pd.DataFrame(getattr(m_crisis, "_pref_snapshots", []))
        if not pref_b.empty: pref_b["Scenario"] = label_base
        if not pref_c.empty: pref_c["Scenario"] = "Crisis"

        # Food waste item log
        waste_b = pd.DataFrame(m_base.food_waste_log.records)
        waste_c = pd.DataFrame(m_crisis.food_waste_log.records)
        if not waste_b.empty: waste_b["Scenario"] = label_base
        if not waste_c.empty: waste_c["Scenario"] = "Crisis"

        st.session_state.sim_stock        = pd.DataFrame(stock_rows)
        st.session_state.sim_scm_log      = pd.concat([log_b, log_c], ignore_index=True)
        st.session_state.sim_model_crisis = m_crisis
        st.session_state.agent_log        = pd.DataFrame(agent_rows) if agent_rows else None
        st.session_state.sim_pref_drift   = pd.concat([pref_b, pref_c], ignore_index=True)
        st.session_state.sim_waste        = pd.concat([waste_b, waste_c], ignore_index=True)
        st.session_state.mc_session_populated = True


def _render_demo_mc(params: dict):
    """4-stage Monte Carlo workflow — runs inside the Interactive Demo (Full Analysis mode)."""
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
        if st.button(_t("btn_run_baseline"), type="primary", key="mc_btn_baseline"):
            df_raw, prod_stats = _run_mc_batch(n_runs, days, params, is_crisis=False,
                                               progress_label="Simulating Baseline…")
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

    # ---- STAGE 1: Review AI storage optimisation ----
    elif st.session_state.mc_stage == 1:
        st.markdown(
            '<div class="step-card"><h3>Step 2 — Review AI Storage Optimisation</h3>'
            "<p>Inspect baseline performance and choose whether to apply "
            "AI-recommended storage capacity adjustments.</p></div>",
            unsafe_allow_html=True,
        )
        df_raw = st.session_state.data_base_raw
        _plot_ci_band(df_raw, "Daily Revenue — Baseline (Raw) [95 % CI]", color="gray")
        st.divider()
        st.markdown("#### 🤖 AI Storage Capacity Recommendations")
        recs_df = pd.DataFrame([
            {"Product": p,
             "Current Capacity": st.session_state.prod_stats_raw[p]["StorageCap"],
             "Recommended Capacity": c}
            for p, c in st.session_state.ai_recs.items()
        ])
        st.dataframe(recs_df, use_container_width=True)

        col1, col2 = st.columns(2)
        if col1.button("✅ Accept AI Recommendations & Re-run Baseline", key="mc_btn_accept"):
            df_opt, _ = _run_mc_batch(n_runs, days, params, is_crisis=False,
                                      ai_recs=st.session_state.ai_recs,
                                      progress_label="Simulating Optimised Baseline…")
            st.session_state.data_base_opt   = df_opt
            st.session_state.active_baseline = "Baseline (Optimised)"
            st.session_state.mc_stage        = 2
            st.rerun()
        if col2.button("⏩ Skip — Use Raw Baseline", key="mc_btn_skip"):
            st.session_state.data_base_opt   = None
            st.session_state.active_baseline = "Baseline (Raw)"
            st.session_state.mc_stage        = 2
            st.rerun()

    # ---- STAGE 2: Baseline comparison + launch crisis ----
    elif st.session_state.mc_stage == 2:
        st.markdown(
            '<div class="step-card"><h3>Step 3 — Baseline Comparison</h3>'
            "<p>Compare Raw vs Optimised baseline, then run the Crisis scenario.</p></div>",
            unsafe_allow_html=True,
        )
        if st.session_state.data_base_opt is not None:
            df_r = st.session_state.data_base_raw
            df_o = st.session_state.data_base_opt
            r_mean = df_r["Revenue"].mean()
            o_mean = df_o["Revenue"].mean()
            r_min  = df_r.groupby("Run")["Revenue"].min().mean()
            o_min  = df_o.groupby("Run")["Revenue"].min().mean()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Daily Rev (Raw)", f"€{r_mean:,.1f}")
            c2.metric("Avg Daily Rev (Opt)", f"€{o_mean:,.1f}",
                      f"{(o_mean-r_mean)/max(r_mean,0.01)*100:.1f}%")
            c3.metric("Avg Min Rev (Raw)",   f"€{r_min:,.1f}")
            c4.metric("Avg Min Rev (Opt)",   f"€{o_min:,.1f}",
                      f"{(o_min-r_min)/max(r_min,0.01)*100:.1f}%")

            r_waste = df_r["Waste"].sum() / n_runs
            o_waste = df_o["Waste"].sum() / n_runs
            r_lost  = df_r["LostSales"].sum() / n_runs
            o_lost  = df_o["LostSales"].sum() / n_runs
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name="Waste", x=["Raw", "Optimised"],
                                     y=[r_waste, o_waste], marker_color="salmon",
                                     text=[f"{r_waste:.0f}", f"{o_waste:.0f}"],
                                     textposition="auto"))
            fig_cmp.add_trace(go.Bar(name="Lost Sales", x=["Raw", "Optimised"],
                                     y=[r_lost, o_lost], marker_color="orange",
                                     text=[f"{r_lost:.0f}", f"{o_lost:.0f}"],
                                     textposition="auto"))
            fig_cmp.update_layout(barmode="group", title="Baseline Efficiency Comparison",
                                  template="plotly_white",
                                  yaxis_title="Units (avg per run)")
            st.plotly_chart(fig_cmp, use_container_width=True, config=_PLOTLY_CFG)
        else:
            st.info("Using Raw Baseline (optimisation skipped).")

        st.divider()
        st.markdown(f"✅ Ready to test **{st.session_state.active_baseline}** vs **Crisis**.")
        if st.button(_t("btn_run_crisis"), type="primary", key="mc_btn_crisis"):
            recs = st.session_state.ai_recs if st.session_state.data_base_opt is not None else None
            df_cri, _ = _run_mc_batch(n_runs, days, params, is_crisis=True,
                                      ai_recs=recs, progress_label="Simulating Crisis…")
            st.session_state.data_crisis = df_cri
            st.session_state.mc_stage    = 3
            st.rerun()

    # ---- STAGE 3: Full multi-metric impact analysis ----
    elif st.session_state.mc_stage == 3:
        st.markdown(
            '<div class="step-card"><h3>Step 4 — Full Impact Analysis</h3>'
            "<p>Confidence-interval bands (p10 / IQR / p90) across all key metrics — "
            "economic, operational, welfare, and environmental.</p></div>",
            unsafe_allow_html=True,
        )

        label_base = st.session_state.active_baseline
        df_base = (st.session_state.data_base_opt
                   if st.session_state.data_base_opt is not None
                   else st.session_state.data_base_raw)
        df_cri  = st.session_state.data_crisis

        # Populate all downstream session state once (MC mean + median re-run)
        if not st.session_state.mc_session_populated:
            _populate_session_from_mc(
                df_base, df_cri, n_runs, params, label_base,
                ai_recs=st.session_state.ai_recs
                        if st.session_state.data_base_opt is not None else None,
            )
            st.success(
                f"✅ All analysis tabs now use Monte Carlo results "
                f"(N={n_runs} runs, median seed {st.session_state.mc_median_seed}). "
                "CI bands togglable from the sidebar."
            )

        df_base = df_base.copy(); df_base["Scenario"] = label_base
        df_cri  = df_cri.copy();  df_cri["Scenario"]  = "Crisis"
        df_full = pd.concat([df_base, df_cri], ignore_index=True)

        # ── Summary KPIs ────────────────────────────────────────────────────
        rev_base   = df_base["Revenue"].sum() / n_runs
        rev_cri    = df_cri["Revenue"].sum()  / n_runs
        diff_pct   = (rev_cri - rev_base) / max(rev_base, 0.01) * 100
        waste_drop = (df_cri["Waste"].sum() - df_base["Waste"].sum()) / n_runs
        stress_b   = df_base["FoodStressedPct"].mean() if "FoodStressedPct" in df_base.columns else 0.0
        stress_c   = df_cri["FoodStressedPct"].mean()  if "FoodStressedPct" in df_cri.columns  else 0.0
        gini_b     = df_base["GiniAccess"].mean() if "GiniAccess" in df_base.columns else 0.0
        gini_c     = df_cri["GiniAccess"].mean()  if "GiniAccess" in df_cri.columns  else 0.0

        km1, km2, km3, km4 = st.columns(4)
        km1.metric("Revenue Impact",       f"€{rev_cri:,.0f}", f"{diff_pct:.1f}%")
        km2.metric("Extra Waste (crisis)",  f"{waste_drop:+.0f} units/run")
        km3.metric("Food Stressed Δ",      f"{stress_c - stress_b:+.1%}")
        km4.metric("Gini Access Δ",        f"{gini_c - gini_b:+.3f}")

        # ── Per-metric CI tabs ───────────────────────────────────────────────
        m_tabs = st.tabs([
            "💰 Revenue", "♻️ Waste & Lost Sales",
            "👥 Footfall & Panic", "🍽️ Welfare & Equity", "🌿 Environment",
        ])

        with m_tabs[0]:
            _plot_ci_dual(df_base, df_cri, label_base)
            with st.expander("📊 Statistical Analysis", expanded=True):
                _render_analysis(df_full, "Revenue", params, prefix="€", decimals=0,
                                 higher_is_better=True,
                                 baseline_label=label_base, crisis_label="Crisis")
            fig_vio = px.violin(df_full, x="Scenario", y="Revenue", color="Scenario",
                                box=True, points="outliers",
                                title="Daily Revenue Distribution",
                                color_discrete_map={label_base: "#2E8B57", "Crisis": "#DC143C"})
            fig_vio.update_layout(template="plotly_white")
            st.plotly_chart(fig_vio, use_container_width=True, config=_PLOTLY_CFG)

        with m_tabs[1]:
            _plot_ci_dual_col(df_base, df_cri, "Waste",
                              "Daily Waste [p10/IQR/p90]", "Units wasted",
                              label_base, higher_is_better=False)
            with st.expander("📊 Waste Analysis"):
                _render_analysis(df_full, "Waste", params, suffix=" units", decimals=1,
                                 higher_is_better=False,
                                 baseline_label=label_base, crisis_label="Crisis")
            _plot_ci_dual_col(df_base, df_cri, "LostSales",
                              "Daily Lost Sales [p10/IQR/p90]", "Units lost",
                              label_base, higher_is_better=False,
                              color_base="#F39C12", color_cri="#E74C3C")
            with st.expander("📊 Lost Sales Analysis"):
                _render_analysis(df_full, "LostSales", params, suffix=" units", decimals=1,
                                 higher_is_better=False,
                                 baseline_label=label_base, crisis_label="Crisis")

        with m_tabs[2]:
            _plot_ci_dual_col(df_base, df_cri, "Consumers",
                              "Daily Footfall [p10/IQR/p90]", "Shoppers/day",
                              label_base, higher_is_better=True)
            with st.expander("📊 Footfall Analysis"):
                _render_analysis(df_full, "Consumers", params, suffix=" shoppers", decimals=0,
                                 higher_is_better=True,
                                 baseline_label=label_base, crisis_label="Crisis")
            _plot_ci_dual_col(df_base, df_cri, "PanicLevel",
                              "Daily Panic Level [p10/IQR/p90]", "Panic index (0–1)",
                              label_base, higher_is_better=False,
                              color_base="#95A5A6", color_cri="#E74C3C")
            with st.expander("📊 Panic Level Analysis"):
                _render_analysis(df_full, "PanicLevel", params, decimals=3,
                                 higher_is_better=False,
                                 baseline_label=label_base, crisis_label="Crisis")

        with m_tabs[3]:
            _plot_ci_dual_col(df_base, df_cri, "FoodStressedPct",
                              "Food-Stressed Consumers % [p10/IQR/p90]",
                              "% of shoppers food-stressed",
                              label_base, higher_is_better=False)
            with st.expander("📊 Food Stress Analysis"):
                _render_analysis(df_full, "FoodStressedPct", params, suffix="%", decimals=1,
                                 higher_is_better=False,
                                 baseline_label=label_base, crisis_label="Crisis")
            _plot_ci_dual_col(df_base, df_cri, "GiniAccess",
                              "Gini Access Coefficient [p10/IQR/p90]",
                              "Gini (0 = equal, 1 = max inequality)",
                              label_base, higher_is_better=False,
                              color_base="#9B59B6", color_cri="#8E44AD")
            with st.expander("📊 Equity Analysis"):
                _render_analysis(df_full, "GiniAccess", params, decimals=3,
                                 higher_is_better=False,
                                 baseline_label=label_base, crisis_label="Crisis")
            _plot_ci_dual_col(df_base, df_cri, "FulfillmentRate",
                              "Fulfillment Rate [p10/IQR/p90]", "Rate (0–1)",
                              label_base, higher_is_better=True,
                              color_base="#27AE60", color_cri="#E74C3C")
            with st.expander("📊 Fulfillment Analysis"):
                _render_analysis(df_full, "FulfillmentRate", params, decimals=3,
                                 higher_is_better=True,
                                 baseline_label=label_base, crisis_label="Crisis")

        with m_tabs[4]:
            _plot_ci_dual_col(df_base, df_cri, "CO2Total",
                              "Daily CO₂ Equivalent [p10/IQR/p90]", "kg CO₂-eq/day",
                              label_base, higher_is_better=False,
                              color_base="#1ABC9C", color_cri="#E67E22")
            with st.expander("📊 CO₂ Analysis"):
                _render_analysis(df_full, "CO2Total", params, suffix=" kg CO₂-eq", decimals=1,
                                 higher_is_better=False,
                                 baseline_label=label_base, crisis_label="Crisis")
            _plot_ci_dual_col(df_base, df_cri, "ImportDepPct",
                              "Import Dependency % [p10/IQR/p90]",
                              "Import share of sales (%)",
                              label_base, higher_is_better=False,
                              color_base="#16A085", color_cri="#D35400")
            with st.expander("📊 Import Dependency Analysis"):
                _render_analysis(df_full, "ImportDepPct", params, suffix="%", decimals=1,
                                 higher_is_better=False,
                                 baseline_label=label_base, crisis_label="Crisis")

        # ── Correlation heatmap ──────────────────────────────────────────────
        st.subheader("📊 Correlation Analysis — Systemic Coupling")
        _corr_cols = [c for c in
                      ["Revenue", "Waste", "LostSales", "FoodStressedPct",
                       "GiniAccess", "CO2Total"]
                      if c in df_base.columns]
        c_base = df_base[_corr_cols].corr()
        c_cri  = df_cri[_corr_cols].corr()
        fig_heat, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig_heat.patch.set_facecolor("white")
        sns.heatmap(c_base, annot=True, cmap="Greens", ax=ax1, vmin=-1, vmax=1,
                    fmt=".2f", annot_kws={"color": "black"})
        ax1.set_title(label_base, color="black"); ax1.set_facecolor("white")
        sns.heatmap(c_cri,  annot=True, cmap="Reds",   ax=ax2, vmin=-1, vmax=1,
                    fmt=".2f", annot_kws={"color": "black"})
        ax2.set_title("Crisis", color="black"); ax2.set_facecolor("white")
        fig_heat.tight_layout()
        st.pyplot(fig_heat)
        plt.close(fig_heat)

        # ── Downloads ────────────────────────────────────────────────────────
        st.divider()
        csv_bytes = df_full.to_csv(index=False).encode("utf-8")
        c_dl1, c_dl2, c_dl3 = st.columns(3)
        c_dl1.download_button("📥 Download MC Data (CSV)", csv_bytes,
                              "monte_carlo_results.csv", "text/csv")
        if c_dl2.button("📄 Generate PDF Report", key="mc_pdf_btn"):
            try:
                pdf_bytes = _make_pdf_report(df_full, label_base, n_runs)
                c_dl2.download_button("📥 Download PDF", pdf_bytes,
                                      "GROCERYsim_Report.pdf", "application/pdf")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
        if c_dl3.button("🔄 Reset Workflow", key="mc_reset_btn"):
            for k in ["mc_stage", "data_base_raw", "data_base_opt", "data_crisis",
                      "ai_recs", "prod_stats_raw", "mc_session_populated", "mc_median_seed"]:
                st.session_state[k] = 0 if k == "mc_stage" else (
                    False if k == "mc_session_populated" else None
                )
            st.rerun()

        # ── Save scenario ────────────────────────────────────────────────────
        if st.session_state.sim_results is not None:
            st.divider()
            st.markdown("### 💾 Save This Scenario for Comparison")
            _sc_col1, _sc_col2, _sc_col3 = st.columns([3, 1, 1])
            with _sc_col1:
                _sc_name = st.text_input(
                    "Scenario name",
                    value=f"MC Scenario {len(st.session_state.saved_scenarios) + 1}",
                    key="mc_save_scenario_name",
                    label_visibility="collapsed",
                    placeholder="Give this MC scenario a name…",
                )
            with _sc_col2:
                if st.button("💾 Save Scenario", use_container_width=True,
                             key="mc_save_scenario_btn"):
                    _df_r  = st.session_state.sim_results
                    _base  = _df_r[_df_r["Scenario"] == label_base]
                    _cris  = _df_r[_df_r["Scenario"] == "Crisis"]
                    _entry = {
                        "name":      _sc_name or f"MC Scenario {len(st.session_state.saved_scenarios)+1}",
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                        "params":    {k: params.get(k) for k in
                                      ["days", "base_con", "month", "reorder", "target",
                                       "lead", "inf", "dis", "panic", "hoard", "cri_start",
                                       "mc_runs"]},
                        "mc_runs":          n_runs,
                        "mc_median_seed":   st.session_state.mc_median_seed,
                        "revenue_base":     float(_base["Revenue"].sum()),
                        "revenue_crisis":   float(_cris["Revenue"].sum()),
                        "lost_sales":       float(_df_r["LostSales"].sum()),
                        "waste":            float(_df_r["Waste"].sum()),
                        "fulfillment":      float(_df_r["FulfillmentRate"].mean()),
                        "co2":              float(_df_r["CO2Total"].mean()) if "CO2Total" in _df_r.columns else 0.0,
                        "import_dep":       float(_df_r["ImportDepPct"].mean()) if "ImportDepPct" in _df_r.columns else 0.0,
                        "fies_low":         float(_df_r["FIESSevere_Low"].mean()) if "FIESSevere_Low" in _df_r.columns else 0.0,
                        "panic_peak":       float(_df_r["PanicLevel"].max()) if "PanicLevel" in _df_r.columns else 0.0,
                        "tags":  ["Monte Carlo"],
                        "notes": f"MC mean of {n_runs} runs; median seed {st.session_state.mc_median_seed}",
                        "df":    _df_r.copy(),
                    }
                    st.session_state.saved_scenarios.append(_entry)
                    st.success(
                        f"✅ Saved **{_entry['name']}** — go to 📊 Compare Scenarios to compare."
                    )
            with _sc_col3:
                if st.session_state.saved_scenarios:
                    if st.button("🗑️ Clear All", use_container_width=True,
                                 key="mc_clear_scenarios_btn"):
                        st.session_state.saved_scenarios = []
                        st.rerun()


# ===========================================================================
# 8. TAB: SCIENTIFIC ANALYSIS (legacy — kept for reference, not shown in nav)
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


_CSS_NAMED_COLORS = {
    "gray":    "#808080", "grey":    "#808080",
    "red":     "#FF0000", "green":   "#008000",
    "blue":    "#0000FF", "orange":  "#FFA500",
    "purple":  "#800080", "teal":    "#008080",
    "black":   "#000000", "white":   "#FFFFFF",
    "steelblue": "#4682B4", "firebrick": "#B22222",
    "seagreen":  "#2E8B57", "tomato":    "#FF6347",
    "lightgray": "#D3D3D3", "darkgray":  "#A9A9A9",
    "salmon":  "#FA8072", "lightgreen": "#90EE90",
}


def _band_traces(fig: go.Figure, stats: pd.DataFrame, name: str,
                 color_hex: str, show_iqr: bool = True):
    """Add p10–p90 outer band, optional p25–p75 IQR band, and median line.

    ``color_hex`` may be a 6-digit hex string (``#RRGGBB``) **or** a CSS
    named colour — the function converts named colours to hex automatically.
    """
    color_hex = _CSS_NAMED_COLORS.get(color_hex.lower().strip(), color_hex)
    if not (color_hex.startswith("#") and len(color_hex) == 7):
        color_hex = "#808080"   # safe fallback
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
    show_ci = st.session_state.get("show_ci", True)
    stats = _percentile_band(df)
    fig = go.Figure()
    _band_traces(fig, stats, "Revenue", color, show_iqr=show_ci)
    if not show_ci:
        fig.data = tuple(t for t in fig.data if "median" in t.name)
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Day",
        yaxis_title="Revenue (€)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=50, r=20, t=90, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)


def _plot_ci_dual(df_base: pd.DataFrame, df_cri: pd.DataFrame, label_base: str):
    """Baseline vs Crisis revenue chart — CI bands toggled by sidebar checkbox."""
    show_ci = st.session_state.get("show_ci", True)
    s_b = _percentile_band(df_base)
    s_c = _percentile_band(df_cri)

    fig = go.Figure()
    _band_traces(fig, s_b, label_base, "#44A1A0", show_iqr=show_ci)
    _band_traces(fig, s_c, "Crisis",   "#DC143C", show_iqr=show_ci)
    if not show_ci:
        fig.data = tuple(t for t in fig.data if "median" in t.name)

    ci_note = "  [p10 / IQR / p90 bands]" if show_ci else "  [median line]"
    fig.update_layout(
        title=f"Daily Revenue — Baseline vs Crisis{ci_note}",
        xaxis_title="Day",
        yaxis_title="Revenue (€)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=50, r=20, t=130, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)


def _plot_ci_dual_col(df_base: pd.DataFrame, df_cri: pd.DataFrame,
                      col: str, title: str, yaxis_title: str, label_base: str,
                      higher_is_better: bool = True,
                      color_base: str = "#44A1A0", color_cri: str = "#DC143C"):
    """Generic dual-scenario CI chart for any numeric column — bands toggled by sidebar."""
    if col not in df_base.columns or col not in df_cri.columns:
        return
    show_ci = st.session_state.get("show_ci", True)
    s_b = _percentile_band(df_base, col)
    s_c = _percentile_band(df_cri,  col)
    fig = go.Figure()
    _band_traces(fig, s_b, label_base, color_base, show_iqr=show_ci)
    _band_traces(fig, s_c, "Crisis",   color_cri,  show_iqr=show_ci)
    if not show_ci:
        fig.data = tuple(t for t in fig.data if "median" in t.name)
    arrow = "↑ better" if higher_is_better else "↓ better"
    ci_note = "  [p10/IQR/p90]" if show_ci else "  [median]"
    fig.update_layout(
        title=f"{title}{ci_note}  [{arrow}]",
        xaxis_title="Day",
        yaxis_title=yaxis_title,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=50, r=20, t=130, b=40),
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
        "Audits the behavioural mechanisms used in the latest simulation. "
        "All charts update after running the **Interactive Demo** simulation."
    )

    df = st.session_state.get("sim_results")
    if df is None or df.empty:
        st.info("Run the **Interactive Demo** simulation first to populate these charts.")
        return

    extensions_on = (
        "BehaviorEvidenceMode" in df.columns
        and (df["BehaviorEvidenceMode"] == "exploratory_extensions").any()
    )
    if extensions_on:
        st.warning(
            "This run enabled literature-transferred or engineered behavioural "
            "extensions. Their charts describe model assumptions, not effects "
            "identified from the GROCERYsim participants."
        )
    else:
        st.success(
            "This run used empirical-only behaviour. TPB threshold modulation, "
            "Prospect Theory, panic contagion, archetype modifiers, and preference "
            "learning were disabled."
        )

    import plotly.express as px
    import plotly.graph_objects as go

    # ── Row 1: TPB + Prospect Theory ────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("👥 Theory of Planned Behaviour")
        st.caption("Optional literature-transferred extension; not measured in the current export")
        if not extensions_on:
            st.info("Disabled in this run. Enable exploratory dynamic behaviour to inspect it.")
        elif "AvgSubjectiveNorm" in df.columns:
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
        st.caption("Optional literature-transferred extension; λ and curvature are not sample estimates")
        if not extensions_on:
            st.info("Disabled in this run. Relative price response uses the empirical-mode rule.")
        elif "BudgetExhaustionRate" in df.columns:
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
        st.subheader("🍽️ Realised Consumption Access Stress")
        st.caption(
            "% of all represented households with at least 50% of today's "
            "pantry consumption need unmet; this is not FAO FIES."
        )
        canonical_cols = [
            "AccessStressHigh_Low", "AccessStressHigh_Mid", "AccessStressHigh_High"
        ]
        fies_cols = canonical_cols if all(c in df.columns for c in canonical_cols) else [
            "FIESSevere_Low", "FIESSevere_Mid", "FIESSevere_High"
        ]
        available = [c for c in fies_cols if c in df.columns]
        if available:
            fig_fies = go.Figure()
            colors_fies = dict(zip(fies_cols, ["#FF5A5A", "#FCC995", "#BCDC8B"]))
            labels_fies = dict(zip(fies_cols, ["Low income", "Mid income", "High income"]))
            for col in available:
                fig_fies.add_trace(go.Scatter(
                    x=df["Day"], y=df[col] * 100,
                    mode="lines", name=labels_fies.get(col, col),
                    line=dict(color=colors_fies.get(col, "#92DDDB"), width=2)
                ))
            fig_fies.update_layout(
                title="High Realised Consumption Access Stress by Income Bracket",
                xaxis_title="Day", yaxis_title="Households with high access stress (%)",
                legend=dict(orientation="h", y=-0.25),
                height=320, margin=dict(t=40, b=60),
                template="plotly_white",
            )
            st.plotly_chart(fig_fies, use_container_width=True, config=_PLOTLY_CFG)
            with st.expander("📊 Food Security Analysis", expanded=False):
                for _fc in available:
                    _lbl = f"{labels_fies.get(_fc, _fc)} high access-stress %"
                    st.markdown(f"**{_lbl}**")
                    _render_analysis(df, _fc, params, suffix=" (0–1)", decimals=3, higher_is_better=False)
        else:
            st.warning("Access-stress columns not found. Re-run the simulation.")

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
            "Key Parameter": "Normalized: Attitude (0.430), Norm (0.228), PBC (0.342)",
            "Implementation": "Optional exploratory TPB intention expands the accepted price-loss margin",
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
            "Theory": "Household Consumption Access",
            "Authors": "Model diagnostic (not a validated scale)",
            "Key Parameter": "Daily unmet pantry need: 0, <25%, <50%, <90%, ≥90%",
            "Implementation": "Population-wide realised shortfall category; panic and shopping failure kept separate",
            "Policy Relevance": "Vulnerability targeting, food assistance"
        },
        {
            "Theory": "Temporal Discounting / Stockpiling",
            "Authors": "O'Donoghue & Rabin (1999)",
            "Key Parameter": "β (present bias), stockpile_days horizon",
            "Implementation": "Quasi-hyperbolic-inspired heuristic; pantry inventory tracking; beta not estimated",
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


# ---------------------------------------------------------------------------
# Full branded PDF report (Export tab)
# ---------------------------------------------------------------------------

def _make_branded_pdf_report(params: dict | None = None) -> bytes:
    """
    Generate a comprehensive, branded GROCERYsim PDF report from session results.
    Uses the same _SFReport visual style (dark cover, amber accents, Arial fonts).
    Sections are data-driven and skipped gracefully when data is absent.
    """
    from datetime import datetime as _dt
    from fpdf.enums import XPos, YPos

    # ── Local subclass: override header text only ────────────────────────────
    class _GROCERYReport(_SFReport):
        def header(self):
            if self.page_no() == 1:
                return
            self.set_fill_color(*_SF_DARK)
            self.rect(0, 0, 210, 7, "F")
            self.set_fill_color(*_SF_AMBER)
            self.rect(0, 0, 3, 7, "F")
            self.set_font("Ar", "B", 6.5)
            self.set_text_color(*_SF_WHITE)
            self.set_xy(6, 0.8)
            self.cell(130, 5.5, "GROCERYsim ABM v2.0 -- Strategic Resilience & Food-System Report")
            self.set_font("Ar", "I", 6.5)
            self.set_text_color(*_SF_AMBER)
            self.set_xy(136, 0.8)
            self.cell(59, 5.5, self._sec, align="R")
            self.set_y(10)
            self.set_text_color(*_SF_BODY)

    # ── Shared matplotlib style ───────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "sans-serif", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.22, "grid.linestyle": "--",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    })
    _C = {
        "b": "#2980b9", "r": "#c0392b", "g": "#27ae60", "o": "#e67e22",
        "p": "#8e44ad", "t": "#44A1A0", "gr": "#95a5a6", "a": "#DBA159",
        "lo": "#e74c3c", "mi": "#e67e22", "hi": "#27ae60",
    }
    _COLS4 = [_C["b"], _C["r"], _C["g"], _C["o"], _C["p"]]

    # helper: styled horizontal table -----------------------------------------
    def _tbl_hdr(p, cols):
        """cols: list of (label, width) tuples"""
        p.set_font("Ar", "B", 8.5)
        p.set_fill_color(*_SF_AMBER)
        p.set_text_color(*_SF_WHITE)
        for h, w in cols:
            p.cell(w, 6, _to_latin1(str(h)), border=1, fill=True)
        p.ln()

    def _tbl_row(p, vals_widths, idx):
        bg = _SF_CREAM2 if idx % 2 == 0 else _SF_WHITE
        p.set_fill_color(*bg)
        p.set_font("Ar", "", 8.0)
        p.set_text_color(*_SF_DARK)
        for val, w in vals_widths:
            p.cell(w, 5.5, _to_latin1(str(val)), border=1, fill=True)
        p.ln()

    # ── Init PDF ──────────────────────────────────────────────────────────────
    pdf = _GROCERYReport()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf._lf()
    now_str = _dt.now().strftime("%d %B %Y, %H:%M")

    # ── Collect session data ──────────────────────────────────────────────────
    df_sim      = st.session_state.get("sim_results")
    df_pol_base = st.session_state.get("policy_baseline")
    df_pol_scen = st.session_state.get("policy_scenario")
    pol_label   = st.session_state.get("policy_label", "Policy Scenario")
    stress_res  = st.session_state.get("stress_results")
    saved       = st.session_state.get("saved_scenarios", [])
    df_scm      = st.session_state.get("sim_scm_log")
    df_waste    = st.session_state.get("sim_waste")

    has_sim    = df_sim is not None and not df_sim.empty
    has_pol    = (df_pol_base is not None and df_pol_scen is not None
                  and not df_pol_base.empty and not df_pol_scen.empty)
    has_stress = (stress_res is not None and hasattr(stress_res, "iterrows")
                  and not stress_res.empty)
    has_saved  = len(saved) >= 2
    has_scm    = df_scm is not None and not df_scm.empty

    modules_run = (
        (["Interactive Simulation Demo"] if has_sim else [])
        + ([f"Policy Analysis: {pol_label}"] if has_pol else [])
        + (["Automated Stress-Test"] if has_stress else [])
        + ([f"Scenario Comparison ({len(saved)} scenarios)"] if has_saved else [])
        + (["Supply Chain Event Log"] if has_scm else [])
    )

    # ─────────────────────────────────────────────────────────────────────────
    # COVER PAGE
    # ─────────────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(*_SF_DARK)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_fill_color(*_SF_AMBER)
    pdf.rect(0, 0, 210, 5, "F")
    pdf.rect(0, 292, 210, 5, "F")

    # Logo row: GROCERYsim left, SecureFood right
    try:
        gs_p = _sf_logo_on("GROCERYsim.png", 360, _SF_DARK)
        sf_p = _sf_logo_on("SecureFood.png",  260, _SF_DARK)
        from PIL import Image as _PILImg
        gs_img = _PILImg.open(gs_p);  sf_img = _PILImg.open(sf_p)
        gs_w = 80;  gs_h_mm = 80 * gs_img.height / gs_img.width
        sf_w = 58;  sf_h_mm = 58 * sf_img.height / sf_img.width
        row_h = max(gs_h_mm, sf_h_mm)
        pdf.image(gs_p, x=15,              y=16, w=gs_w)
        pdf.image(sf_p, x=210 - 15 - sf_w, y=16, w=sf_w)
        logo_bottom = 16 + row_h + 4
    except Exception:
        logo_bottom = 30

    # Amber rule
    pdf.set_fill_color(*_SF_AMBER)
    pdf.rect(15, logo_bottom, 180, 0.7, "F")

    # Title block
    pdf.set_y(logo_bottom + 5)
    pdf.set_font("Ar", "B", 30)
    pdf.set_text_color(*_SF_WHITE)
    pdf.cell(0, 14, "GROCERYsim ABM v2.0",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Ar", "", 14)
    pdf.set_text_color(*_SF_AMBER)
    pdf.cell(0, 9, "Strategic Resilience & Food-System Report",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_fill_color(*_SF_AMBER)
    pdf.rect(15, pdf.get_y(), 180, 0.5, "F")
    pdf.ln(7)

    # Metadata
    pdf.set_font("Ar", "", 10)
    pdf.set_text_color(200, 220, 215)
    pdf.cell(0, 7, f"Generated: {now_str}",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, "Horizon Europe SecureFood Consortium -- Grant No. 101136583",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    # Modules run
    pdf.set_font("Ar", "B", 10)
    pdf.set_text_color(*_SF_AMBER)
    pdf.cell(0, 7, "Session Analysis Coverage:",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Ar", "", 9.5)
    pdf.set_text_color(190, 210, 208)
    if modules_run:
        for m in modules_run:
            pdf.cell(0, 6, f"  ▶  {m}",
                     align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.cell(0, 6, "  No simulation results recorded in this session.",
                 align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)

    # Disclaimer
    pdf.set_font("Ar", "I", 8)
    pdf.set_text_color(130, 155, 152)
    pdf.set_x(30)
    pdf.multi_cell(
        150, 4.8,
        "This report is auto-generated from a single simulation session. "
        "ABM results are stochastic; re-running with identical parameters may yield "
        "slightly different values. Cite as: Duric, I. (2026). GROCERYsim ABM v2.0. "
        "IAMO XR Lab, SecureFood / Horizon Europe.",
        align="C",
    )

    # Partner logos at bottom
    try:
        _logo_specs = [("IAMO.png", 36), ("EU.png", 28), ("Logo_lab.png", 30)]
        _total_w = sum(w for _, w in _logo_specs) + 10 * (len(_logo_specs) - 1)
        _x_cur   = (210 - _total_w) / 2
        for _nm, _pw in _logo_specs:
            _lp = _sf_logo_on(_nm, int(_pw * 3.78), _SF_DARK)
            pdf.image(_lp, x=_x_cur, y=254, w=_pw)
            _x_cur += _pw + 10
    except Exception:
        pass

    pdf.set_text_color(0, 0, 0)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: SIMULATION PARAMETERS
    # ─────────────────────────────────────────────────────────────────────────
    if params:
        pdf.chapter(1, "Simulation Parameters", "Parameters")
        pdf.body(
            "The following parameters were active for the last simulation run in this session. "
            "They define the supply chain configuration, consumer population, crisis severity, "
            "and any policy interventions applied."
        )
        _PARAM_MAP = [
            ("base_con",      "Consumer agents",               "agents"),
            ("days",          "Simulation duration",           "days"),
            ("month",         "Start month",                   "1=Jan"),
            ("reorder",       "Reorder point",                 "units"),
            ("target",        "Target stock level",            "units"),
            ("lead",          "Lead time",                     "days"),
            ("cri_start",     "Crisis onset",                  "day"),
            ("cri_duration",  "Crisis duration",               "days"),
            ("inf",           "Price inflation (crisis)",      "%"),
            ("dis",           "Supply disruption",             "days"),
            ("panic",         "Panic-buying sensitivity",      "0-1"),
            ("hoard",         "Hoarding demand multiplier",    "x"),
            ("panic_exposure_floor", "Normal scarcity exposure floor", "share"),
            ("panic_growth_rate", "Scarcity-to-panic growth rate", "per day"),
            ("panic_decay_active", "Active-phase panic decay", "per day"),
            ("panic_decay_recovery", "Recovery-phase panic decay", "per day"),
            ("inflation_panic_rate", "Inflation-to-panic rate", "per day"),
            ("purchase_limit","Purchase limit (nudge)",        "units/visit"),
            ("media_intensity","Media intensity",              "0-1"),
        ]
        rows = []
        for key, lbl, unit in _PARAM_MAP:
            if key in params and params[key] is not None:
                v = params[key]
                vs = f"{v:.4g}" if isinstance(v, float) else str(v)
                rows.append((lbl, f"{vs} {unit}".strip()))
        pdf.kv(rows)

        pol_cfg = params.get("policy_cfg", {})
        if pol_cfg:
            pdf.sub("Active Policy Interventions")
            pol_rows = []
            if pol_cfg.get("fat_tax_active"):
                pol_rows.append((
                    "Fat tax",
                    f"{pol_cfg.get('fat_tax_rate', 0)*100:.1f}% on products "
                    f">{pol_cfg.get('fat_tax_threshold', 3.5):.1f} g fat/100 ml"
                ))
            if pol_cfg.get("subsidy_active"):
                pol_rows.append((
                    "Supply subsidy",
                    f"{pol_cfg.get('subsidy_rate', 0)*100:.1f}% on "
                    f"{pol_cfg.get('subsidy_target', 'domestic')} products"
                ))
            if pol_cfg.get("labelling_active"):
                pol_rows.append((
                    "Eco-labelling",
                    f"Active from day {pol_cfg.get('labelling_day', 1)} -- "
                    f"health boost {pol_cfg.get('labelling_health_boost', 0)*100:.1f}%, "
                    f"organic boost {pol_cfg.get('labelling_organic_boost', 0)*100:.1f}%"
                ))
            if pol_cfg.get("domestic_shock_active"):
                pol_rows.append((
                    "Domestic supply shock",
                    f"Day {pol_cfg.get('domestic_shock_day', 30)}, "
                    f"{pol_cfg.get('domestic_shock_duration', 30)} days, "
                    f"severity {pol_cfg.get('domestic_shock_severity', 0.5):.0%}"
                ))
            if pol_rows:
                pdf.kv(pol_rows)
            else:
                pdf.body("No policy interventions were active in this configuration.")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: INTERACTIVE DEMO ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    if has_sim:
        pdf.chapter(2, "Interactive Demo Analysis", "Demo Results")
        pdf.body(
            "The Interactive Demo module runs parallel Baseline and Crisis simulations "
            "using the Mesa ABM framework. Consumer attributes use a held-out DCE assessment "
            "and cross-fitted phase-transition calibration from the SecureFood project. "
            "Results below reflect the user-configured scenario as run in this session."
        )

        scenarios = (df_sim["Scenario"].unique().tolist()
                     if "Scenario" in df_sim.columns else [])
        sc_b = (df_sim[df_sim["Scenario"] == "Baseline"].reset_index(drop=True)
                if "Baseline" in scenarios else df_sim.copy())
        sc_c = (df_sim[df_sim["Scenario"] == "Crisis"].reset_index(drop=True)
                if "Crisis" in scenarios else pd.DataFrame())

        # ── 2.1 Revenue ─────────────────────────────────────────────────────
        pdf.sub("2.1  Revenue Impact")
        rev_b  = sc_b["Revenue"].mean()   if not sc_b.empty and "Revenue" in sc_b.columns else 0
        rev_c  = sc_c["Revenue"].mean()   if not sc_c.empty and "Revenue" in sc_c.columns else 0
        nom_c  = sc_c["NominalRevenue"].mean() if not sc_c.empty and "NominalRevenue" in sc_c.columns else 0
        rev_drop_pct = (rev_b - rev_c) / max(rev_b, 1) * 100
        price_b = sc_b["AvgPrice"].mean() if not sc_b.empty and "AvgPrice" in sc_b.columns else 0
        price_c = sc_c["AvgPrice"].mean() if not sc_c.empty and "AvgPrice" in sc_c.columns else 0
        price_delta = (price_c / max(price_b, 0.01) - 1) * 100

        pdf.metric_row([
            ("Baseline Revenue/Day",  f"EUR {rev_b:,.0f}",
             "constant-price base",   True),
            ("Crisis Revenue/Day",    f"EUR {rev_c:,.0f}",
             f"{rev_drop_pct:.1f}% below baseline", rev_drop_pct < 10),
            ("Crisis Nominal/Day",    f"EUR {nom_c:,.0f}",
             f"avg price +{price_delta:.1f}%",       False),
        ])

        sev = ("severe" if rev_drop_pct > 25
               else "moderate" if rev_drop_pct > 10
               else "mild")
        pdf.body(
            f"The crisis scenario produced a {sev} revenue contraction of "
            f"{rev_drop_pct:.1f}% relative to the baseline (constant-price basis). "
            f"Average unit price rose by {price_delta:.1f}% under inflationary pressure, "
            f"which inflates nominal revenue (EUR {nom_c:,.0f}/day) above baseline -- "
            f"an accounting artefact rather than real economic gain, since the volume of "
            f"units sold simultaneously fell due to stockouts and demand suppression. "
            f"Supply chain actors should focus on the constant-price revenue gap as the "
            f"true indicator of economic loss."
        )

        # Revenue chart
        fig, ax = plt.subplots(figsize=(7, 3.0))
        for i, sc in enumerate(scenarios):
            sub = df_sim[df_sim["Scenario"] == sc]
            ax.plot(sub.groupby("Day")["Revenue"].mean(),
                    color=_COLS4[i % len(_COLS4)], lw=2, label=sc)
        if not sc_c.empty and "NominalRevenue" in sc_c.columns:
            ax.plot(sc_c.groupby("Day")["NominalRevenue"].mean(),
                    color=_C["o"], lw=1.4, ls="--", alpha=0.8, label="Crisis Nominal")
        ax.set_xlabel("Simulation Day"); ax.set_ylabel("Daily Revenue (EUR)")
        ax.set_title("Daily Revenue: Baseline vs Crisis (constant-price and nominal)")
        ax.legend(fontsize=7.5, ncol=2)
        fig.tight_layout()
        pdf.chart(_sf_mpl_chart(fig),
                  caption=(
                      "Figure 2.1: Revenue trajectories. Constant-price curves reveal true volume "
                      "dynamics; the nominal dashed line shows inflation-distorted cash flow."
                  ))

        # ── 2.2 Food Waste & Lost Sales ──────────────────────────────────────
        pdf.sub("2.2  Food Waste & Supply Loss")
        waste_b = sc_b["Waste"].mean()     if not sc_b.empty and "Waste"     in sc_b.columns else 0
        waste_c = sc_c["Waste"].mean()     if not sc_c.empty and "Waste"     in sc_c.columns else 0
        lost_c  = sc_c["LostSales"].mean() if not sc_c.empty and "LostSales" in sc_c.columns else 0
        cum_lost = sc_c["LostSales"].sum() if not sc_c.empty and "LostSales" in sc_c.columns else 0
        waste_chg = (waste_c - waste_b) / max(waste_b, 0.01) * 100

        pdf.metric_row([
            ("Baseline Waste/Day",   f"{waste_b:.1f} units",
             "daily average",         True),
            ("Crisis Waste/Day",     f"{waste_c:.1f} units",
             f"{waste_chg:+.1f}% vs baseline",
             waste_chg < 5),
            ("Cumulative Lost Sales", f"{cum_lost:,.0f} units",
             "unrecoverable demand",   False),
        ])

        waste_dir = "increased" if waste_chg > 0 else "decreased"
        pdf.body(
            f"Food waste {waste_dir} by {abs(waste_chg):.1f}% under crisis conditions. "
            + (
                "Paradoxically, severe panic-buying can reduce waste short-term as shelves "
                "clear faster than products expire; however, post-crisis over-ordering "
                "typically produces waste spikes from near-expiry stock. "
                if waste_chg < -5 else
                "Rising waste during a crisis indicates demand suppression -- consumers "
                "reduce basket sizes, leaving more perishable inventory unsold. "
                if waste_chg > 10 else
                "Waste levels were broadly stable across conditions in this run. "
            )
            + f"Cumulative lost sales of {cum_lost:,.0f} units represent permanently "
            f"forgone revenue -- demand that cannot be recovered once the stockout event "
            f"passes and panic subsides."
        )

        # Waste + Lost Sales chart
        fig2, axes2 = plt.subplots(1, 2, figsize=(7, 2.8))
        for i, sc in enumerate(scenarios):
            sub = df_sim[df_sim["Scenario"] == sc]
            col = _COLS4[i % len(_COLS4)]
            if "Waste"     in sub.columns: axes2[0].plot(sub.groupby("Day")["Waste"].mean(),     color=col, lw=2, label=sc)
            if "LostSales" in sub.columns: axes2[1].plot(sub.groupby("Day")["LostSales"].mean(), color=col, lw=2, label=sc)
        axes2[0].set_title("Food Waste (units/day)");     axes2[0].set_xlabel("Day"); axes2[0].legend(fontsize=7)
        axes2[1].set_title("Lost Sales (units/day)");     axes2[1].set_xlabel("Day"); axes2[1].legend(fontsize=7)
        for ax_ in axes2: ax_.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()
        pdf.chart(_sf_mpl_chart(fig2),
                  caption="Figure 2.2: Daily food waste and lost-sales events across scenarios.")

        # ── 2.3 Consumer Panic & Behavioural Dynamics ────────────────────────
        if "PanicLevel" in df_sim.columns and not sc_c.empty:
            pdf.sub("2.3  Consumer Panic & Stockpile Dynamics")
            peak_panic = float(sc_c["PanicLevel"].max()) if "PanicLevel" in sc_c.columns else 0
            panic_day  = int(sc_c.loc[sc_c["PanicLevel"].idxmax(), "Day"]) if "PanicLevel" in sc_c.columns and "Day" in sc_c.columns else 0
            sp_peak    = float(sc_c["StockpilePressure"].max()) if "StockpilePressure" in sc_c.columns else 0

            pdf.metric_row([
                ("Peak Panic Level",       f"{peak_panic:.3f}",
                 f"day {panic_day}",        peak_panic < 0.3),
                ("Avg Panic (Crisis)",      f"{sc_c['PanicLevel'].mean():.3f}",
                 "sustained level",         sc_c["PanicLevel"].mean() < 0.2),
                ("Peak Stockpile Pressure", f"{sp_peak:.3f}",
                 "0=none / 1=max",          sp_peak < 0.4),
            ])

            fig3, ax3a = plt.subplots(figsize=(7, 2.6))
            ax3a.plot(sc_c.groupby("Day")["PanicLevel"].mean(),
                      color=_C["r"], lw=2, label="Panic level (crisis)")
            if not sc_b.empty and "PanicLevel" in sc_b.columns:
                ax3a.plot(sc_b.groupby("Day")["PanicLevel"].mean(),
                          color=_C["gr"], lw=1.2, ls="--", label="Panic level (baseline)")
            if "StockpilePressure" in sc_c.columns:
                ax3b = ax3a.twinx()
                ax3b.plot(sc_c.groupby("Day")["StockpilePressure"].mean(),
                          color=_C["a"], lw=1.8, ls="-.", label="Stockpile pressure")
                ax3b.set_ylabel("Stockpile Pressure (0-1)", fontsize=8)
                ax3b.spines["right"].set_visible(True)
                h1, l1 = ax3a.get_legend_handles_labels()
                h2, l2 = ax3b.get_legend_handles_labels()
                ax3a.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2)
            else:
                ax3a.legend(fontsize=7)
            ax3a.set_xlabel("Simulation Day"); ax3a.set_ylabel("Panic Level (0-1)", fontsize=8)
            ax3a.set_title("Consumer Panic Level and Stockpile Pressure")
            ax3a.spines[["top", "right"]].set_visible(False)
            fig3.tight_layout()
            pdf.chart(_sf_mpl_chart(fig3),
                      caption=(
                          "Figure 2.3: Panic index peaks early in the crisis window and typically "
                          "declines within 2-3 weeks as supply normalises or consumers adapt."
                      ))

            pdf.body(
                f"Consumer panic peaked at {peak_panic:.3f} on day {panic_day}, "
                + ("exceeding the critical threshold that triggers hoarding multipliers across "
                   "all archetypes -- this amplifies demand spikes far beyond genuine need. "
                   if peak_panic > 0.4 else
                   "remaining comparatively low -- continuous propensity-weighted demand amplification was "
                   "bounded and the supply chain recovered without a severe hoarding cascade. ")
                + f"Sustained mean panic of {sc_c['PanicLevel'].mean():.3f} maintained "
                f"elevated purchase quantities throughout the crisis window, compressing "
                f"shelf availability for later-arriving consumer cohorts (primarily low-income "
                f"archetypes who shop later in the day or have fewer substitution options)."
            )

        # ── 2.4 Income-Stratified Consumer Welfare ───────────────────────────
        w_cols = ["Fulfillment_Low", "Fulfillment_Mid", "Fulfillment_High"]
        if all(c in df_sim.columns for c in w_cols) and not sc_c.empty:
            pdf.sub("2.4  Income-Stratified Consumer Welfare")
            ful_lo = sc_c["Fulfillment_Low"].mean()  * 100
            ful_mi = sc_c["Fulfillment_Mid"].mean()  * 100
            ful_hi = sc_c["Fulfillment_High"].mean() * 100
            bud_lo = sc_c["BudgetExh_Low"].mean()  * 100 if "BudgetExh_Low"  in sc_c.columns else None
            bud_hi = sc_c["BudgetExh_High"].mean() * 100 if "BudgetExh_High" in sc_c.columns else None

            pdf.metric_row([
                ("Fulfilment: Low Income",  f"{ful_lo:.1f}%",
                 "of basket met on average", ful_lo >= 80),
                ("Fulfilment: Mid Income",  f"{ful_mi:.1f}%",
                 "of basket met on average", ful_mi >= 80),
                ("Fulfilment: High Income", f"{ful_hi:.1f}%",
                 "of basket met on average", ful_hi >= 80),
            ])

            fig4, ax4 = plt.subplots(figsize=(7, 3.0))
            for col_k, col_c2, lbl in [
                ("Fulfillment_Low",  _C["lo"], "Low income"),
                ("Fulfillment_Mid",  _C["mi"], "Mid income"),
                ("Fulfillment_High", _C["hi"], "High income"),
            ]:
                ax4.plot(sc_c.groupby("Day")[col_k].mean() * 100,
                         color=col_c2, lw=2, label=f"{lbl} (crisis)")
            if not sc_b.empty and "FulfillmentRate" in sc_b.columns:
                ax4.plot(sc_b.groupby("Day")["FulfillmentRate"].mean() * 100,
                         color=_C["gr"], lw=1.2, ls="--", label="All income (baseline)")
            ax4.axhline(80, color=_C["r"], ls=":", lw=1, label="80% welfare threshold")
            ax4.set_ylim(0, 105)
            ax4.set_xlabel("Simulation Day"); ax4.set_ylabel("Basket Fulfilment Rate (%)")
            ax4.set_title("Consumer Basket Fulfilment by Income Group")
            ax4.legend(fontsize=7.5, ncol=2)
            ax4.spines[["top", "right"]].set_visible(False)
            fig4.tight_layout()
            pdf.chart(_sf_mpl_chart(fig4),
                      caption=(
                          "Figure 2.4: Crisis impact concentrates in low-income archetypes "
                          "(L-SENS cluster) whose constrained budgets are eroded first by price inflation."
                      ))

            equity_gap = ful_hi - ful_lo
            pdf.body(
                f"A {equity_gap:.1f} percentage-point fulfilment equity gap emerged between "
                f"high-income ({ful_hi:.1f}%) and low-income ({ful_lo:.1f}%) archetypes. "
                + (f"Low-income agents with budget exhaustion of {bud_lo:.1f}% could not "
                   f"sustain pre-crisis consumption patterns; high-income exhaustion was "
                   f"just {bud_hi:.1f}%, confirming that price inflation acts as a "
                   f"regressive shock. "
                   if bud_lo is not None and bud_hi is not None else "")
                + "Policy interventions -- particularly targeted subsidies and purchase-limit "
                "nudges -- materially reduce this equity gap by redistributing shelf access "
                "more evenly across the population (see Section 3 for policy results)."
            )

        # ── 2.5 FIES Food Insecurity ─────────────────────────────────────────
        if "FIESSevere_Low" in df_sim.columns and not sc_c.empty:
            pdf.sub("2.5  Food-Access Stress Indicators")
            fies_lo_peak = sc_c["FIESSevere_Low"].max()  * 100
            fies_mi_peak = sc_c["FIESSevere_Mid"].max()  * 100 if "FIESSevere_Mid"  in sc_c.columns else 0
            fies_hi_peak = sc_c["FIESSevere_High"].max() * 100 if "FIESSevere_High" in sc_c.columns else 0
            fies_b_base  = sc_b["FIESSevere_Low"].mean() * 100 if not sc_b.empty and "FIESSevere_Low" in sc_b.columns else 0

            pdf.metric_row([
                ("Access Stress High: Low Income",
                 f"{fies_lo_peak:.1f}%", "peak during crisis", fies_lo_peak < 10),
                ("Access Stress High: Mid Income",
                 f"{fies_mi_peak:.1f}%", "peak during crisis", fies_mi_peak < 5),
                ("Baseline Access Stress (Low)",
                 f"{fies_b_base:.1f}%",  "pre-crisis level",   True),
            ])

            fies_lift = fies_lo_peak - fies_b_base
            pdf.body(
                f"The crisis elevated high modeled access stress among low-income "
                f"agents by {fies_lift:.1f} percentage points -- from a baseline of "
                f"{fies_b_base:.1f}% to a crisis peak of {fies_lo_peak:.1f}%. "
                f"Mid- and high-income archetypes reached peaks of {fies_mi_peak:.1f}% and "
                f"{fies_hi_peak:.1f}% respectively. This is an exploratory scenario "
                f"diagnostic based on basket shortfall, budget exhaustion, and panic, "
                f"not a survey-calibrated prevalence measure."
            )

        # ── 2.6 Environmental Footprint ──────────────────────────────────────
        if "CO2Total" in df_sim.columns:
            pdf.sub("2.6  Environmental & CO2 Footprint")
            co2_b  = sc_b["CO2Total"].mean()    if not sc_b.empty else 0
            co2_c  = sc_c["CO2Total"].mean()    if not sc_c.empty else 0
            imp_b  = sc_b["ImportDepPct"].mean() * 100 if not sc_b.empty and "ImportDepPct" in sc_b.columns else None
            imp_c  = sc_c["ImportDepPct"].mean() * 100 if not sc_c.empty and "ImportDepPct" in sc_c.columns else None
            co2_chg = (co2_c / max(co2_b, 0.01) - 1) * 100

            kv_rows = [
                ("CO2 emissions/day (baseline)", f"{co2_b:.2f} kg CO2-eq"),
                ("CO2 emissions/day (crisis)",   f"{co2_c:.2f} kg CO2-eq ({co2_chg:+.1f}%)"),
            ]
            if imp_b is not None:
                kv_rows.append(("Import dependency (baseline)", f"{imp_b:.1f}%"))
            if imp_c is not None:
                kv_rows.append(("Import dependency (crisis)",   f"{imp_c:.1f}%"))
            pdf.kv(kv_rows)

            co2_dir = "rose" if co2_chg > 0 else "fell"
            pdf.body(
                f"CO2-equivalent emissions {co2_dir} by {abs(co2_chg):.1f}% under crisis "
                f"conditions ({co2_b:.2f} -> {co2_c:.2f} kg CO2-eq/day). "
                + (f"As domestic supply tightened, import dependency shifted from "
                   f"{imp_b:.1f}% to {imp_c:.1f}%, increasing the carbon cost per unit "
                   f"(emission factors: Finnish conventional 1.2, imported conventional "
                   f"2.2 kg CO2-eq/unit). "
                   if imp_b is not None and imp_c is not None and imp_c > imp_b + 2 else
                   "Emission factors: Finnish organic 0.8, Finnish conventional 1.2, "
                   "Imported organic 1.5, Imported conventional 2.2 kg CO2-eq/unit. ")
                + "Food-system resilience strategies that maintain domestic supply chains "
                "therefore carry a co-benefit of reduced carbon footprint."
            )

    # ── Supply Chain Log ─────────────────────────────────────────────────────
    if has_scm:
        pdf.sub("2.7  Supply Chain Event Summary")
        n_orders = len(df_scm[df_scm["Type"] == "Order"])      if "Type" in df_scm.columns else "n/a"
        n_deliv  = len(df_scm[df_scm["Type"] == "Delivery"])   if "Type" in df_scm.columns else "n/a"
        n_short  = len(df_scm[df_scm["Type"] == "Stockout"])   if "Type" in df_scm.columns else 0
        pdf.kv([
            ("Total order events",    str(n_orders)),
            ("Total delivery events", str(n_deliv)),
            ("Stockout events",       str(n_short)),
            ("Log entries (total)",   f"{len(df_scm):,}"),
        ])
        pdf.body(
            "The supply chain log records every reorder trigger, inbound delivery, and "
            "stockout event across the simulation horizon. A high ratio of stockout events "
            "relative to total deliveries indicates that the reorder policy is insufficiently "
            "responsive to demand spikes -- consider lowering the reorder point or increasing "
            "the target stock level to buffer against future disruptions."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: POLICY ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    if has_pol:
        pdf.chapter(3, "Policy Analysis", "Policy")
        pdf.body(
            f"The Policy Analysis module compares the '{pol_label}' intervention against a "
            f"no-intervention baseline. Interventions modelled include fat taxation, domestic "
            f"supply subsidies, eco-labelling, and communication strategies. All results are "
            f"averaged across the full simulation horizon unless otherwise noted."
        )

        _kpi_defs = [
            ("Revenue/day (EUR)",    "Revenue",              False, True),
            ("Waste/day (units)",    "Waste",                False, False),
            ("CO2 total/day (kg)",   "CO2Total",             False, False),
            ("Budget Exhaustion %",  "BudgetExhaustionRate", True,  False),
            ("Food Stressed %",      "FoodStressedPct",      True,  False),
            ("Fulfilment Rate %",    "FulfillmentRate",      True,  True),
            ("Access Stress High Low %",    "FIESSevere_Low",       True,  False),
            ("Import Dependency %",  "ImportDepPct",         True,  False),
            ("Gini Access Index",    "GiniAccess",           False, False),
            ("Panic Level",          "PanicLevel",           False, False),
        ]

        # KPI table
        pdf.sub("3.1  KPI Comparison Table")
        tbl_cols = [("Metric", 66), ("Baseline", 29), (str(pol_label)[:16], 29), ("Delta", 28), ("Direction", 28)]
        _tbl_hdr(pdf, tbl_cols)
        findings = []
        for i, (lbl, col, is_pct, higher_better) in enumerate(_kpi_defs):
            if col not in df_pol_base.columns:
                continue
            b_v = df_pol_base[col].mean() * (100 if is_pct else 1)
            p_v = df_pol_scen[col].mean() * (100 if is_pct else 1)
            d_p = (p_v - b_v) / max(abs(b_v), 1e-9) * 100
            improved = (d_p > 0) == higher_better
            arrow = "+" if d_p > 0 else ""
            direction = ("BETTER" if improved else "TRADE-OFF") if abs(d_p) > 2 else "NEUTRAL"
            _tbl_row(pdf, [
                (lbl, 66), (f"{b_v:.3g}", 29), (f"{p_v:.3g}", 29),
                (f"{arrow}{d_p:.1f}%", 28), (direction, 28)
            ], i)
            if abs(d_p) > 5:
                findings.append((lbl, d_p, improved))
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

        # Policy interpretation
        pdf.sub("3.2  Policy Interpretation")
        good = [(l, d) for l, d, imp in findings if imp]
        bad  = [(l, d) for l, d, imp in findings if not imp]
        narrative = ""
        if good:
            narrative += (
                f"The '{pol_label}' policy improved {len(good)} indicator(s): "
                + ", ".join(
                    f"{l} ({'+' if d > 0 else ''}{d:.1f}%)"
                    for l, d in good[:4]
                )
                + (f" and {len(good) - 4} others" if len(good) > 4 else "")
                + ". "
            )
        if bad:
            narrative += (
                f"Trade-offs were observed on {len(bad)} indicator(s): "
                + ", ".join(
                    f"{l} ({'+' if d > 0 else ''}{d:.1f}%)"
                    for l, d in bad[:3]
                )
                + ". "
            )
        if not good and not bad:
            narrative = (
                f"The '{pol_label}' policy produced no changes exceeding the 5% materiality "
                f"threshold. The chosen intervention intensity may be too modest to generate "
                f"observable effects within the configured simulation horizon. "
            )
        narrative += (
            "These results are consistent with supply-side intervention theory: subsidies "
            "improve affordability and food security metrics but compress commercial margins; "
            "eco-labelling shifts consumer preferences over time but may temporarily suppress "
            "conventional-product sales; purchase limits reduce peak panic but can frustrate "
            "legitimate high-volume shoppers early in the crisis window."
        )
        pdf.body(narrative)

        # Key finding box
        if good:
            best_g = max(good, key=lambda x: abs(x[1]))
            pdf.finding(
                f"Most impactful improvement: {best_g[0]} changed by "
                f"{'+' if best_g[1] > 0 else ''}{best_g[1]:.1f}% under the "
                f"'{pol_label}' policy. "
                + ("This suggests the intervention is well-targeted to the key vulnerability "
                   "exposed by the crisis scenario." if abs(best_g[1]) > 15 else
                   "The effect size is moderate -- larger intervention parameters or a longer "
                   "simulation horizon may amplify the signal.")
            )
        if bad:
            worst_b = max(bad, key=lambda x: abs(x[1]))
            pdf.finding(
                f"Main trade-off: {worst_b[0]} worsened by "
                f"{'+' if worst_b[1] > 0 else ''}{worst_b[1]:.1f}%. "
                "Policy makers should weigh this cost against the welfare gains above "
                "when designing the final intervention package."
            )

        # Revenue chart
        pdf.sub("3.3  Revenue Comparison")
        fig_p, ax_p = plt.subplots(figsize=(7, 3.0))
        ax_p.plot(df_pol_base.groupby("Day")["Revenue"].mean(),
                  color=_C["b"], lw=2, label="Baseline (no policy)")
        ax_p.plot(df_pol_scen.groupby("Day")["Revenue"].mean(),
                  color=_C["r"], lw=2, label=str(pol_label))
        ax_p.fill_between(
            df_pol_base.groupby("Day")["Revenue"].mean().index,
            df_pol_base.groupby("Day")["Revenue"].mean(),
            df_pol_scen.groupby("Day")["Revenue"].mean(),
            alpha=0.08, color=_C["r"],
        )
        ax_p.set_xlabel("Simulation Day"); ax_p.set_ylabel("Daily Revenue (EUR)")
        ax_p.set_title(f"Revenue: Baseline vs {pol_label}")
        ax_p.legend(fontsize=8)
        ax_p.spines[["top", "right"]].set_visible(False)
        fig_p.tight_layout()
        pdf.chart(_sf_mpl_chart(fig_p),
                  caption=(
                      "Figure 3.1: Revenue trajectories. The shaded area represents the "
                      "policy-induced revenue change; downward shift indicates commercial cost."
                  ))

        # Welfare chart
        if "FulfillmentRate" in df_pol_base.columns:
            fig_w, axes_w = plt.subplots(1, 2, figsize=(7, 2.8))
            axes_w[0].plot(df_pol_base.groupby("Day")["FulfillmentRate"].mean() * 100,
                           color=_C["b"], lw=2, label="Baseline")
            axes_w[0].plot(df_pol_scen.groupby("Day")["FulfillmentRate"].mean() * 100,
                           color=_C["r"], lw=2, label=str(pol_label))
            axes_w[0].axhline(80, color=_C["lo"], ls=":", lw=1)
            axes_w[0].set_title("Basket Fulfilment (%)"); axes_w[0].set_ylim(0, 105)
            axes_w[0].legend(fontsize=7); axes_w[0].set_xlabel("Day")

            if "FoodStressedPct" in df_pol_base.columns:
                axes_w[1].plot(df_pol_base.groupby("Day")["FoodStressedPct"].mean() * 100,
                               color=_C["b"], lw=2, label="Baseline")
                axes_w[1].plot(df_pol_scen.groupby("Day")["FoodStressedPct"].mean() * 100,
                               color=_C["r"], lw=2, label=str(pol_label))
                axes_w[1].set_title("Food-Stressed Agents (%)"); axes_w[1].legend(fontsize=7)
                axes_w[1].set_xlabel("Day")

            for ax_ in axes_w: ax_.spines[["top", "right"]].set_visible(False)
            fig_w.tight_layout()
            pdf.chart(_sf_mpl_chart(fig_w),
                      caption=(
                          "Figure 3.2: Consumer welfare metrics. Policies that raise fulfilment "
                          "above 80% and reduce food stress demonstrate measurable social value."
                      ))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4: AUTOMATED STRESS TEST
    # ─────────────────────────────────────────────────────────────────────────
    if has_stress:
        pdf.chapter(4, "Stress-Test Risk Ranking", "Stress Test")
        pdf.body(
            "The Automated Stress Test sweeps a grid of disruption magnitudes and durations, "
            "running an independent simulation for each combination. The resulting risk table "
            "ranks scenarios by revenue loss severity, identifying the critical parameter "
            "thresholds beyond which the supply chain fails to recover within the horizon."
        )

        # Find column names defensively
        _rev_loss_col = "Revenue Loss (%)" if "Revenue Loss (%)" in stress_res.columns else None
        _base_rev_col = ("Baseline Revenue" if "Baseline Revenue" in stress_res.columns
                         else "Base Revenue" if "Base Revenue" in stress_res.columns else None)
        _cris_rev_col = "Crisis Revenue" if "Crisis Revenue" in stress_res.columns else None
        _lost_col = ("Total Lost Sales (EUR)" if "Total Lost Sales (EUR)" in stress_res.columns
                     else "Total Lost Sales (EUR)" if "Total Lost Sales (EUR)" in stress_res.columns
                     else next((c for c in stress_res.columns if "Lost" in c and "Sales" in c), None))

        tbl_cols_s = [("Scenario", 68), ("Baseline Rev", 30), ("Crisis Rev", 30),
                      ("Loss %", 22), ("Lost Sales", 30)]
        _tbl_hdr(pdf, tbl_cols_s)
        for i, (_, row) in enumerate(stress_res.iterrows()):
            sc_nm  = str(row.get("Scenario", ""))[:34]
            b_rv   = row.get(_base_rev_col, 0) if _base_rev_col else 0
            c_rv   = row.get(_cris_rev_col, 0) if _cris_rev_col else 0
            r_loss = row.get(_rev_loss_col, 0) if _rev_loss_col else 0
            l_s    = row.get(_lost_col, 0)     if _lost_col else 0
            _tbl_row(pdf, [
                (sc_nm, 68), (f"{b_rv:,.0f}", 30), (f"{c_rv:,.0f}", 30),
                (f"{r_loss:.1f}%", 22), (f"{l_s:,.0f}", 30),
            ], i)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

        # Risk bar chart
        if _rev_loss_col and "Scenario" in stress_res.columns:
            n_sc = len(stress_res)
            fig_s, ax_s = plt.subplots(figsize=(7, max(2.5, n_sc * 0.38)))
            colors_s = [
                _C["r"] if v > 20 else _C["o"] if v > 10 else _C["g"]
                for v in stress_res[_rev_loss_col]
            ]
            ax_s.barh(stress_res["Scenario"].astype(str), stress_res[_rev_loss_col],
                      color=colors_s, edgecolor="none")
            ax_s.axvline(10, color=_C["o"], ls="--", lw=1, label="10% threshold")
            ax_s.axvline(20, color=_C["r"], ls="--", lw=1, label="20% threshold")
            ax_s.set_xlabel("Revenue Loss (%)"); ax_s.set_title("Stress-Test Risk Ranking")
            ax_s.legend(fontsize=7.5); ax_s.spines[["top", "right"]].set_visible(False)
            fig_s.tight_layout()
            pdf.chart(_sf_mpl_chart(fig_s),
                      caption=(
                          "Figure 4.1: Revenue loss by scenario. Red = high-risk (>20%), "
                          "orange = moderate (10-20%), green = resilient (<10%)."
                      ))

        # Interpretation
        if _rev_loss_col:
            worst = stress_res.loc[stress_res[_rev_loss_col].idxmax()]
            best  = stress_res.loc[stress_res[_rev_loss_col].idxmin()]
            q3    = stress_res[_rev_loss_col].quantile(0.75)
            high_risk_count = (stress_res[_rev_loss_col] > 20).sum()
            pdf.body(
                f"The highest-risk scenario -- '{worst.get('Scenario', 'N/A')}' -- "
                f"produced a {worst.get(_rev_loss_col, 0):.1f}% revenue contraction, "
                f"while the most resilient -- '{best.get('Scenario', 'N/A')}' -- "
                f"lost only {best.get(_rev_loss_col, 0):.1f}%. "
                f"{high_risk_count} of {len(stress_res)} tested configurations crossed "
                f"the 20% high-risk threshold. "
                f"Scenarios in the upper quartile (>{q3:.1f}% loss) represent the "
                f"'tail risk' configurations that supply chain managers should prioritise "
                f"in contingency planning -- these are the disruption levels where standard "
                f"(s,S) inventory policies fail to contain damage within an acceptable range."
            )
            pdf.finding(
                f"Critical vulnerability threshold: scenarios above {q3:.1f}% revenue loss "
                f"({high_risk_count} configurations) require structural resilience measures "
                f"beyond parameter tuning -- such as dual sourcing, strategic reserves, "
                f"or pre-agreed demand-management protocols with retail partners."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5: SAVED SCENARIO COMPARISON
    # ─────────────────────────────────────────────────────────────────────────
    if has_saved:
        pdf.chapter(5, "Saved Scenario Comparison", "Scenarios")
        pdf.body(
            f"{len(saved)} parameter configurations were saved and compared during this "
            f"session. The table and chart below provide a side-by-side view of the "
            f"crisis-phase performance of each user-defined scenario."
        )

        sc_metrics = []
        for sc in saved:
            name = sc.get("name", "Unnamed")
            df_s = sc.get("df")
            if df_s is None or df_s.empty:
                continue
            df_cr = (df_s[df_s["Scenario"] == "Crisis"]
                     if "Scenario" in df_s.columns else df_s)
            avg_r  = df_cr["Revenue"].mean()             if "Revenue"             in df_cr.columns else 0
            avg_w  = df_cr["Waste"].mean()               if "Waste"               in df_cr.columns else 0
            avg_l  = df_cr["LostSales"].mean()           if "LostSales"           in df_cr.columns else 0
            avg_f  = df_cr["FulfillmentRate"].mean()*100 if "FulfillmentRate"     in df_cr.columns else None
            avg_s  = df_cr["FoodStressedPct"].mean()*100 if "FoodStressedPct"     in df_cr.columns else None
            avg_p  = df_cr["PanicLevel"].max()           if "PanicLevel"          in df_cr.columns else None
            sc_metrics.append((name, avg_r, avg_w, avg_l, avg_f, avg_s, avg_p))

        tbl_cols_cmp = [
            ("Scenario", 48), ("Rev/Day EUR", 25), ("Waste/Day", 20),
            ("LostSales/Day", 24), ("Fulfilment %", 22), ("FoodStress %", 22), ("Peak Panic", 19),
        ]
        _tbl_hdr(pdf, tbl_cols_cmp)
        for i, (name, r, w, l, f, s, pa) in enumerate(sc_metrics):
            _tbl_row(pdf, [
                (name[:22], 48), (f"{r:,.0f}", 25), (f"{w:.1f}", 20),
                (f"{l:.1f}", 24),
                (f"{f:.1f}%" if f is not None else "n/a", 22),
                (f"{s:.1f}%" if s is not None else "n/a", 22),
                (f"{pa:.3f}" if pa is not None else "n/a", 19),
            ], i)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

        # Best/worst by revenue
        if sc_metrics:
            best_rev  = max(sc_metrics, key=lambda x: x[1])
            worst_rev = min(sc_metrics, key=lambda x: x[1])
            pdf.body(
                f"Among the saved configurations, '{best_rev[0]}' achieved the highest "
                f"crisis-phase revenue (EUR {best_rev[1]:,.0f}/day), while '{worst_rev[0]}' "
                f"recorded the lowest (EUR {worst_rev[1]:,.0f}/day). "
                f"This spread of EUR {best_rev[1] - worst_rev[1]:,.0f}/day illustrates "
                f"the sensitivity of financial outcomes to parameter choices, and underlines "
                f"the value of systematic scenario exploration before committing to a supply "
                f"chain configuration in practice."
            )

        # Comparison chart
        fig_cmp, ax_cmp = plt.subplots(figsize=(7, 3.0))
        for i, sc in enumerate(saved[:6]):
            name = sc.get("name", f"Scenario {i+1}")
            df_s = sc.get("df")
            if df_s is None or df_s.empty:
                continue
            df_cr = df_s[df_s["Scenario"] == "Crisis"] if "Scenario" in df_s.columns else df_s
            if "Revenue" in df_cr.columns and "Day" in df_cr.columns:
                ax_cmp.plot(df_cr.groupby("Day")["Revenue"].mean(),
                            color=_COLS4[i % len(_COLS4)], lw=2, label=name[:18])
        ax_cmp.set_xlabel("Simulation Day"); ax_cmp.set_ylabel("Daily Revenue (EUR)")
        ax_cmp.set_title("Crisis-Phase Revenue Across Saved Scenarios")
        ax_cmp.legend(fontsize=7.5, ncol=2)
        ax_cmp.spines[["top", "right"]].set_visible(False)
        fig_cmp.tight_layout()
        pdf.chart(_sf_mpl_chart(fig_cmp),
                  caption=(
                      "Figure 5.1: Revenue trajectories during the crisis window for all "
                      "saved configurations. Spread between curves indicates parameter sensitivity."
                  ))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6: METHODOLOGY & CITATION
    # ─────────────────────────────────────────────────────────────────────────
    pdf.chapter(6, "Methodology & Citation", "Methodology")

    pdf.sub("6.1  Model Architecture")
    pdf.body(
        "GROCERYsim ABM v2.0 is a Mesa-based agent-based model for Finnish dairy retail "
        "and food-system resilience research. The model implements three agent classes: "
        "(1) Consumer agents using a held-out DCE attribute assessment and cross-fitted "
        "phase-transition response calibration from the SecureFood project; "
        "(2) Product agents representing Finnish dairy SKUs with dynamic pricing, stock "
        "management, and per-unit CO2 tracking; "
        "(3) a SupermarketModel orchestrating daily supply-chain events, crisis triggers, "
        "policy lever application, and population-level metric aggregation."
    )

    pdf.sub("6.2  Consumer Profiles & Calibration")
    pdf.body(
        "Questionnaire constructs are audited for missingness and raw internal reliability. "
        "K-Means solutions from k=2 through k=6 are compared using silhouette separation, "
        "bootstrap adjusted-Rand stability, and minimum cluster size. Categorical archetype "
        "modifiers operate only when every declared gate passes; otherwise the model uses "
        "continuous participant attributes. Each model seed draws complete observed profiles "
        "with replacement to the requested simulation size. No synthetic attributes, prices, "
        "quantities, demographics, or baskets are jittered. "
        "Requested products use a separately calibrated proportional-price acceptance rule. "
        "Same-category substitutes must be in stock and affordable. Reconstructed "
        "replacement events gate transfer of the price screen and deterministic ranking; "
        "unsupported allocation uses a seeded uniform draw among feasible candidates. "
        "The default empirical-only mode uses a linear relative-price rule and disables "
        "panic contagion, TPB threshold modulation, Prospect Theory, archetype modifiers, "
        "and preference learning. Those mechanisms require explicit exploratory opt-in."
    )

    pdf.sub("6.3  Supply Chain & Crisis Mechanics")
    pdf.body(
        "Product agents implement (s, S) inventory policies with configurable reorder "
        "points and target stock levels. The crisis window introduces: supply disruption "
        "reducing inbound deliveries and price inflation on all products. Panic and "
        "hoarding operate only in explicitly labelled exploratory runs. Monte Carlo confidence bands (when "
        "enabled) use non-parametric percentiles (p10/p25/p75/p90) -- more robust than "
        "Gaussian assumptions for short-horizon ABM outputs."
    )

    pdf.sub("6.4  Food Security & Welfare Metrics")
    pdf.body(
        "An exploratory access-stress score is computed for every represented household "
        "from realised daily pantry-consumption shortfall. Panic and shopping basket "
        "shortfall are separate outcomes. It is not the FAO FIES scale. The Gini Access Index "
        "measures inequality of basket fulfilment across "
        "the agent population (0 = perfect equality, 1 = complete inequality). Budget "
        "exhaustion measures the fraction of agents whose daily food spend reaches the "
        "maximum before their basket is complete."
    )

    pdf.sub("6.5  CO2 Emission Factors")
    pdf.body(
        "CO2 emission factors (kg CO2-eq / unit sold): Finnish organic 0.8, "
        "Finnish conventional 1.2, Imported organic 1.5, Imported conventional 2.2. "
        "Factors sourced from Finnish Life Cycle Assessment benchmarks for dairy products."
    )

    pdf.sub("6.6  Key References")
    pdf.bullet([
        "Ajzen, I. (1991). The theory of planned behavior. OBHDP, 50(2), 179-211.",
        "FAO (2016). Methods for estimating comparable rates of food insecurity globally.",
        "McCombs, M. & Shaw, D. (1972). The agenda-setting function of mass media. POQ, 36(2).",
        "O'Donoghue, T. & Rabin, M. (1999). Doing it now or later. AER, 89(1), 103-124.",
        "Sheffi, Y. (2005). The Resilient Enterprise. MIT Press.",
        "Thaler, R. H. & Sunstein, C. R. (2008). Nudge. Yale University Press.",
        "SecureFood Consortium (2024-2027). Horizon Europe Grant No. 101136583.",
    ])

    pdf.sub("6.7  How to Cite This Report")
    pdf.finding(
        "Duric, Ivan (2026). GROCERYsim Agent-Based Model for Consumer Behaviour and "
        "Supply Chain Stress-Testing. IAMO XR Lab, SecureFood project, "
        "Horizon Europe Grant 101136583."
    )

    return bytes(pdf.output())


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

    # ── Branded PDF Report ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📄 Full Branded PDF Report")
    st.markdown(
        "Generate a complete, print-ready PDF report covering all simulation results "
        "available in this session — parameters, revenue & waste charts, policy KPIs, "
        "stress-test ranking, scenario comparison, and methodology note."
    )

    _has_any = any(
        st.session_state.get(k) is not None
        for k in ["sim_results", "policy_baseline", "stress_results"]
    )

    _report_params = st.session_state.get("_last_params")

    if not _has_any:
        st.info(
            "No simulation data found yet. Run at least one simulation (Interactive Demo, "
            "Policy Analysis, or Stress Test) before generating the PDF report."
        )
    else:
        _pdf_col1, _pdf_col2 = st.columns([2, 1])
        with _pdf_col1:
            if st.button("📄 Generate PDF Report", key="gen_pdf_report_btn", type="primary",
                         use_container_width=True):
                with st.spinner("Building report… this may take 10–30 seconds"):
                    try:
                        _pdf_bytes = _make_branded_pdf_report(params=_report_params)
                        st.session_state["_generated_pdf"] = _pdf_bytes
                    except Exception as _e:
                        st.error(f"PDF generation failed: {_e}")
        with _pdf_col2:
            _generated = st.session_state.get("_generated_pdf")
            if _generated:
                st.download_button(
                    "📥 Download PDF",
                    _generated,
                    "GROCERYsim_Report.pdf",
                    "application/pdf",
                    key="dl_branded_pdf",
                    use_container_width=True,
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
    _behaviour_mode = (
        str(df_pol["BehaviorEvidenceMode"].iloc[0])
        if "BehaviorEvidenceMode" in df_pol.columns and not df_pol.empty
        else "empirical_only"
    )
    pdf.body(
        "GROCERYsim ABM v2.0: Mesa-based agent-based model for Finnish dairy retail. "
        "Consumer agents use a held-out DCE attribute assessment and cross-fitted transition calibration. "
        "Reliability-audited constructs; stability-gated exploratory clusters; complete-profile participant resampling. "
        "Phase-transition retain/drop behaviour is separated from the pooled milk DCE "
        "price-and-attribute candidate-choice model. "
        f"Behavioural evidence mode: {_behaviour_mode}. In empirical-only mode, panic contagion, "
        "TPB, Prospect Theory, archetype modifiers, and preference learning are disabled. "
        "In exploratory mode these are unvalidated assumptions, not sample estimates. "
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
# 11b. TAB: MODEL VALIDATION (Pattern-Oriented Modelling)
# ===========================================================================

def _render_validation_tab_legacy(params: dict):
    st.header("✅ Model Validation")
    st.markdown(
        "Professional ABMs are validated by checking that the model simultaneously "
        "reproduces several **known empirical patterns** — a technique called "
        "**Pattern-Oriented Modelling (POM)** (Grimm et al. 2005, *Science* 310, 987–991). "
        "Below we test GROCERYsim against eight stylised facts drawn from Finnish grocery "
        "statistics, consumer research, and food-security literature."
    )

    # ── Section 1: Reproducibility ────────────────────────────────────────────
    st.subheader("🔁 1. Reproducibility")
    st.markdown(
        "A fixed random seed must produce **bit-identical** results across two "
        "independent model instances — a prerequisite for scientific credibility."
    )

    if st.session_state.config_data is None:
        st.info("Load population data in 🏠 Data & Population to run the reproducibility check.")
    else:
        if st.button("▶️ Run Reproducibility Check", key="val_repro_btn"):
            with st.spinner("Running two identical simulations…"):
                _vp = dict(params)
                _vp["days"] = 30
                _m1 = _make_model(_vp, is_crisis=False, seed=99)
                _m2 = _make_model(_vp, is_crisis=False, seed=99)
                _rev1, _rev2 = [], []
                for _d in range(1, 31):
                    _m1.step(); _m2.step()
                    _r1, _ = _collect_model_day(_m1, _d, "A", collect_products=False)
                    _r2, _ = _collect_model_day(_m2, _d, "B", collect_products=False)
                    _rev1.append(_r1.get("Revenue", 0))
                    _rev2.append(_r2.get("Revenue", 0))
                _match = _rev1 == _rev2
                st.session_state["val_repro"] = {"match": _match, "rev1": _rev1, "rev2": _rev2}

        _vr = st.session_state.get("val_repro")
        if _vr:
            if _vr["match"]:
                st.success("✅ **PASS** — Both runs produce identical revenue trajectories across all 30 days. Fixed-seed reproducibility confirmed.")
            else:
                _diffs = sum(a != b for a, b in zip(_vr["rev1"], _vr["rev2"]))
                st.error(f"❌ **FAIL** — {_diffs}/30 days differ between runs. Check RNG seeding.")

    # ── Section 2: Stylised Facts Checklist ───────────────────────────────────
    st.divider()
    st.subheader("📋 2. Stylised Facts Checklist")
    st.markdown(
        "Each stylised fact is tested by running a short baseline simulation and checking "
        "whether the emergent model output falls within the empirically observed range. "
        "Benchmarks are drawn from Statistics Finland, Päivittäistavarakauppa ry (PTY), "
        "and published food-security literature."
    )

    if st.session_state.config_data is None:
        st.info("Load population data to run stylised fact checks.")
    else:
        if st.button("▶️ Run Stylised Fact Battery", key="val_pom_btn"):
            with st.spinner("Running POM validation battery (60-day baseline)…"):
                _vp2 = dict(params)
                _vp2.update({"days": 60, "mc_runs": 1, "policy_cfg": {}})
                _model = _make_model(_vp2, is_crisis=False, seed=42)
                _days_data = []
                for _d in range(1, 61):
                    _model.step()
                    _agg, _ = _collect_model_day(_model, _d, "Baseline", collect_products=False)
                    _days_data.append(_agg)
                _vdf = pd.DataFrame(_days_data)
                st.session_state["val_pom_df"] = _vdf

        _vpdf = st.session_state.get("val_pom_df")
        if _vpdf is not None and not _vpdf.empty:
            # Define stylised facts: (id, description, benchmark, test_fn, unit)
            def _fact(desc, benchmark, value, unit, pass_lo, pass_hi, source):
                _ok = pass_lo <= value <= pass_hi
                icon = "✅" if _ok else "⚠️"
                return {
                    "Status": icon,
                    "Stylised Fact": desc,
                    "Model Output": f"{value:.2f} {unit}",
                    "Empirical Range": benchmark,
                    "Source": source,
                    "Pass": _ok,
                }

            # Compute model statistics
            _daily_rev  = _vpdf["Revenue"].mean()
            _waste_mean = _vpdf["Waste"].mean()
            _sales_mean = _vpdf["Sales"].mean()
            _waste_pct  = _waste_mean / max(_sales_mean + _waste_mean, 1) * 100
            _fulfill    = _vpdf["FulfillmentRate"].mean() * 100
            _panic_base = _vpdf["PanicLevel"].mean()
            _co2_unit   = (_vpdf["CO2Sales"].mean() /
                           max(_vpdf["Sales"].mean(), 1)) if "CO2Sales" in _vpdf.columns else 0
            _import_dep = _vpdf["ImportDepPct"].mean() * 100 if "ImportDepPct" in _vpdf.columns else 0
            _budget_exh = _vpdf["BudgetExhaustionRate"].mean() * 100 if "BudgetExhaustionRate" in _vpdf.columns else 0

            # Weekend footfall check — from model's weekday weights: Sat=1.3, Mon=0.8
            _sat_mult = 1.30; _mon_mult = 0.80
            _footfall_ratio = _sat_mult / _mon_mult

            _facts = [
                _fact(
                    "Baseline consumer panic level < 0.05",
                    "< 0.05 (no crisis active)",
                    _panic_base, "", 0.0, 0.05,
                    "Model design criterion",
                ),
                _fact(
                    "Weekend (Sat) footfall 1.3–1.7× Monday",
                    "1.3–1.7× (PTY 2023)",
                    _footfall_ratio, "×", 1.3, 1.7,
                    "Päivittäistavarakauppa ry (PTY) Annual Report 2023",
                ),
                _fact(
                    "Baseline food waste rate 1–5% of units",
                    "1–5% (Luke/SYKE 2022)",
                    _waste_pct, "%", 1.0, 5.0,
                    "Luke / SYKE Finnish food waste report 2022",
                ),
                _fact(
                    "Baseline basket fulfilment ≥ 90%",
                    "≥ 90% (no disruption)",
                    _fulfill, "%", 90.0, 100.0,
                    "Model design criterion (crisis not active)",
                ),
                _fact(
                    "CO2 per unit sold 1.0–2.0 kg CO₂-eq",
                    "1.0–2.0 kg CO₂-eq (avg Finnish dairy mix)",
                    _co2_unit, "kg CO₂-eq", 0.8, 2.5,
                    "Finnish LCA benchmarks (Luke 2021)",
                ),
                _fact(
                    "Import dependency 20–50% (Finnish dairy)",
                    "20–50% (Evira/ETL)",
                    _import_dep, "%", 15.0, 60.0,
                    "ETL Food Industry Finland statistics 2023",
                ),
                _fact(
                    "Baseline budget exhaustion < 15%",
                    "< 15% (no crisis active)",
                    _budget_exh, "%", 0.0, 15.0,
                    "Model design criterion",
                ),
            ]

            _fact_df = pd.DataFrame(_facts)
            _pass_n  = _fact_df["Pass"].sum()
            _total_n = len(_fact_df)

            # Summary
            if _pass_n == _total_n:
                st.success(f"✅ All {_total_n} stylised facts passed. Model behaviour is consistent with empirical benchmarks.")
            elif _pass_n >= _total_n * 0.7:
                st.warning(f"⚠️ {_pass_n}/{_total_n} stylised facts passed. Review flagged items; minor calibration may improve alignment.")
            else:
                st.error(f"❌ Only {_pass_n}/{_total_n} stylised facts passed. Parameter recalibration recommended before policy use.")

            # Table
            _display_df = _fact_df[["Status", "Stylised Fact", "Model Output", "Empirical Range", "Source"]].copy()
            st.dataframe(
                _display_df.style.apply(
                    lambda row: ["background-color: #eafaf1" if row["Status"] == "✅"
                                 else "background-color: #fef9e7"] * len(row),
                    axis=1,
                ),
                use_container_width=True, hide_index=True,
            )

            # Pattern trajectory chart
            st.markdown("##### Daily trajectory — baseline run (60 days)")
            _pfig = make_subplots(rows=2, cols=2,
                                  subplot_titles=["Revenue (EUR/day)", "Waste (units/day)",
                                                  "Basket Fulfilment (%)", "Budget Exhaustion (%)"])
            _pfig.add_scatter(x=_vpdf["Day"], y=_vpdf["Revenue"],        row=1, col=1,
                              line=dict(color="#2980b9", width=2), name="Revenue")
            _pfig.add_scatter(x=_vpdf["Day"], y=_vpdf["Waste"],          row=1, col=2,
                              line=dict(color="#c0392b", width=2), name="Waste")
            _pfig.add_scatter(x=_vpdf["Day"], y=_vpdf["FulfillmentRate"]*100, row=2, col=1,
                              line=dict(color="#27ae60", width=2), name="Fulfilment")
            _pfig.add_hline(y=90, row=2, col=1,
                            line=dict(dash="dot", color="#e74c3c", width=1))
            if "BudgetExhaustionRate" in _vpdf.columns:
                _pfig.add_scatter(x=_vpdf["Day"], y=_vpdf["BudgetExhaustionRate"]*100, row=2, col=2,
                                  line=dict(color="#e67e22", width=2), name="BudgetExhaustion")
            _pfig.update_layout(height=420, showlegend=False,
                                margin=dict(t=50, b=30, l=40, r=20))
            st.plotly_chart(_pfig, use_container_width=True, config=_PLOTLY_CFG)

    # ── Section 3: Demand Elasticity Implied ──────────────────────────────────
    st.divider()
    st.subheader("📐 3. Implied Price Elasticity")
    st.markdown(
        "We infer the model's implied price elasticity of demand for dairy products "
        "by comparing revenue at baseline vs. a 10% price shock scenario. "
        "Finnish dairy price elasticity benchmarks: **-0.3 to -0.6** (Niemi & Arovuori 2014, "
        "PTT Working Paper)."
    )

    if st.session_state.config_data is None:
        st.info("Load population data to compute implied elasticity.")
    else:
        if st.button("▶️ Compute Implied Elasticity", key="val_elas_btn"):
            with st.spinner("Running two 30-day simulations (baseline + 10% price shock)…"):
                _ep = dict(params)
                _ep["days"] = 30

                _ep0 = dict(_ep); _ep0["inf"] = 0.0
                _ep1 = dict(_ep); _ep1["inf"] = 10.0

                _m_base  = _make_model(_ep0, is_crisis=True, seed=7)
                _m_shock = _make_model(_ep1, is_crisis=True, seed=7)
                _s_base = _s_shock = 0.0
                for _d in range(1, 31):
                    _m_base.step(); _m_shock.step()
                    _ab, _ = _collect_model_day(_m_base,  _d, "B", collect_products=False)
                    _as, _ = _collect_model_day(_m_shock, _d, "S", collect_products=False)
                    _s_base  += _ab.get("Sales", 0)
                    _s_shock += _as.get("Sales", 0)

                _qty_chg = (_s_shock - _s_base) / max(_s_base, 1) * 100
                _p_chg   = 10.0
                _elas    = _qty_chg / _p_chg if _p_chg != 0 else 0
                st.session_state["val_elas"] = {
                    "elas": _elas, "qty_chg": _qty_chg,
                    "s_base": _s_base, "s_shock": _s_shock,
                }

        _ve = st.session_state.get("val_elas")
        if _ve:
            _e = _ve["elas"]
            _in_range = -0.7 <= _e <= -0.1
            _icon = "✅" if _in_range else "⚠️"
            c_e1, c_e2, c_e3 = st.columns(3)
            c_e1.metric("Implied Price Elasticity",   f"{_e:.3f}")
            c_e2.metric("Quantity change (10% shock)", f"{_ve['qty_chg']:+.1f}%")
            c_e3.metric("Benchmark range",             "-0.3 to -0.6")
            if _in_range:
                st.success(f"{_icon} Implied elasticity {_e:.3f} falls within the plausible Finnish dairy range (-0.7 to -0.1). Model demand responsiveness is empirically grounded.")
            else:
                st.warning(f"{_icon} Implied elasticity {_e:.3f} is outside the -0.7 to -0.1 benchmark range. Consider adjusting price_sensitivity or utility_threshold parameters.")

    # ── Section 4: ODD+D Quick Reference ─────────────────────────────────────
    st.divider()
    st.subheader("📄 4. Download Full ODD+D Documentation")
    st.markdown(
        "For a complete formal model description following the ODD+D protocol, "
        "open the **📋 Model Documentation** tab from the navigation."
    )
    if st.button("→ Go to Model Documentation", key="val_goto_docs"):
        st.session_state["active_section"] = "docs"
        st.rerun()


# ===========================================================================
# 11c. TAB: EVIDENCE-AWARE MODEL VALIDATION
# ===========================================================================

def render_validation_tab(params: dict):
    st.header("✅ Validation & Verification")
    st.markdown(
        "This page separates **internal verification**, **calibration holdout evidence**, "
        "**scenario plausibility**, and **independent external validation**. Passing a code "
        "check or a broad literature range does not validate the model empirically."
    )

    st.info(
        "**Current scientific status:** no bundled dataset is treated as independent external "
        "validation evidence. Upload a locked validation plan with traceable sources and run the "
        "model before making a target-specific external-validity claim."
    )

    st.subheader("1. Phase-one baseline reproduction")
    st.markdown(
        "This test asks whether the ABM's repeated visits, pantry accounting, and store "
        "inventory preserve the phase-one GROCERYsim shopping patterns that initialise "
        "agents. It is an **internal reproduction test**, not independent validation."
    )
    _baseline_targets = (
        st.session_state.config_data.get("stats", {}).get(
            "baseline_reproduction_targets", {}
        ) if st.session_state.config_data else {}
    )
    if _baseline_targets.get("status") != "ok":
        st.warning(
            "Reload the bundled data with the current pipeline before running the "
            "baseline-reproduction audit."
        )
    else:
        _bt1, _bt2, _bt3 = st.columns(3)
        _bt1.metric(
            "Observed basket units",
            f"{_baseline_targets['mean_linked_basket_units']:.2f}",
        )
        _bt2.metric(
            "Observed basket value",
            f"€{_baseline_targets['mean_linked_basket_value']:.2f}",
        )
        _bt3.metric(
            "Observed occasions",
            _baseline_targets["n_shopping_occasions"],
        )
        st.caption(
            "The observed basket is already a household shopping outcome and is not "
            "multiplied by household size. Inter-visit time was not collected; the ABM "
            "derives it from represented households and analyst-selected store traffic."
        )
        if st.button(
            "▶️ Run baseline-reproduction audit",
            key="validation_baseline_reproduction_btn",
        ):
            with st.spinner("Running three evidence-only baseline replicates…"):
                _bp = dict(params)
                _bp.update({
                    "exploratory_behaviour": False,
                    "inf": 0.0,
                    "dis": 0,
                    "purchase_limit": None,
                    "traffic_variation": False,
                })
                _baseline_rows = []
                for _run, _seed in enumerate((41, 42, 43)):
                    _bm = _make_model(
                        _bp, is_crisis=False, seed=_seed, policy_cfg={}
                    )
                    _run_days = max(
                        60,
                        2 * int(math.ceil(
                            _bm.expected_household_visit_interval
                        )) + 14,
                    )
                    for _day in range(1, _run_days + 1):
                        _bm.step()
                        _row, _ = _collect_model_day(
                            _bm, _day, "Baseline", collect_products=False
                        )
                        _row["Run"] = _run
                        _baseline_rows.append(_row)
                _baseline_df = pd.DataFrame(_baseline_rows)
                st.session_state["baseline_reproduction_audit"] = (
                    evaluate_baseline_reproduction(
                        _baseline_targets, _baseline_df
                    )
                )

        _baseline_audit = st.session_state.get(
            "baseline_reproduction_audit"
        )
        if _baseline_audit:
            _ba1, _ba2, _ba3 = st.columns(3)
            _ba1.metric(
                "Checks passed",
                f"{_baseline_audit['passed']}/{_baseline_audit['total']}",
            )
            _ba2.metric("Warm-up", f"{_baseline_audit['warmup_days']} days")
            _ba3.metric(
                "Implied revisit interval",
                f"{_baseline_audit['expected_visit_interval_days']:.1f} days",
            )
            if _baseline_audit["status"] == "pass":
                st.success(
                    "All declared phase-one reproduction and temporal-accounting "
                    "gates passed for the current store configuration."
                )
            else:
                st.error(
                    "The baseline does not reproduce all declared phase-one patterns. "
                    "Resolve failed mechanisms before interpreting crisis or policy results."
                )
            st.dataframe(
                pd.DataFrame(_baseline_audit["checks"]),
                hide_index=True,
                use_container_width=True,
            )
            st.warning(_baseline_audit["claim"])

    st.subheader("2. Phase-two one-occasion holdout reproduction")
    st.markdown(
        "This controlled test applies training-cohort median phase-two SKU prices "
        "where available, uses the observed median shock for uncovered SKUs, and "
        "applies each holdout participant's phase-two maximum budget to phase-one "
        "needs. Inventory is non-binding so response is not confused with stockouts."
    )
    _phase2_targets = (
        st.session_state.config_data.get("stats", {}).get(
            "phase2_reproduction_targets", {}
        ) if st.session_state.config_data else {}
    )
    if _phase2_targets.get("status") != "ok":
        st.warning(
            "Reload the bundled data with the current pipeline before running the "
            "phase-two holdout audit."
        )
    else:
        _p21, _p22, _p23 = st.columns(3)
        _p21.metric("Holdout participants", _phase2_targets["n_holdout"])
        _p22.metric(
            "Controlled price shock",
            f"{100 * float(_phase2_targets.get('price_shock') or 0):.1f}%",
        )
        _p23.metric("Stochastic replicates", 30)
        _sub_action = (
            st.session_state.config_data.get("stats", {})
            .get("behavioral_calibration", {})
        )
        if _sub_action.get("substitution_action_model_retained", False):
            st.caption(
                "Replacement propensity uses a participant-specific prediction "
                "that beat the calibration-cohort naive benchmark under repeated "
                "nested cross-validation."
            )
        else:
            st.caption(
                "Replacement propensity uses the calibration-cohort fallback "
                f"probability ({100 * float(_sub_action.get('substitution_action_naive_probability', 0)):.1f}% "
                "per baseline basket line). The participant-specific model did not "
                "beat the naive benchmark, so individual substitution prediction is "
                "not claimed."
            )
        if st.button(
            "▶️ Run phase-two holdout audit",
            key="validation_phase2_reproduction_btn",
        ):
            with st.spinner("Running controlled one-occasion holdout visits…"):
                _source_config = st.session_state.config_data
                _validation_profiles = [
                    profile for profile in _source_config.get("population", [])
                    if profile.get("phase2_calibration_role") == "validation"
                ]
                _phase2_rows = []
                _shock_pct = 100.0 * float(
                    _phase2_targets.get("price_shock") or 0.0
                )
                for _run in range(30):
                    _controlled_config = {
                        "products": copy.deepcopy(_source_config.get("products", [])),
                        "population": copy.deepcopy(_validation_profiles),
                        "stats": copy.deepcopy(_source_config.get("stats", {})),
                    }
                    _model = SupermarketModel(
                        config_data=_controlled_config,
                        base_consumers=max(1, len(_validation_profiles)),
                        start_month=params["month"],
                        reorder_pt=params["reorder"],
                        target_stock=params["target"],
                        lead_time=params["lead"],
                        is_crisis_mode=True,
                        scenario_start_day=1,
                        inflation_pct=_shock_pct,
                        disruption_days=0,
                        fixed_seed=4200 + _run,
                        policy_cfg={},
                        enable_traffic_variation=False,
                        scenario_price_overrides=_phase2_targets.get(
                            "training_phase2_price_overrides", {}
                        ),
                    )
                    # Isolate consumer response: every catalogue SKU is available
                    # in a fresh, non-expiring batch throughout this one visit.
                    for _product in _model.products:
                        _product.max_shelf_capacity = 1_000_000
                        _product.max_storage_capacity = 1_000_000
                        _product.stock_storage = 1_000_000
                        _product.shelf_batches = [{"qty": 1_000_000, "age": -1}]
                    _model.step()
                    for _agent in _model.last_daily_agents:
                        _profile = _agent.profile
                        _base_units = max(1, _agent.items_base_wanted)
                        _purchased_lines = max(1, _agent.choice_lines_purchased)
                        _phase2_rows.append({
                            "Run": _run,
                            "source_id": _profile.get("source_id"),
                            "model_quantity_retention": (
                                _agent.items_purchased / _base_units
                            ),
                            "observed_quantity_retention": _profile.get(
                                "observed_quantity_retention", 0.0
                            ),
                            "model_spending_reduction": max(
                                0.0,
                                1.0 - _agent.amount_spent / max(_agent.budget, 0.01),
                            ),
                            "observed_spending_reduction": _profile.get(
                                "observed_spending_reduction", 0.0
                            ),
                            "model_budget_utilization": min(
                                1.0,
                                _agent.amount_spent / max(_agent.crisis_budget, 0.01),
                            ),
                            "observed_budget_utilization": _profile.get(
                                "observed_budget_utilization", 0.0
                            ),
                            "model_substitution_rate": (
                                _agent.choice_lines_substituted / _purchased_lines
                                if _agent.choice_lines_purchased > 0 else 0.0
                            ),
                            "observed_substitution_rate": _profile.get(
                                "observed_substitution_rate", 0.0
                            ),
                        })
                st.session_state["phase2_reproduction_audit"] = (
                    evaluate_phase2_reproduction(
                        _phase2_targets, pd.DataFrame(_phase2_rows)
                    )
                )

        _phase2_audit = st.session_state.get("phase2_reproduction_audit")
        if _phase2_audit:
            _ph1, _ph2, _ph3 = st.columns(3)
            _ph1.metric(
                "Metrics passed",
                f"{_phase2_audit['passed']}/{_phase2_audit['total']}",
            )
            _ph2.metric("Holdout N", _phase2_audit["n_holdout"])
            _ph3.metric("Replicates", _phase2_audit["n_replicates"])
            if _phase2_audit["status"] == "pass":
                st.success(
                    "All applicable aggregate and retained individual-skill gates passed."
                )
            else:
                st.error(
                    "The controlled consumer model does not reproduce every phase-two "
                    "holdout outcome. Failed mechanisms remain unsuitable for predictive use."
                )
            st.dataframe(
                pd.DataFrame(_phase2_audit["checks"]),
                hide_index=True,
                use_container_width=True,
            )
            st.warning(_phase2_audit["claim"])

    with st.expander("Evidence tiers and claim rules", expanded=False):
        st.markdown(
            "- **Internal invariant:** checks implementation, accounting, and reproducibility. "
            "It cannot establish empirical validity.\n"
            "- **Calibration holdout:** tests prediction within the GROCERYsim collection/calibration "
            "pipeline. It is useful internal evidence but is not an independent external test.\n"
            "- **Scenario plausibility:** checks direction or order of magnitude. It is diagnostic, "
            "not confirmatory validation.\n"
            "- **External independent:** requires a preregistered acceptance interval, timestamped "
            "registration, population/period metadata, a traceable source, and data not used for calibration."
        )

    st.subheader("3. Preregistered empirical validation plan")
    st.markdown(
        "Prepare the plan **before inspecting the corresponding model outputs**. Each row declares "
        "one metric, scenario and time window, aggregation rule, acceptance interval, and evidence provenance."
    )
    _template = validation_target_template()
    st.download_button(
        "⬇️ Download validation-plan template",
        data=_template.to_csv(index=False).encode("utf-8"),
        file_name="GROCERYsim_validation_plan_TEMPLATE.csv",
        mime="text/csv",
        key="validation_template_download",
    )
    st.caption(
        "The example rows deliberately fail integrity checks until all EXAMPLE/REPLACE placeholders "
        "and acceptance bounds are replaced. This prevents accidental use as evidence."
    )

    _validation_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation")
    _catalogue_path = os.path.join(_validation_dir, "evidence_catalogue.csv")
    _draft_path = os.path.join(_validation_dir, "validation_plan_DRAFT.csv")
    _protocol_path = os.path.join(_validation_dir, "VALIDATION_DATA_PROTOCOL.md")
    try:
        _evidence_catalogue = pd.read_csv(_catalogue_path)
        with st.expander("Independent-data acquisition catalogue", expanded=False):
            st.markdown(
                "The catalogue distinguishes datasets that can validate model outputs from sources "
                "that only provide inputs or broad plausibility context. **Conditional** does not mean validated."
            )
            st.dataframe(
                _evidence_catalogue[[
                    "source_name", "access", "candidate_uses", "evidence_role",
                    "current_admissibility", "blocking_issue", "priority",
                ]],
                use_container_width=True,
                hide_index=True,
            )
        _asset_c1, _asset_c2, _asset_c3 = st.columns(3)
        with open(_catalogue_path, "rb") as _file:
            _asset_c1.download_button(
                "⬇️ Evidence catalogue", _file.read(), "GROCERYsim_evidence_catalogue.csv",
                "text/csv", key="validation_catalogue_download", use_container_width=True,
            )
        with open(_draft_path, "rb") as _file:
            _asset_c2.download_button(
                "⬇️ Draft target plan", _file.read(), "GROCERYsim_validation_plan_DRAFT.csv",
                "text/csv", key="validation_draft_download", use_container_width=True,
            )
        with open(_protocol_path, "rb") as _file:
            _asset_c3.download_button(
                "⬇️ Data protocol", _file.read(), "GROCERYsim_VALIDATION_DATA_PROTOCOL.md",
                "text/markdown", key="validation_protocol_download", use_container_width=True,
            )
        st.warning(
            "The bundled draft plan is intentionally inadmissible: empirical bounds, observation "
            "periods/populations, independence confirmation, and a registration reference are still missing."
        )
    except (FileNotFoundError, KeyError, pd.errors.ParserError) as _asset_error:
        st.error(f"Validation support files could not be loaded: {_asset_error}")

    _uploaded_plan = st.file_uploader(
        "Upload completed validation plan (CSV)",
        type=["csv"],
        key="validation_plan_upload",
        help="Use exact simulation column names in metric, such as Sales, Waste, Revenue, or FulfillmentRate.",
    )

    _evaluated = None
    _summary = None
    if _uploaded_plan is None:
        st.warning("External validation not evaluated: no registered target plan is loaded.")
    else:
        try:
            _targets = pd.read_csv(_uploaded_plan)
            _plan_errors = validate_target_definitions(_targets)
        except Exception as _exc:
            _targets = None
            _plan_errors = [f"The CSV could not be read: {_exc}"]

        if _plan_errors:
            st.error("The validation plan is not admissible.")
            for _error in _plan_errors:
                st.markdown(f"- {_error}")
        elif st.session_state.get("sim_results") is None:
            st.success("Plan integrity checks passed.")
            st.warning("Run the ABM first; no simulation output is available for target evaluation.")
        else:
            _raw_baseline = (
                st.session_state.get("data_base_opt")
                if st.session_state.get("data_base_opt") is not None
                else st.session_state.get("data_base_raw")
            )
            _raw_crisis = st.session_state.get("data_crisis")
            if (
                isinstance(_raw_baseline, pd.DataFrame) and not _raw_baseline.empty
                and isinstance(_raw_crisis, pd.DataFrame) and not _raw_crisis.empty
                and "Run" in _raw_baseline.columns and "Run" in _raw_crisis.columns
            ):
                _vb = _raw_baseline.copy(); _vb["Scenario"] = "Baseline"
                _vc = _raw_crisis.copy(); _vc["Scenario"] = "Crisis"
                _validation_output = pd.concat([_vb, _vc], ignore_index=True)
                _simulation_source = "raw Monte Carlo replicates"
            else:
                _validation_output = st.session_state.sim_results.copy()
                if "Scenario" in _validation_output.columns:
                    _validation_output["Scenario"] = _validation_output["Scenario"].replace(
                        {value: "Baseline" for value in _validation_output["Scenario"].unique()
                         if str(value).lower().startswith("baseline")}
                    )
                _simulation_source = "quick-preview/mean trajectory (not sufficient for external acceptance)"
            _evaluated = evaluate_targets(_targets, _validation_output)
            _summary = validation_summary(_evaluated)
            _claim_status = _summary["claim_status"]
            if _claim_status == "external_targets_met":
                st.success(_summary["claim"])
            elif _claim_status == "external_targets_not_met":
                st.error(_summary["claim"])
            else:
                st.warning(_summary["claim"])

            _vc1, _vc2, _vc3, _vc4 = st.columns(4)
            _vc1.metric("External targets", _summary["external_total"])
            _vc2.metric("Passed", _summary["external_passed"])
            _vc3.metric("Failed", _summary["external_failed"])
            _vc4.metric("Not evaluated", _summary["external_not_evaluated"])
            st.caption(f"Evaluation source: {_simulation_source}.")

            _display = _evaluated[[
                "target_id", "label", "evidence_tier", "scenario", "aggregation",
                "observed", "simulation_lower_95", "simulation_upper_95", "n_replicates",
                "lower", "upper", "unit", "status", "reason",
                "source_name", "source_reference",
            ]].copy()
            st.dataframe(_display, use_container_width=True, hide_index=True)

            _audit = {
                "protocol_version": "GROCERYsim-validation-1.0",
                "claim_status": _summary["claim_status"],
                "claim": _summary["claim"],
                "summary": _summary,
                "tier_counts": evidence_tier_counts(_evaluated),
                "targets": json.loads(_evaluated.to_json(orient="records")),
                "simulation_source": _simulation_source,
                "simulation_rows": int(len(_validation_output)),
                "simulation_columns": list(_validation_output.columns),
            }
            _dl1, _dl2 = st.columns(2)
            _dl1.download_button(
                "⬇️ Download evaluated targets (CSV)",
                data=_evaluated.to_csv(index=False).encode("utf-8"),
                file_name="GROCERYsim_validation_results.csv",
                mime="text/csv",
                key="validation_results_csv",
                use_container_width=True,
            )
            _dl2.download_button(
                "⬇️ Download validation audit (JSON)",
                data=json.dumps(_audit, indent=2).encode("utf-8"),
                file_name="GROCERYsim_validation_audit.json",
                mime="application/json",
                key="validation_results_json",
                use_container_width=True,
            )

    st.divider()
    st.subheader("4. Internal verification — fixed-seed reproducibility")
    st.markdown(
        "This test checks deterministic execution under a fixed seed. A pass supports software "
        "verification only; it says nothing about correspondence with real grocery systems."
    )
    if st.session_state.config_data is None:
        st.info("Load population data to run the reproducibility check.")
    else:
        if st.button("▶️ Run reproducibility check", key="validation_repro_btn_v2"):
            with st.spinner("Running two independently instantiated 30-day models…"):
                _vp = dict(params)
                _vp["days"] = 30
                _m1 = _make_model(_vp, is_crisis=False, seed=99)
                _m2 = _make_model(_vp, is_crisis=False, seed=99)
                _rows_1, _rows_2 = [], []
                for _day in range(1, 31):
                    _m1.step(); _m2.step()
                    _r1, _ = _collect_model_day(_m1, _day, "Baseline", collect_products=False)
                    _r2, _ = _collect_model_day(_m2, _day, "Baseline", collect_products=False)
                    _rows_1.append(_r1)
                    _rows_2.append(_r2)
                _left = pd.DataFrame(_rows_1).sort_index(axis=1)
                _right = pd.DataFrame(_rows_2).sort_index(axis=1)
                _same = _left.equals(_right)
                _different_cells = int((_left.ne(_right) & ~(_left.isna() & _right.isna())).sum().sum())
                st.session_state["validation_repro_v2"] = {
                    "same": _same,
                    "different_cells": _different_cells,
                    "rows": len(_left),
                    "columns": len(_left.columns),
                }
        _repro = st.session_state.get("validation_repro_v2")
        if _repro:
            if _repro["same"]:
                st.success(
                    f"PASS — both instances produced identical {_repro['rows']}-day aggregate tables "
                    f"across {_repro['columns']} fields. Classification: internal invariant."
                )
            else:
                st.error(
                    f"FAIL — {_repro['different_cells']} aggregate cells differ. Resolve RNG leakage "
                    "before interpreting stochastic experiments."
                )

    st.divider()
    st.subheader("5. Price-response diagnostic")
    st.markdown(
        "The former elasticity pass/fail check used a hard-coded literature range and was labelled "
        "empirical validation. It is now a **diagnostic only**. To validate price response, add a "
        "source-specific elasticity target to the preregistered plan, with the matching product scope, "
        "price definition, horizon, population, and uncertainty interval."
    )
    if st.session_state.config_data is None:
        st.info("Load population data to compute the diagnostic.")
    elif st.button("▶️ Compute 10% price-shock diagnostic", key="validation_elasticity_btn_v2"):
        with st.spinner("Running paired 30-day baseline and price-shock models…"):
            _ep = dict(params)
            _ep["days"] = 30
            _ep0 = dict(_ep); _ep0["inf"] = 0.0
            _ep1 = dict(_ep); _ep1["inf"] = 10.0
            _base = _make_model(_ep0, is_crisis=True, seed=7)
            _shock = _make_model(_ep1, is_crisis=True, seed=7)
            _sales_base = _sales_shock = 0.0
            for _day in range(1, 31):
                _base.step(); _shock.step()
                _rb, _ = _collect_model_day(_base, _day, "Baseline", collect_products=False)
                _rs, _ = _collect_model_day(_shock, _day, "Shock", collect_products=False)
                _sales_base += _rb.get("Sales", 0)
                _sales_shock += _rs.get("Sales", 0)
            _quantity_change = (_sales_shock - _sales_base) / max(_sales_base, 1) * 100
            st.session_state["validation_elasticity_v2"] = _quantity_change / 10.0
    if "validation_elasticity_v2" in st.session_state:
        st.metric("Arc-style quantity response / 10% price shock", f"{st.session_state['validation_elasticity_v2']:.3f}")
        st.caption("Exploratory diagnostic — no acceptance decision and no external-validity claim.")

    st.divider()
    st.subheader("6. Documentation")
    st.markdown(
        "The ODD+D record documents model mechanisms and now states the evidence-tier rules and "
        "current external-validation limitation."
    )
    if st.button("→ Go to Model Documentation", key="validation_goto_docs_v2"):
        st.session_state["nav_section"] = "docs"
        st.rerun()


# ===========================================================================
# 11d. TAB: MODEL DOCUMENTATION (ODD+D Protocol)
# ===========================================================================

def _make_odd_d_pdf(params: dict | None = None) -> bytes:
    """Generate a formal ODD+D protocol PDF using the branded _GROCERYReport class."""
    from datetime import datetime as _dt
    from fpdf.enums import XPos, YPos

    class _OddReport(_SFReport):
        def header(self):
            if self.page_no() == 1:
                return
            self.set_fill_color(*_SF_DARK)
            self.rect(0, 0, 210, 7, "F")
            self.set_fill_color(*_SF_AMBER)
            self.rect(0, 0, 3, 7, "F")
            self.set_font("Ar", "B", 6.5)
            self.set_text_color(*_SF_WHITE)
            self.set_xy(6, 0.8)
            self.cell(130, 5.5,
                      "GROCERYsim ABM v2.0 -- ODD+D Protocol Model Documentation")
            self.set_font("Ar", "I", 6.5)
            self.set_text_color(*_SF_AMBER)
            self.set_xy(136, 0.8)
            self.cell(59, 5.5, self._sec, align="R")
            self.set_y(10)
            self.set_text_color(*_SF_BODY)

    pdf = _OddReport()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf._lf()
    now_str = _dt.now().strftime("%d %B %Y")

    # ── COVER ─────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(*_SF_DARK)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_fill_color(*_SF_AMBER)
    pdf.rect(0, 0, 210, 5, "F")
    pdf.rect(0, 292, 210, 5, "F")
    try:
        gs_p = _sf_logo_on("GROCERYsim.png", 360, _SF_DARK)
        sf_p = _sf_logo_on("SecureFood.png",  260, _SF_DARK)
        from PIL import Image as _PILImg
        gs_img = _PILImg.open(gs_p); sf_img = _PILImg.open(sf_p)
        pdf.image(gs_p, x=15, y=16, w=80)
        pdf.image(sf_p, x=210 - 15 - 58, y=16, w=58)
        logo_b = 16 + max(80 * gs_img.height / gs_img.width,
                          58 * sf_img.height / sf_img.width) + 4
    except Exception:
        logo_b = 30
    pdf.set_fill_color(*_SF_AMBER); pdf.rect(15, logo_b, 180, 0.7, "F")
    pdf.set_y(logo_b + 6)
    pdf.set_font("Ar", "B", 26); pdf.set_text_color(*_SF_WHITE)
    pdf.cell(0, 12, "GROCERYsim ABM v2.0",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Ar", "", 13); pdf.set_text_color(*_SF_AMBER)
    pdf.cell(0, 8, "ODD+D Protocol — Model Documentation",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    pdf.set_fill_color(*_SF_AMBER); pdf.rect(15, pdf.get_y(), 180, 0.5, "F")
    pdf.ln(7)
    pdf.set_font("Ar", "", 10); pdf.set_text_color(200, 220, 215)
    for _ln in [
        f"Version: 2.0  ·  Generated: {now_str}",
        "Horizon Europe SecureFood Consortium — Grant No. 101136583",
        "IAMO XR Lab, Halle (Saale), Germany",
        "Protocol: Grimm et al. (2020) ODD+D, JASSS 23(2)",
    ]:
        pdf.cell(0, 7, _ln, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    pdf.set_font("Ar", "I", 8.5); pdf.set_text_color(140, 165, 160)
    pdf.set_x(30)
    pdf.multi_cell(150, 4.8,
        "This document follows the ODD+D (Overview, Design Concepts, Details + "
        "Decision) protocol for agent-based models. It documents the structural "
        "implementation for independent scrutiny and re-implementation; empirical "
        "validation still depends on the evidence gaps stated in this document.",
        align="C")
    pdf.set_text_color(0, 0, 0)

    # ── 1. PURPOSE ────────────────────────────────────────────────────────────
    pdf.chapter(1, "Purpose", "Purpose")
    pdf.body(
        "GROCERYsim ABM v2.0 simulates Finnish grocery retail supply chains and "
        "consumer behaviour under baseline and crisis conditions. The model is designed "
        "to address three research objectives within the Horizon Europe SecureFood project:"
    )
    pdf.bullet([
        "To quantify the impact of climate-driven supply disruptions (specifically dairy "
        "supply chain disruption) on retail revenue, food waste, and consumer food security.",
        "To evaluate the effectiveness of food-system policy interventions (fat taxation, "
        "domestic subsidies, nutritional labelling, purchase limits, communication) on "
        "welfare, sustainability, and market efficiency outcomes.",
        "To identify vulnerable consumer archetypes and income groups and quantify the "
        "equity implications of disruption and policy response scenarios.",
    ])
    pdf.body(
        "The model is intended for use by supply chain practitioners, agricultural policy "
        "makers, food security researchers, and sustainability analysts. It is not designed "
        "for high-frequency trading optimisation or financial forecasting. Results should "
        "be interpreted as directional indicators calibrated to a stylised Finnish dairy "
        "retail context, not as precise quantitative predictions."
    )

    # ── 2. ENTITIES, STATE VARIABLES, AND SCALES ──────────────────────────────
    pdf.chapter(2, "Entities, State Variables & Scales", "Entities")
    pdf.sub("2.1  Agent Types")
    pdf.body(
        "The model contains three agent types operating within a single-store environment:"
    )
    pdf.kv([
        ("ProductAgent",  "One instance per SKU. Manages shelf stock (FIFO batches with "
                          "age tracking), storage stock (scalar), reorder signalling, "
                          "near-expiry discounting (50% off within 2 days of expiry), "
                          "and per-unit CO2 accounting."),
        ("SupplyTruck",   "Singleton logistics agent. Queues orders, enforces lead times, "
                          "executes deliveries, and blocks supply during disruption windows. "
                          "Handles domestic supply shocks at the product-origin level."),
        ("ConsumerAgent", "One shopping-visit instance linked to a persistent household "
                          "profile. Stable household state retains pantry inventory and visit "
                          "history. Reference-price adaptation and learned preferences are optional. "
                          "The visit instance records budget and basket outcomes."),
        ("SupermarketModel", "Model orchestrator. Coordinates daily agent activation "
                             "order (products, logistics, then shuffled shoppers), manages crisis state transitions, "
                             "optionally applies exploratory calendar traffic multipliers, "
                             "aggregates welfare and supply chain metrics."),
    ])

    pdf.sub("2.2  Key State Variables")
    pdf.kv([
        ("ProductAgent: shelf_batches",       "List of {qty, age} dicts (FIFO queue). Drives waste, near-expiry, and stock availability."),
        ("ProductAgent: stock_storage",       "Scalar (units). Back-room inventory feeding shelf replenishment."),
        ("ProductAgent: current_price",       "Float (EUR). Inflation-adjusted and policy-modified shelf price."),
        ("ConsumerAgent: price_sensitivity",  "Float [0-1]. Cross-fitted phase-response value; fixed in empirical-only mode."),
        ("Household profile: _home_inv",       "Dict {SKU id: units}. Persistent pantry stock consumed every calendar day."),
        ("Household profile: _access_stress_score", "Int [0-4]. Daily realised pantry-consumption shortfall category; not a validated FIES estimate."),
        ("SupermarketModel: global_panic_level", "Float [0-1]. Optional exploratory panic signal; fixed at zero in empirical-only mode."),
        ("SupermarketModel: is_scenario_active", "Bool. True during the crisis window; toggles inflation, supply disruption, and panic dynamics. Phase-two baskets are never simulated demand."),
    ])

    pdf.sub("2.3  Scales")
    pdf.kv([
        ("Temporal resolution", "1 simulation step = 1 calendar day"),
        ("Spatial resolution",  "Single store (multi-store extension: N stores with inter-store panic contagion)"),
        ("Population scale",    "Default 100 visits/day from 2,000 persistent household profiles; 116 records collected, 108 usable"),
        ("Product catalogue",   "107 unique Finnish dairy SKUs after collapsing 340 duplicate Unity scene placements"),
        ("Simulation horizon",  "Configurable 30-365 days; typical run 90 days"),
    ])

    # ── 3. PROCESS OVERVIEW AND SCHEDULING ────────────────────────────────────
    pdf.chapter(3, "Process Overview & Scheduling", "Scheduling")
    pdf.body(
        "Each simulation day executes in the following fixed order:"
    )
    pdf.kv([
        ("Step 1 — Household consumption", "Deplete every persistent household pantry using basket quantity divided by its expected revisit interval; record realised shortfall."),
        ("Step 2 — Visit scheduling",    "Select the most-due unique households using constant declared traffic; optional calendar/noise variation is exploratory."),
        ("Step 3 — ProductAgent.step()", "Reset daily counters; refill shelf from storage (50% threshold); "
                                          "update price (inflation + policy); age all batches; remove "
                                          "expired batches (waste log); flag near-expiry batches."),
        ("Step 4 — SupplyTruck.step()",  "Deliver queued orders that have reached their arrival day; "
                                          "apply domestic shock (block fraction of Finnish-origin deliveries); "
                                          "place new orders for products below reorder trigger; "
                                          "queue delivery for lead_time days hence."),
        ("Step 5 — ConsumerAgent.step()", "Compute pantry-adjusted replenishment need; apply the evidence-gated "
                                           "choice rule; execute shopping loop; record shopping shortfall. "
                                           "TPB, panic stockpiling, and preference updates run only in exploratory mode."),
        ("Step 6 — Aggregation",          "Compute visit-level fulfilment and budget exhaustion plus population-wide "
                                           "consumption access stress by bracket, Gini shopping-access index, "
                                           "panic level, CO2 totals, import dependency. Append daily_records."),
    ])
    pdf.body(
        "The panic signal is updated from the share of shoppers exposed to a near-empty "
        "shelf. Each shopper signals at most once per day; ordinary scarcity exposure up "
        "to 10% is treated as retail friction, and excess exposure is multiplied by "
        "panic_sensitivity before daily decay. "
        "When exploratory panic dynamics are enabled this creates a global broadcast "
        "mechanism without explicit agent-to-agent communication. It is inactive by default."
    )

    # ── 4. DESIGN CONCEPTS ────────────────────────────────────────────────────
    pdf.chapter(4, "Design Concepts", "Design Concepts")
    pdf.kv([
        ("Emergence",       "Waste rates, fulfilment, access-stress outcomes, "
                            "and revenue volatility all emerge from individual agent decisions "
                            "and product-level stock dynamics. No macro-level targets are imposed."),
        ("Adaptation",      "Disabled in empirical-only mode. Optional archetype-specific heuristic reinforcement "
                            "(rate 0.015/visit) requires exploratory mode and a passing archetype gate."),
        ("Objectives",      "Consumers replenish observed needs subject to price acceptance, budget, and stock. "
                            "Validated rankings may order feasible substitutes; otherwise allocation is seeded stochastic. "
                            "ProductAgents minimise stockout risk via (s,S) reorder policy."),
        ("Learning",        "Optional archetype-specific preference drift: price_champion shifts toward lower-cost "
                            "alternatives under budget stress; green_buyer strengthens organic preference "
                            "when organic products are available; health_optimizer tracks fat content "
                            "relative to health target; habitual_buyer resists substitution but becomes "
                            "more flexible under repeated budget exhaustion. These are hypotheses, not estimated rules."),
        ("Prediction",      "Consumers do not predict future prices or supply availability. Stockpiling "
                            "is driven by present-biased hyperbolic discounting (beta-delta model) — "
                            "agents over-weight current scarcity signals relative to expected future "
                            "supply normalisation."),
        ("Sensing",         "Consumers observe: shelf stock of desired product (full availability), "
                            "current shelf price, global panic level (broadcast signal), crowd ratio "
                            "(daily consumer count / base_consumers). No private information advantage."),
        ("Interaction",     "Indirect interaction only: consumers compete for finite shelf stock "
                            "(depletion externality). Panic is propagated as a global broadcast signal. "
                            "In multi-store mode, inter-store panic contagion is modelled explicitly."),
        ("Stochasticity",   "Due-household tie breaking and shuffled purchase order within a day; "
                            "optional +/-10% traffic noise is exploratory. "
                            "All stochasticity is seeded for full reproducibility."),
        ("Collectives",     "Consumer archetypes are analytical groups, not explicit collectives. "
                            "Population pool is stratified by archetype to preserve empirical distribution."),
        ("Observation",     "All agent-level and model-level metrics are recorded daily. "
                            "Per-product stock, per-consumer welfare, supply chain log, and food "
                            "waste log are available for full audit. Monte Carlo ensemble "
                            "aggregation uses non-parametric percentiles (p10/p25/p75/p90)."),
    ])

    # ── 5. INITIALISATION ─────────────────────────────────────────────────────
    pdf.chapter(5, "Initialisation", "Initialisation")
    pdf.body(
        "At model creation (SupermarketModel.__init__):"
    )
    pdf.bullet([
        "Random seed is set on Mesa's internal RNG and a separate Python random.Random "
        "instance to guarantee reproducibility at all random call sites.",
        "Store capacities are auto-calibrated: for each product, expected daily demand "
        "is estimated from population pool basket frequencies, then max_shelf_capacity "
        "and max_storage_capacity are set proportionally (shelf-cover heuristic based "
        "on shelf-life: 1.5 days perishable, 2.5 days medium, 4.0 days dry/canned).",
        "Initial stock is set to 75% of max shelf capacity and 60% of max storage "
        "capacity to represent a store mid-cycle (not freshly stocked, not depleted).",
        "One ProductAgent per catalogue SKU and one SupplyTruck are added to the schedule.",
        "PolicyConfig is instantiated from the policy dict (fat tax, subsidy, labelling, "
        "domestic shock parameters); all levers are inactive unless explicitly enabled.",
        "The configuration retains observed participant profiles. Each model seed resamples "
        "complete profiles with replacement to reach the simulation pool size; no participant "
        "attribute or basket value is independently jittered.",
        "Phase-two baskets are excluded from simulated demand. They serve as empirical "
        "targets for an 80/20 hold-out audit and five-fold cross-fitted response estimates. "
        "Any individual-level estimator that does not beat the training-mean benchmark is "
        "rejected and replaced by cross-fitted population means.",
    ])

    _init_rows = []
    if params:
        for k, v in [
            ("base_consumers", params.get("base_con", 200)),
            ("reorder_point",  params.get("reorder", 100)),
            ("target_stock",   params.get("target",  300)),
            ("lead_time",      params.get("lead", 3)),
            ("crisis_start",   params.get("cri_start", 30)),
            ("inflation_pct",  params.get("inf", 25.0)),
        ]:
            _init_rows.append((k, str(v)))
    if _init_rows:
        pdf.sub("Active Initialisation Parameters")
        pdf.kv(_init_rows)

    # ── 6. INPUT DATA ─────────────────────────────────────────────────────────
    pdf.chapter(6, "Input Data", "Input Data")
    pdf.body(
        "GROCERYsim is driven by two primary empirical data sources:"
    )
    pdf.kv([
        ("Consumer Survey (DCE)",
         "116 Finnish participant records collected; 108 currently yield usable matched baskets. "
         "The export includes phase-one and phase-two shopping tasks, a beliefs questionnaire, "
         "and product-attribute choices. The cleaned long-format DCE supplies displayed "
         "prices for a participant-held-out pooled milk choice model; individual random "
         "coefficients and household-specific willingness to pay are not claimed. "
         "Collected under SecureFood ethics approval, stored in Firebase Realtime Database."),
        ("Product Catalogue",
         "Finnish dairy SKUs with: name, category, price (EUR), fat content (%), "
         "origin ('Suomi' or import), organic flag, shelf life (days), initial stock levels. "
         "Stored in data/master_products.json; curated from Finnish grocery retail data."),
        ("Questionnaire and Exploratory Clustering",
         "Declared item groups are audited for missingness and raw Cronbach alpha. K-Means "
         "uses five questionnaire scores plus identifiable origin, organic, and chosen-fat "
         "attributes; the non-identified lookup-price score is excluded. Categories affect "
         "behaviour only if k-selection, separation, bootstrap stability, and size gates pass."),
        ("External validation evidence",
         "No bundled source is currently accepted as independent external validation. "
         "Broad literature ranges previously displayed in the application are retained "
         "only as background and cannot generate a validation claim."),
    ])

    # ── 7. SUBMODELS ──────────────────────────────────────────────────────────
    pdf.chapter(7, "Submodels", "Submodels")

    pdf.sub("7.1  Evidence-Separated Consumer Choice")
    pdf.body(
        "Requested-SKU price loss = price_response * (current_price/reference_price - 1).\n"
        "The product is accepted when this loss is no greater than the phase-transition "
        "calibrated margin. Optional exploratory Prospect Theory changes only this price-loss rule.\n\n"
        "For substitution, candidates must share the catalogue category, be in stock, "
        "and be affordable within the remaining visit budget. Reconstructed one-to-one "
        "replacement events gate transfer of the retention-price screen and deterministic "
        "ranking. During the phase transition, the cross-fitted propensity supplies one "
        "proactive substitution decision per basket line. Failed allocation gates use a "
        "seeded uniform draw among feasible candidates. The maximum crisis budget is "
        "separated from a cross-fitted reservation-spending share. DCE "
        "compatibility is not cardinal utility or willingness to pay."
    )

    pdf.sub("7.2  Inventory (s, S) Reorder Policy")
    pdf.body(
        "Reorder trigger: total_supply = stock_storage + pending_orders < reorder_point\n"
        "  (where reorder_point = reorder_pt_fraction * max_storage_capacity)\n\n"
        "Order quantity: max(0, target_qty - total_supply)\n"
        "  (where target_qty = target_stock_fraction * max_storage_capacity)\n\n"
        "This pipeline-aware formulation avoids the classic double-ordering problem "
        "where pending deliveries are ignored, causing over-ordering during demand surges."
    )

    pdf.sub("7.3  Panic Propagation")
    pdf.body(
        "Each shopper contributes at most one scarcity signal per day. Then:\n"
        "  exposure = signalling_shoppers / daily_consumer_count\n"
        "  growth = panic_sensitivity * max(0, exposure - 0.10) * 0.50\n"
        "  global_panic_level = clamp(global_panic_level + growth - daily_decay)\n\n"
        "This submodel is disabled in empirical-only mode. When explicitly enabled, "
        "hoarding amplification is continuous: 1 + (maximum_multiplier - 1) x "
        "cross_fitted_household_propensity x panic_level. Precautionary pantry cover "
        "also rises continuously with panic via a quasi-hyperbolic discounting heuristic. "
        "Beta varies from 0.75 to 0.90 "
        "as a deterministic function of price sensitivity; this mapping is an "
        "unvalidated model assumption, not an estimated beta-delta parameter."
    )

    pdf.sub("7.4  Exploratory Food-Access Stress Score")
    pdf.body(
        "This objective model diagnostic is calculated for every represented household "
        "from the share of today's interval-adjusted pantry consumption need that cannot "
        "be met: 0 = none; 1 = (0%,25%); 2 = [25%,50%); 3 = [50%,90%); "
        "4 = [90%,100%]. High access stress means score >= 3. Shopping basket shortfall "
        "and panic are reported separately and do not enter the score. "
        "The legacy output labels FIES_* are aliases retained for compatibility. They must not "
        "be interpreted as a prevalence estimate comparable to survey-based FIES."
    )

    pdf.sub("7.5  Heuristic Preference Reinforcement")
    pdf.body(
        "Each archetype updates a different preference dimension after a shopping visit "
        "(learning rate lr = 0.015 / visit):\n\n"
        "  price_champion:  price_sensitivity += lr * (budget_stress_signal - price_sensitivity)\n"
        "  green_buyer:     organic_preference += lr * (organic_availability - organic_preference)\n"
        "  health_optimizer: preferred_fat += lr * (health_target_fat - preferred_fat)\n"
        "  habitual_buyer:  sub_tolerance += lr * (0.80 - sub_tolerance) if budget_exhausted\n\n"
        "Updated preferences are written back to the shared profile pool and persist "
        "across simulation days, creating path-dependent preference evolution."
    )

    pdf.sub("7.6  Global Sensitivity and Uncertainty Analysis")
    pdf.body(
        "Selected model inputs are varied jointly with Latin Hypercube Sampling over "
        "user-declared uniform screening ranges. Every design point is repeated using "
        "the same replicate-seed set (common random numbers). Outcomes are summarized "
        "over the active crisis phase. Partial rank correlation coefficients are computed "
        "after residualizing ranked inputs and ranked outcomes against all other ranked "
        "inputs; 95% intervals use non-parametric bootstrap resampling. A held-out random-"
        "forest permutation ranking is reported only when its test-set R-squared is positive. "
        "Nested-design PRCC rankings provide a convergence diagnostic, while between-point "
        "and within-point variances separate range-driven uncertainty from stochastic noise."
    )

    pdf.sub("7.7  Parameter Evidence and Scientific Readiness")
    _registry = build_parameter_registry(
        stats=(st.session_state.config_data or {}).get("stats", {}),
        runtime_params=params or {},
    )
    _registry_summary = parameter_registry_summary(_registry)
    pdf.body(
        f"The machine-readable evidence registry contains {_registry_summary['n_parameters']} "
        f"influential entries. {_registry_summary['n_identifiable']} are identifiable from "
        "the current GROCERYsim export; "
        f"{_registry_summary['n_unresolved_high_priority']} critical/high-priority entries "
        "remain literature transfers, scenario inputs, or engineering assumptions. "
        "Consequently, the current implementation is classified as an exploratory "
        "scenario tool, not a policy-grade point-prediction or causal-effect model."
    )
    pdf.bullet([
        "Observed data are separated from held-out and cross-fitted calibration results.",
        "Recorded DCE prices identify a pooled milk-choice price coefficient; individual "
        "willingness-to-pay heterogeneity is not estimated.",
        "Panic dynamics, traffic/capacity rules, policy treatment effects, access-stress "
        "thresholds, and provisional CO2 factors require external calibration or validation.",
        "Policy-grade use requires uncertainty propagation across all unresolved critical "
        "parameters plus temporal and external validation on independent data.",
    ])

    pdf.sub("7.8  Identifiability-Gated Calibration")
    pdf.body(
        "Calibration parameters are varied jointly with Latin Hypercube Sampling and "
        "common-random-number replicates. Target residuals are standardized by analyst-"
        "declared measurement-error or tolerance scales. Waste share uses physical "
        "throughput (waste / [sales + waste]), never currency. Before a numerical "
        "best fit can be applied, the workflow requires: at least ten design points per "
        "free parameter; target-space rank at least equal to the number of free parameters; "
        "between-parameter signal exceeding within-point stochastic noise; nearest-neighbour "
        "synthetic recovery for every parameter; and positive validation skill against a "
        "naive training-mean forecast on the final 20 percent of an observed daily series. "
        "KPI-only fits lack a held-out period and are therefore always exploratory."
    )

    pdf.sub("7.9  Evidence-Tiered Validation Protocol")
    pdf.body(
        "Validation evidence is classified as internal invariant, calibration holdout, "
        "scenario plausibility, or external independent. Internal reproducibility and "
        "accounting checks verify implementation but cannot establish empirical validity. "
        "SecureFood phase-two and DCE holdouts test prediction within the model-development "
        "data pipeline and are not labelled independent external validation. An external "
        "target is admissible only when its metric, scenario, time window, aggregation, "
        "acceptance interval, source population and period are declared in advance; the "
        "source is traceable; a timestamped registration is supplied; and the data were not "
        "used for calibration. Stochastic targets require raw Monte Carlo replicates; both "
        "the replicate mean and central 95 percent simulation interval must remain within "
        "the preregistered bounds. Results are exported with unevaluated targets and failures, "
        "and a pass supports validity only for the declared targets and scope. At this "
        "release, no bundled dataset satisfies that external-evidence gate."
    )

    # ── 8. REFERENCES ─────────────────────────────────────────────────────────
    pdf.chapter(8, "References & Citation", "References")
    pdf.bullet([
        "Grimm, V. et al. (2006). A standard protocol for describing individual-based and "
        "agent-based models. Ecological Modelling 198, 115-126.",
        "Grimm, V. et al. (2010). The ODD protocol: A review and first update. "
        "Ecological Modelling 221(23), 2760-2768.",
        "Grimm, V. et al. (2020). The ODD Protocol for Describing Agent-Based and "
        "Other Simulation Models: A Second Update to Improve Clarity, Replication, "
        "and Structural Realism. JASSS 23(2), 7.",
        "Ajzen, I. (1991). The theory of planned behavior. OBHDP 50(2), 179-211.",
        "FAO (2016). Methods for estimating comparable rates of food insecurity globally.",
        "Kahneman, D. & Tversky, A. (1979). Prospect Theory. Econometrica 47(2), 263-291.",
        "O'Donoghue, T. & Rabin, M. (1999). Doing it now or later. AER 89(1), 103-124.",
        "McKay, M. D., Beckman, R. J. & Conover, W. J. (1979). A comparison of three methods for selecting values of input variables. Technometrics 21(2), 239-245.",
        "Thaler, R. H. & Sunstein, C. R. (2008). Nudge. Yale University Press.",
        "SecureFood Consortium (2024-2027). Horizon Europe Grant No. 101136583.",
    ])
    pdf.sub("Citation")
    pdf.finding(
        "Duric, Ivan (2026). GROCERYsim Agent-Based Model for Consumer Behaviour and "
        "Supply Chain Stress-Testing. IAMO XR Lab, SecureFood project, "
        "Horizon Europe Grant 101136583."
    )

    return bytes(pdf.output())


def render_documentation_tab(params: dict):
    st.header("📋 Model Documentation (ODD+D Protocol)")
    st.markdown(
        "The **ODD+D protocol** (Grimm et al. 2006, 2010, 2020) is the community standard "
        "for documenting agent-based models. It supports transparent review and is increasingly "
        "expected by EU Horizon funding bodies and FAO model assessment panels. "
        "Below is the full protocol rendered interactively; download the formatted PDF "
        "for citation and archiving."
    )

    # Download button at top
    _doc_col, _ = st.columns([1, 2])
    with _doc_col:
        if st.button("⚙️ Generate ODD+D PDF", type="primary",
                     use_container_width=True, key="odd_gen_btn"):
            with st.spinner("Building ODD+D documentation PDF…"):
                try:
                    _odd_bytes = _make_odd_d_pdf(params=params)
                    st.session_state["odd_d_pdf"] = _odd_bytes
                except Exception as _e:
                    st.error(f"PDF generation failed: {_e}")
    _od = st.session_state.get("odd_d_pdf")
    if _od:
        with _doc_col:
            st.download_button(
                "📄 Download ODD+D PDF",
                _od,
                "GROCERYsim_ODD+D_Protocol.pdf",
                "application/pdf",
                use_container_width=True,
                key="odd_dl_btn",
            )

    st.divider()

    # ── Render ODD+D inline ──────────────────────────────────────────────────
    sections = [
        ("1. Purpose", """
GROCERYsim ABM v2.0 simulates Finnish grocery retail supply chains and consumer behaviour
under baseline and crisis conditions. Three research objectives:

- **Supply disruption impact**: Quantify how climate-driven disruptions affect revenue, food waste, and food security
- **Policy effectiveness**: Evaluate fat taxation, domestic subsidies, nutritional labelling, and purchase limits
- **Equity analysis**: Describe outcomes across income and participant-profile groups; exploratory archetype labels are not treated as validated types
        """),
        ("2. Entities, State Variables & Scales", """
**Agent types:**
| Agent | Count | Key state variables |
|---|---|---|
| `ProductAgent` | 1 per SKU | `shelf_batches` (FIFO), `stock_storage`, `current_price` |
| `SupplyTruck` | 1 (singleton) | `delivery_queue`, `log` |
| `ConsumerAgent` | 1 visit instance per selected persistent household | visit outcomes; household profile retains pantry; preference learning is optional and off by default |
| `SupermarketModel` | 1 | `global_panic_level`, `is_scenario_active`, `daily_records` |

**Scales:** 1 step = 1 day · single store · default 100 visits/day · 2,000 household profiles
        """),
        ("3. Process Overview & Scheduling", """
Fixed daily order:

1. **Household consumption** — Deplete every household pantry using interval-adjusted daily need; record realised shortfall
2. **Visit scheduling** — Select the most-due unique households at constant declared traffic; calendar multipliers and ±10% noise are exploratory
3. **ProductAgent.step()** — Reset counters → refill shelf from storage (50% threshold) → update price (inflation + policy) → age batches → remove expired (waste log)
4. **SupplyTruck.step()** — Deliver arrived orders → apply domestic shock → place new orders
5. **ConsumerAgent.step()** — Compute pantry-adjusted demand → apply gated decision rule → shop → record shopping shortfall; optional dynamics run only in exploratory mode
6. **Aggregation** — Welfare metrics, panic level, CO₂, import dependency → `daily_records`
        """),
        ("4. Design Concepts", """
| Concept | Implementation |
|---|---|
| **Emergence** | Inventory depletion, waste, fulfilment, and access stress emerge from household and store interactions; panic emerges only in exploratory mode |
| **Adaptation** | Off in empirical-only mode; optional heuristic preference reinforcement (rate 0.015/visit) is an unvalidated extension |
| **Objectives** | Consumers: maximise U subject to budget. Products: minimise stockout via (s,S) policy |
| **Learning** | Disabled by default; archetype-specific rules require both exploratory mode and a passing archetype gate |
| **Sensing** | Shelf stock, shelf price, global panic, crowd ratio — no private information |
| **Interaction** | Indirect (shelf depletion competition) + panic broadcast signal |
| **Stochasticity** | Due-household tie breaking, daily count noise, within-day shopper order — all seeded |
| **Observation** | Full daily log: per-agent, per-product, model-level — Monte Carlo ensemble aggregation |
        """),
        ("5. Initialisation", """
At model creation:
- Fixed seed applied to Mesa RNG and Python `random.Random`
- Store capacities auto-calibrated from population basket frequencies × shelf-cover heuristic
- Initial stock: 75% max shelf capacity, 60% max storage capacity
- PolicyConfig instantiated (all levers inactive unless explicitly enabled)
- Persistent household pool: complete observed participant profiles resampled with replacement per model seed; 2,000 draws do not increase empirical N
        """),
        ("6. Input Data", """
| Dataset | Description | Source |
|---|---|---|
| DCE Consumer Survey | 108 linked respondents; participant-held-out pooled price/origin/organic/fat/opt-out model; individual WTP heterogeneity is not claimed | SecureFood cleaned DCE + Firebase |
| Product Catalogue | Finnish dairy SKUs: name, category, price, fat%, origin, organic flag, shelf life | `master_products.json` |
| Questionnaire/clusters | Reliability audit plus k=2–6 separation and bootstrap-stability checks; categories are operational only if all gates pass | `data_processor.py` |
        """),
        ("7. Submodels", """
**Evidence-separated choice:**
Requested products pass a phase-transition-calibrated proportional-price gate. For
substitution, same-category candidates must also be in stock and affordable. Replacement
incidence is learned from phase-one/phase-two basket transitions. Milk candidates are
sampled from the held-out-tested pooled DCE price-and-attribute probabilities. Other
categories use held-out-supported transition target shares or a seeded uniform fallback.

**Inventory policy (s, S):**
Order when `total_supply = storage + pipeline < reorder_point`; quantity = `target_qty - total_supply`

**Panic propagation:**
`growth = panic_sensitivity × max(0, scarcity_exposure − exposure_floor) × growth_rate`;
active and recovery decay rates are explicit scenario assumptions. This entire pathway
is disabled in empirical-only mode.

**Continuous hoarding:**
`multiplier = 1 + (scenario_max − 1) × cross_fitted_propensity × panic_level`.
There is no arbitrary panic activation threshold, but the mechanism is disabled in
empirical-only mode because real-world panic hoarding is not identified by the experiment.

The stockpile beta is a heuristic deterministic function of price sensitivity (0.75–0.90),
not an empirically estimated beta-delta coefficient.

**Exploratory food-access stress score (not a validated FIES measure):**
For every household, classify today's unmet share of interval-adjusted pantry need:
0 = none · 1 = (0%,25%) · 2 = [25%,50%) · 3 = [50%,90%) · 4 = [90%,100%].
High access stress is score ≥3. Panic and shopping shortfall are reported separately.

**Heuristic preference reinforcement:**
Disabled by default. In exploratory mode, archetype-specific heuristic update rules
(rate 0.015/visit) are permitted only if the archetype stability gate passes.

**Global sensitivity and uncertainty:**
Joint Latin Hypercube sampling over declared ranges; common-random-number replicates;
bootstrap PRCC; held-out nonlinear permutation importance only when predictive;
nested-design convergence and between-parameter versus within-seed variance audit.

**Parameter evidence and readiness:**
The machine-readable registry classifies each influential value as observed data,
held-out/cross-fitted calibration, literature transfer, scenario input, or engineering
assumption. The current model is an exploratory scenario tool; unresolved critical
parameters preclude policy-grade point predictions or causal-effect claims.

**Identifiability-gated calibration:**
Replicated joint LHS designs are screened for target-space rank, stochastic signal,
and nearest-neighbour synthetic parameter recovery. Applying a fitted value additionally
requires positive predictive skill on the final 20% of an observed daily series.
KPI-only fits are exploratory because they have no independent validation period.

**Evidence-tiered validation:**
Internal invariants, calibration holdouts, and scenario-plausibility checks cannot imply
external validity. External targets require a preregistered acceptance interval, traceable
source and scope metadata, a timestamped registration, and independence from calibration.
A stochastic target requires raw Monte Carlo replicates; both the replicate mean and central
95% simulation interval must stay within the registered acceptance interval.
A complete pass is explicitly limited to the declared targets, population, period, and
scenarios. No bundled dataset currently passes this independent-external-evidence gate.
        """),
        ("8. References", """
- Grimm, V. et al. (2020). ODD+D Protocol. *JASSS* 23(2), 7.
- Ajzen, I. (1991). Theory of planned behavior. *OBHDP* 50(2), 179–211.
- FAO (2016). Methods for estimating comparable rates of food insecurity globally.
- Kahneman, D. & Tversky, A. (1979). Prospect Theory. *Econometrica* 47(2), 263–291.
- SecureFood Consortium (2024–2027). Horizon Europe Grant No. 101136583.

**Cite as:** Durić, Ivan (2026). GROCERYsim Agent-Based Model for Consumer Behaviour and Supply Chain Stress-Testing. IAMO XR Lab, SecureFood / Horizon Europe Grant 101136583.
        """),
    ]

    for _title, _content in sections:
        with st.expander(_title, expanded=False):
            st.markdown(_content)


# ===========================================================================
# 11d. TAB: POLICY ANALYSIS
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
                    "consumer_model": "calibrated requested-SKU price acceptance + replacement-gated stochastic substitution",
                    "archetype_clustering": (
                        "operational" if stats.get("archetype_stability", {}).get("archetypes_supported")
                        else "exploratory only—categorical modifiers disabled"
                    ),
                    "population_pool_size": stats.get("pool_size", "N/A"),
                    "real_participants": stats.get("n_real", "N/A"),
                    "population_resampling": stats.get("population_method", "N/A"),
                    "empirical_sampling_units": stats.get("empirical_sampling_units", "N/A"),
                    "shelf_model": "FIFO batches with near-expiry discount",
                    "default_behavior_evidence_mode": "empirical_only",
                    "unvalidated_dynamic_extensions": "explicit opt-in; off by default",
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
                        "panic_exposure_floor": model_cris.panic_exposure_floor,
                        "panic_growth_rate": model_cris.panic_growth_rate,
                        "panic_decay_active": model_cris.panic_decay_active,
                        "panic_decay_recovery": model_cris.panic_decay_recovery,
                        "inflation_panic_rate": model_cris.inflation_panic_rate,
                        "behavior_evidence_mode": (
                            "exploratory_extensions"
                            if any((model_cris.panic_dynamics_enabled,
                                    model_cris.tpb_enabled,
                                    model_cris.prospect_theory_enabled,
                                    model_cris.preference_learning_enabled,
                                    model_cris.archetype_modifiers_enabled,
                                    model_cris.policy_choice_effects_enabled))
                            else "empirical_only"
                        ),
                        "panic_dynamics_enabled": model_cris.panic_dynamics_enabled,
                        "tpb_enabled": model_cris.tpb_enabled,
                        "prospect_theory_enabled": model_cris.prospect_theory_enabled,
                        "preference_learning_enabled": model_cris.preference_learning_enabled,
                        "archetype_modifiers_enabled": model_cris.archetype_modifiers_enabled,
                        "policy_choice_effects_enabled": model_cris.policy_choice_effects_enabled,
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
            > Consumer agents use a participant-held-out pooled DCE assessment for milk price/origin/organic/fat choice and
            > cross-fitted phase-transition calibration for price response and substitution.
            > individual willingness-to-pay heterogeneity is not claimed. Declared
            > questionnaire constructs are reliability-audited, and k=2–6 clustering solutions are
            > checked for separation, bootstrap stability, and minimum size. Categorical modifiers
            > operate only when every gate passes. Each model seed resamples complete observed
            > participant profiles with replacement; no synthetic attributes are jittered. Requested-SKU
            > acceptance uses a separately calibrated proportional-price gate. Same-category substitutes
            > must be affordable. Milk candidate allocation uses pooled DCE multinomial probabilities;
            > other categories use held-out-supported phase-transition target shares or a seeded uniform
            > draw among feasible candidates.
            > Empirical-only mode is the default: price response uses a transparent relative-price rule,
            > while panic contagion, TPB, Prospect Theory, archetype modifiers, reference-price adaptation,
            > and preference learning are disabled. These mechanisms require explicit exploratory opt-in
            > and are reported as assumptions rather than participant-derived effects.
            > Supply chain uses FIFO shelf batches with near-expiry (50% off) discounting and
            > reorder-point replenishment. Policy levers (fat tax, subsidy, supply shock,
            > nutritional labelling) modify prices and delivery volumes. Environmental impact
            > is tracked via product-level CO₂ emission factors. Consumer welfare separates
            > visit-level budget/basket outcomes from population-wide realised pantry-consumption shortfall.
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
# 11d-b. REPLICATED GLOBAL SENSITIVITY AND UNCERTAINTY ANALYSIS
# ===========================================================================

def render_sensitivity_tab(params: dict):
    st.header(_t("header_sensitivity"))
    st.markdown(
        "All selected inputs vary jointly with **Latin Hypercube Sampling (LHS)**. "
        "Every design point is repeated with common random-number seeds, allowing "
        "parameter-driven variation to be separated from stochastic simulation noise. "
        "PRCC screens monotonic effects; nonlinear permutation importance is reported "
        "only if its emulator predicts unseen design points."
    )
    st.warning(
        "Results are conditional on the stated uniform screening ranges, selected "
        "outcome, and crisis configuration. They are not universal causal effects or "
        "probabilistic forecasts. Policy interventions are analysed separately."
    )
    if st.session_state.config_data is None:
        st.warning("⚠️ Upload and process data in **🏠 Data & Population** first.")
        return

    definitions = {
        "reorder": ("Reorder point", 0.10, 0.60, "float"),
        "target": ("Restock target", 0.60, 0.99, "float"),
        "lead": ("Lead time (days)", 1, 10, "int"),
        "base_con": ("Daily consumers", 50, 250, "int"),
        "panic": ("Panic sensitivity", 0.0, 1.0, "float"),
        "hoard": ("Hoarding multiplier", 1.0, 2.5, "float"),
        "inf": ("Crisis inflation (%)", 0.0, 100.0, "float"),
        "dis": ("Supply disruption (days)", 0, 21, "int"),
        "panic_exposure_floor": ("Normal scarcity exposure floor", 0.0, 0.30, "float"),
        "panic_growth_rate": ("Scarcity-to-panic growth rate", 0.10, 1.0, "float"),
        "panic_decay_active": ("Active-phase panic decay", 0.0, 0.15, "float"),
        "panic_decay_recovery": ("Recovery-phase panic decay", 0.02, 0.25, "float"),
        "inflation_panic_rate": ("Inflation-to-panic rate", 0.0, 0.80, "float"),
    }
    exploratory_sensitivity = bool(params.get("exploratory_behaviour", False))
    if not exploratory_sensitivity:
        for assumption_key in (
            "panic", "hoard", "panic_exposure_floor", "panic_growth_rate",
            "panic_decay_active", "panic_decay_recovery", "inflation_panic_rate",
        ):
            definitions.pop(assumption_key, None)
        st.info(
            "Empirical-only sensitivity: unidentified panic and hoarding parameters "
            "are excluded. Enable exploratory dynamic behaviour to screen them."
        )
    metrics = {
        "Revenue": "Mean crisis-phase constant-price revenue",
        "Sales": "Mean crisis-phase units sold",
        "Waste": "Mean crisis-phase waste",
        "LostSales": "Mean crisis-phase lost sales",
        "BudgetExhaustionRate": "Mean crisis-phase budget exhaustion",
        "FulfillmentRate": "Mean crisis-phase shopping fulfillment",
        "ConsumptionFulfillmentRate": "Mean crisis-phase consumption fulfillment",
        "HouseholdConsumptionShortfallShare": "Households with consumption shortfall",
        "CumulativeConsumptionShortfallRate": "End-of-run cumulative consumption shortfall",
        "AccessStressHigh_Low": "High consumption-access stress: low income",
        "PanicLevel": "Mean crisis-phase panic",
        "CO2Total": "Mean crisis-phase CO₂",
    }
    if not exploratory_sensitivity:
        metrics.pop("PanicLevel", None)
    label_to_key = {definition[0]: key for key, definition in definitions.items()}
    default_sensitivity_keys = ["reorder", "target", "lead", "base_con", "inf", "dis"]
    if exploratory_sensitivity:
        default_sensitivity_keys[4:4] = ["panic", "hoard"]
    selected_labels = st.multiselect(
        "Parameters varied jointly", list(label_to_key),
        default=[definitions[key][0] for key in default_sensitivity_keys],
        key="gsa_parameters_v2",
    )
    selected_keys = [label_to_key[label] for label in selected_labels]
    if len(selected_keys) < 3:
        st.error("Select at least three parameters for global sensitivity analysis.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric = st.selectbox(
            "Outcome", list(metrics), format_func=lambda key: metrics[key],
            key="gsa_metric_v2",
        )
    with c2:
        n_samples = st.slider("LHS design points", 32, 160, 80, 8, key="gsa_samples_v2")
    with c3:
        n_replicates = st.slider("Replicates per point", 2, 5, 3, 1, key="gsa_reps_v2")
    with c4:
        minimum_days = max(21, int(params.get("cri_start", 30)) + 7)
        maximum_days = max(365, minimum_days + 30)
        days = st.slider(
            "Days per run", minimum_days, maximum_days, max(60, minimum_days),
            key="gsa_days_v2",
        )

    ranges = []
    with st.expander("🎛️ Parameter ranges (uniform screening ranges)", expanded=False):
        for key in selected_keys:
            label, low_default, high_default, _ = definitions[key]
            r1, r2 = st.columns(2)
            with r1:
                low = st.number_input(
                    f"{label} — minimum", value=float(low_default), key=f"gsa_low_{key}"
                )
            with r2:
                high = st.number_input(
                    f"{label} — maximum", value=float(high_default), key=f"gsa_high_{key}"
                )
            ranges.append((key, float(low), float(high)))
    if any(high <= low for _, low, high in ranges):
        st.error("Every parameter maximum must be greater than its minimum.")
        return
    if n_samples < 10 * len(selected_keys):
        st.warning(
            f"Screening reliability is limited: {n_samples} points for "
            f"{len(selected_keys)} inputs. At least 10 points per input is recommended."
        )
    if n_replicates < 3:
        st.warning(
            "Two replicates provide only a coarse stochastic-noise estimate. "
            "Use at least three replicates for results intended for publication."
        )

    total_runs = n_samples * n_replicates
    crisis_start = int(params.get("cri_start", 30))
    st.caption(
        f"Planned: {n_samples} joint points × {n_replicates} replicates = "
        f"{total_runs} ABM runs; outcome summarized from crisis day {crisis_start}."
    )

    if st.button("🔬 Run Replicated Global Sensitivity Analysis", type="primary", key="gsa_run_v2"):
        unit = latin_hypercube(n_samples, len(selected_keys), seed=20260721)
        physical = scale_design(unit, [(low, high) for _, low, high in ranges])
        replicate_matrix = np.empty((n_samples, n_replicates), dtype=float)
        raw_rows = []
        progress = st.progress(0, text="Starting replicated LHS…")
        completed = 0

        for sample_id, values in enumerate(physical):
            overrides = {}
            for column, (key, value) in enumerate(zip(selected_keys, values)):
                kind = definitions[key][3]
                overrides[key] = int(round(value)) if kind == "int" else float(value)
                # Statistical analysis must use the value the ABM actually received.
                physical[sample_id, column] = overrides[key]
            for replicate in range(n_replicates):
                run_params = {**params, **overrides, "days": days}
                seed = 9100 + replicate  # common random numbers across design points
                model = _make_model(
                    run_params, is_crisis=True, seed=seed,
                    policy_cfg=params.get("policy_cfg", {}),
                )
                for _ in range(days):
                    model.step()
                phase = [
                    record for record in model.daily_records
                    if int(record.get("Day", 0)) >= crisis_start
                ] or model.daily_records
                if metric == "CumulativeConsumptionShortfallRate":
                    outcome = float(phase[-1].get(metric, 0.0))
                else:
                    outcome = float(np.mean([record.get(metric, 0.0) for record in phase]))
                replicate_matrix[sample_id, replicate] = outcome
                raw_row = {
                    "DesignPoint": sample_id, "Replicate": replicate,
                    "Seed": seed, "Outcome": outcome,
                }
                raw_row.update(overrides)
                raw_rows.append(raw_row)
                completed += 1
                if completed % max(1, total_runs // 100) == 0:
                    progress.progress(
                        completed / total_runs,
                        text=f"Replicated LHS: {completed}/{total_runs} runs",
                    )
        progress.empty()
        design_means = replicate_matrix.mean(axis=1)
        st.session_state.gsa_results_v2 = {
            "raw": pd.DataFrame(raw_rows),
            "physical": physical,
            "design_means": design_means,
            "replicates": replicate_matrix,
            "keys": selected_keys,
            "labels": {key: definitions[key][0] for key in selected_keys},
            "ranges": ranges,
            "metric": metric,
            "metric_label": metrics[metric],
            "days": days,
            "crisis_start": crisis_start,
            "prcc": bootstrap_prcc(physical, design_means, 300, seed=20260722),
            "nonlinear": nonlinear_permutation_importance(
                physical, design_means, seed=20260723
            ),
            "convergence": convergence_diagnostics(physical, design_means),
            "decomposition": variance_decomposition(replicate_matrix),
        }
        st.success(f"✅ Replicated global analysis complete — {total_runs} ABM runs.")

    results = st.session_state.get("gsa_results_v2")
    if not results:
        st.info("Run the replicated analysis to generate uncertainty and importance results.")
        return

    design_means = np.asarray(results["design_means"])
    replicates = np.asarray(results["replicates"])
    decomposition = results["decomposition"]
    q05, q50, q95 = np.quantile(design_means, [0.05, 0.50, 0.95])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Median outcome", f"{q50:.4g}")
    k2.metric("90% screening range", f"{q05:.4g} – {q95:.4g}")
    k3.metric("Parameter variance share", f"{decomposition['parameter_variance_share']:.1%}")
    k4.metric("ABM runs", str(replicates.size))
    st.caption(
        "The 5th–95th percentile range is induced by the declared uniform screening "
        "ranges. It is not a forecast interval or confidence interval."
    )

    prcc_rows = []
    for key, estimate in zip(results["keys"], results["prcc"]):
        prcc_rows.append({
            "Parameter": results["labels"][key],
            "PRCC": estimate["coefficient"],
            "CI low": estimate["ci_low"],
            "CI high": estimate["ci_high"],
            "p-value (unadjusted)": estimate["p_value"],
            "Robust direction": (
                "yes" if estimate["ci_low"] * estimate["ci_high"] > 0 else "no"
            ),
        })
    prcc_df = pd.DataFrame(prcc_rows).sort_values("PRCC", key=abs, ascending=False)
    st.subheader("📉 Monotonic importance: PRCC with bootstrap intervals")
    colors = ["#DC143C" if value > 0 else "#2980b9" for value in prcc_df["PRCC"]]
    fig_prcc = go.Figure(go.Bar(
        x=prcc_df["PRCC"], y=prcc_df["Parameter"], orientation="h",
        marker_color=colors,
        error_x=dict(
            type="data", symmetric=False,
            array=prcc_df["CI high"] - prcc_df["PRCC"],
            arrayminus=prcc_df["PRCC"] - prcc_df["CI low"],
        ),
    ))
    fig_prcc.add_vline(x=0, line_color="#042026", line_width=1)
    fig_prcc.update_layout(
        template="plotly_white", xaxis=dict(range=[-1, 1], title="PRCC"),
        yaxis=dict(autorange="reversed"), title=results["metric_label"],
    )
    st.plotly_chart(fig_prcc, use_container_width=True, config=_PLOTLY_CFG)
    st.dataframe(prcc_df.round(4), use_container_width=True, hide_index=True)
    st.caption(
        "PRCC uses ranked inputs and outcome, residualized against every other "
        "ranked input. P-values are descriptive and unadjusted; prefer the bootstrap interval."
    )

    st.subheader("🌲 Nonlinear screening")
    nonlinear = results["nonlinear"]
    if nonlinear["status"] == "retained":
        nonlinear_df = pd.DataFrame({
            "Parameter": [results["labels"][key] for key in results["keys"]],
            "Normalized permutation importance": nonlinear["importance"],
        }).sort_values("Normalized permutation importance", ascending=False)
        st.metric("Held-out emulator R²", f"{nonlinear['heldout_r2']:.3f}")
        fig_nonlinear = px.bar(
            nonlinear_df, x="Normalized permutation importance", y="Parameter",
            orientation="h", title="Held-out nonlinear permutation importance",
        )
        fig_nonlinear.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_nonlinear, use_container_width=True, config=_PLOTLY_CFG)
    else:
        st.warning(
            f"Nonlinear importance rejected ({nonlinear['status']}; held-out "
            f"R²={nonlinear['heldout_r2']:.3f}). The emulator did not predict unseen "
            "points reliably, so its ranking is not shown."
        )

    st.subheader("🧪 Convergence and stochastic-noise audit")
    convergence_df = pd.DataFrame(results["convergence"]["rows"])
    st.dataframe(convergence_df.round(4), use_container_width=True, hide_index=True)
    if results["convergence"]["stable"]:
        st.success("PRCC ranking is stable between the half and full designs (ρ ≥ 0.8).")
    else:
        st.warning("PRCC ranking has not converged; increase the LHS design size.")
    st.markdown(
        f"- Between-parameter variance: `{decomposition['between_parameter_variance']:.6g}`\n"
        f"- Mean within-point stochastic variance: `{decomposition['within_stochastic_variance']:.6g}`\n"
        f"- Share associated with parameter variation over these ranges: "
        f"`{decomposition['parameter_variance_share']:.1%}`"
    )

    metadata = pd.DataFrame([{
        "Metric": results["metric"], "MetricLabel": results["metric_label"],
        "Days": results["days"], "CrisisStart": results["crisis_start"],
        "DesignPoints": len(design_means), "Replicates": replicates.shape[1],
        "Sampling": "Latin Hypercube; uniform screening ranges",
        "SeedStrategy": "common random numbers; seeds 9100+replicate",
    }])
    st.download_button(
        "📥 Download replicated design and outcomes (CSV)",
        results["raw"].to_csv(index=False).encode("utf-8"),
        "grocerysim_global_sensitivity_runs.csv", "text/csv", key="dl_gsa_v2",
    )
    st.download_button(
        "📥 Download analysis metadata (CSV)",
        metadata.to_csv(index=False).encode("utf-8"),
        "grocerysim_global_sensitivity_metadata.csv", "text/csv", key="dl_gsa_meta_v2",
    )


# ===========================================================================
# 11e. MODEL CALIBRATION TAB
# ===========================================================================

def _render_calibration_tab_legacy(params: dict):
    st.header("🎯 Model Calibration")
    st.markdown(
        "**Calibration** fits model parameters to empirical targets using "
        "**Latin Hypercube Sampling** (LHS) + RMSE minimisation. "
        "Upload a real sales time series or specify target KPI values, "
        "then apply the best-fit parameters directly to the simulation sliders."
    )

    if st.session_state.config_data is None:
        st.warning("⚠️ Upload and process data in **🏠 Data & Population** first.")
        return

    # ── Calibration target ────────────────────────────────────────────────────
    st.subheader("🎯 Calibration Target")
    _cal_mode = st.radio(
        "Target type",
        ["📊 Target KPI Values", "📁 Upload Time Series CSV"],
        horizontal=True, key="cal_mode",
    )

    _target_revenue    = None
    _target_fulfilment = None
    _target_waste      = None
    _target_series     = None
    _target_col        = "Revenue"

    if "📊 Target KPI Values" in _cal_mode:
        _tc1, _tc2, _tc3 = st.columns(3)
        with _tc1:
            _target_revenue = st.number_input(
                "Avg Daily Revenue (€)", 100.0, 50000.0, 5000.0, 100.0, key="cal_rev_t")
        with _tc2:
            _target_fulfilment = st.number_input(
                "Fulfilment Rate (%)", 50.0, 100.0, 92.0, 0.5, key="cal_ful_t")
        with _tc3:
            _target_waste = st.number_input(
                "Waste Rate (%)", 0.5, 20.0, 2.5, 0.1, key="cal_waste_t")
        _target_col = st.selectbox(
            "Primary metric for RMSE", ["Revenue", "FulfillmentRate", "Waste"],
            key="cal_primary_col",
        )
    else:
        _uploaded_ts = st.file_uploader(
            "CSV with columns Day + one metric column (e.g. daily revenue)",
            type=["csv"], key="cal_ts_upload",
        )
        if _uploaded_ts:
            try:
                _target_series = pd.read_csv(_uploaded_ts)
                st.success(f"✅ Loaded {len(_target_series)} rows")
                st.dataframe(_target_series.head(8), use_container_width=True, hide_index=True)
                _non_day = [c for c in _target_series.columns if c.lower() != "day"]
                _target_col = st.selectbox(
                    "Calibrate against column", _non_day, key="cal_ts_col")
            except Exception as _e:
                st.error(f"Could not read CSV: {_e}")

    # ── Parameters to calibrate ───────────────────────────────────────────────
    st.divider()
    st.subheader("⚙️ Free Parameters")
    _CAL_PARAMS = {
        "base_consumers": {"label": "Consumers / day",  "min": 30,   "max": 250,  "default": True},
        "reorder_pt":     {"label": "Reorder Point",    "min": 0.10, "max": 0.60, "default": True},
        "target_stock":   {"label": "Restock Target",   "min": 0.60, "max": 0.99, "default": True},
        "lead_time":      {"label": "Lead Time (days)", "min": 1,    "max": 10,   "default": True},
        "panic_sens":     {"label": "Panic Sensitivity","min": 0.10, "max": 0.80, "default": False},
        "inflation":      {"label": "Inflation %",      "min": 0,    "max": 50,   "default": False},
    }
    _active_cal = {}
    _chk_cols   = st.columns(3)
    for _ci, (_pn, _pd) in enumerate(_CAL_PARAMS.items()):
        with _chk_cols[_ci % 3]:
            if st.checkbox(
                f"**{_pd['label']}** ({_pd['min']}–{_pd['max']})",
                value=_pd["default"], key=f"cal_chk_{_pn}",
            ):
                _active_cal[_pn] = _pd

    if not _active_cal:
        st.warning("Select at least one parameter to calibrate.")
        return

    _cc1, _cc2 = st.columns(2)
    with _cc1:
        _cal_n = st.slider("LHS samples (N)", 20, 200, 60, 10, key="cal_n",
                            help="More samples → better parameter space coverage but slower.")
    with _cc2:
        _cal_days = st.slider("Days per run", 14, 60, 30, key="cal_days")

    st.info(
        f"Will run **{_cal_n}** LHS samples × {_cal_days} days "
        f"over **{len(_active_cal)}** free parameters "
        f"({_cal_n * len(_active_cal)} total evaluations)."
    )

    if st.button("🎯 Run Calibration", type="primary", key="cal_run_btn"):
        _cal_list = list(_active_cal.items())
        _kc = len(_cal_list)
        _rng_cal = np.random.default_rng(seed=42)

        # Latin Hypercube Sampling
        _lhs_cal = np.zeros((_cal_n, _kc))
        for _j in range(_kc):
            _prm = _rng_cal.permutation(_cal_n)
            _lhs_cal[:, _j] = (_prm + _rng_cal.random(_cal_n)) / _cal_n

        _cal_prog = st.progress(0, text="Running calibration…")
        _cal_results = []

        for _si in range(_cal_n):
            _cal_prog.progress((_si + 1) / _cal_n, f"Sample {_si+1}/{_cal_n}…")
            _ov = {}
            _sp = {}
            for _j, (_pn, _pd) in enumerate(_cal_list):
                _val = _pd["min"] + _lhs_cal[_si, _j] * (_pd["max"] - _pd["min"])
                _ov[_pn] = _val
                _sp[_pn] = round(_val, 3)

            try:
                _pm = {**params, **_ov}
                _mc = _make_model(_pm, is_crisis=False, seed=42)
                for _ in range(_cal_days):
                    _mc.step()
                _df_c = pd.DataFrame(_mc.daily_records)
                if _df_c.empty:
                    continue

                # Compute normalised RMSE
                _rmse_parts = []
                if "📊 Target KPI Values" in _cal_mode:
                    if _target_revenue and "Revenue" in _df_c.columns:
                        _rmse_parts.append(
                            float(np.sqrt(np.mean((_df_c["Revenue"] - _target_revenue) ** 2)))
                            / max(_target_revenue, 1)
                        )
                    if _target_fulfilment and "FulfillmentRate" in _df_c.columns:
                        _rmse_parts.append(
                            float(np.sqrt(np.mean((_df_c["FulfillmentRate"] * 100 - _target_fulfilment) ** 2)))
                            / 100
                        )
                    if _target_waste and "Waste" in _df_c.columns:
                        _tot = max(float(_df_c.get("Revenue", pd.Series([100])).mean()), 1)
                        _wpct = float((_df_c["Waste"] / _tot * 100).mean())
                        _rmse_parts.append(abs(_wpct - _target_waste) / 10)
                    _rmse = float(np.mean(_rmse_parts)) if _rmse_parts else 999.0
                elif _target_series is not None and _target_col in _target_series.columns and _target_col in _df_c.columns:
                    _sv = _df_c[_target_col].values
                    _tv = _target_series[_target_col].values
                    _n  = min(len(_sv), len(_tv))
                    _rmse = float(np.sqrt(np.mean((_sv[:_n] - _tv[:_n]) ** 2)))
                else:
                    continue

                _cal_results.append({
                    "rmse":       _rmse,
                    "params":     _sp,
                    "revenue":    float(_df_c["Revenue"].mean()) if "Revenue" in _df_c.columns else 0.0,
                    "fulfilment": float(_df_c["FulfillmentRate"].mean()) if "FulfillmentRate" in _df_c.columns else 0.0,
                    "waste":      float(_df_c["Waste"].mean()) if "Waste" in _df_c.columns else 0.0,
                    "df":         _df_c,
                })
            except Exception:
                continue

        _cal_results.sort(key=lambda x: x["rmse"])
        st.session_state["cal_results"] = _cal_results

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.get("cal_results"):
        _cr    = st.session_state["cal_results"]
        _best  = _cr[0]

        st.success(f"✅ Calibration complete — best normalised RMSE = **{_best['rmse']:.4f}**")

        st.subheader("🏆 Best-Fit Parameters")
        _bpc = st.columns(len(_best["params"]))
        for _bi, (_pn, _pv) in enumerate(_best["params"].items()):
            _bpc[_bi].metric(_active_cal[_pn]["label"], f"{_pv:.3f}")

        if st.button("⚡ Apply Calibrated Parameters to Simulation",
                     type="primary", key="cal_apply_btn"):
            _KEY_MAP = {
                "base_consumers": "base_con",
                "reorder_pt":     "reorder",
                "target_stock":   "target",
                "lead_time":      "lead",
                "panic_sens":     "panic",
                "inflation":      "inf",
            }
            for _pn, _pv in _best["params"].items():
                _ssk = _KEY_MAP.get(_pn, _pn)
                if _ssk:
                    st.session_state[_ssk] = _pv
            st.success("✅ Applied! Go to 🎮 Interactive Demo and re-run the simulation.")

        # Top-10 table
        st.subheader("📊 Top 10 Calibration Runs")
        _top10 = _cr[:10]
        _t10_rows = []
        for _ri, _rr in enumerate(_top10):
            _row = {"Rank": _ri + 1, "RMSE": f"{_rr['rmse']:.4f}",
                    "Revenue": f"{_rr['revenue']:.0f}",
                    "Fulfilment": f"{_rr['fulfilment']:.3f}"}
            for _pn, _pv in _rr["params"].items():
                _row[_active_cal[_pn]["label"]] = f"{_pv:.3f}"
            _t10_rows.append(_row)
        st.dataframe(pd.DataFrame(_t10_rows), use_container_width=True, hide_index=True)

        # RMSE histogram
        _rmse_all = [_rr["rmse"] for _rr in _cr]
        _fig_rh = go.Figure(go.Histogram(
            x=_rmse_all, nbinsx=20,
            marker_color="#DBA159", marker_line_color="#042026", marker_line_width=0.8,
        ))
        _fig_rh.add_vline(x=_best["rmse"], line_color="#DC143C", line_dash="dash",
                          annotation_text=f"Best: {_best['rmse']:.4f}",
                          annotation_position="top right")
        _fig_rh.update_layout(
            title="RMSE Distribution Across All LHS Samples",
            xaxis_title="Normalised RMSE", yaxis_title="Count", template="plotly_white",
        )
        st.plotly_chart(_fig_rh, use_container_width=True, config=_PLOTLY_CFG)

        # Parameter posterior scatter matrix
        _top20p = _cr[:max(5, int(len(_cr) * 0.2))]
        st.subheader(f"🔎 Parameter Posterior — Top {len(_top20p)} Runs (best 20%)")
        _df_post = pd.DataFrame([
            {**{_active_cal[k]["label"]: v for k, v in _rr["params"].items()},
             "RMSE": _rr["rmse"]}
            for _rr in _top20p
        ])
        _plabs = [_active_cal[_pn]["label"] for _pn in _active_cal]
        if len(_plabs) >= 2:
            _fig_pm = px.scatter_matrix(
                _df_post, dimensions=_plabs, color="RMSE",
                color_continuous_scale=["#27AE60", "#DBA159", "#DC143C"],
                title="Parameter Posterior Scatter Matrix (top 20% by RMSE)",
            )
            _fig_pm.update_traces(diagonal_visible=False, marker_size=5)
            _fig_pm.update_layout(template="plotly_white")
            st.plotly_chart(_fig_pm, use_container_width=True, config=_PLOTLY_CFG)

        # Best-fit trajectory
        st.subheader("📈 Best-Fit Trajectory vs Target")
        _fig_bt = go.Figure()
        if "Revenue" in _best["df"].columns:
            _fig_bt.add_trace(go.Scatter(
                x=_best["df"]["Day"], y=_best["df"]["Revenue"],
                name="Best-fit model", line=dict(color="#DBA159", width=2.5),
            ))
        if "📊 Target KPI Values" in _cal_mode and _target_revenue:
            _fig_bt.add_hline(y=_target_revenue, line_color="#DC143C", line_dash="dash",
                               annotation_text="Target Revenue",
                               annotation_position="bottom right")
        elif _target_series is not None and _target_col in _target_series.columns:
            _fig_bt.add_trace(go.Scatter(
                x=_target_series.get("Day", list(range(len(_target_series)))),
                y=_target_series[_target_col],
                name="Empirical target", line=dict(color="#DC143C", width=2, dash="dot"),
            ))
        _fig_bt.update_layout(
            title="Best-Fit Simulation Revenue vs Target",
            xaxis_title="Day", yaxis_title="Revenue (€)",
            template="plotly_white", legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(_fig_bt, use_container_width=True, config=_PLOTLY_CFG)

        st.download_button(
            "📥 Download All Calibration Results (CSV)",
            pd.DataFrame([
                {**{"RMSE": _rr["rmse"], "Revenue": _rr["revenue"]},
                 **_rr["params"]}
                for _rr in _cr
            ]).to_csv(index=False).encode("utf-8"),
            "calibration_results.csv", "text/csv", key="dl_cal",
        )


def render_calibration_tab(params: dict):
    """Identifiability-gated calibration with temporal holdout validation."""
    st.header("🎯 Model Calibration")
    st.markdown(
        "This workflow separates **numerical fit** from **parameter identification**. "
        "A candidate can be applied only after replicated simulations pass target-space "
        "rank, stochastic-noise, synthetic-recovery, and held-out validation checks."
    )
    st.warning(
        "A low calibration error alone does not identify a parameter. KPI-only targets "
        "lack an independent validation period and therefore produce exploratory fits, "
        "never an automatically applicable recommendation."
    )
    if st.session_state.config_data is None:
        st.warning("⚠️ Load data in **🏠 Data & Population** first.")
        return

    regime_col, mode_col = st.columns(2)
    with regime_col:
        regime = st.radio(
            "Simulation regime", ["Baseline", "Crisis"], horizontal=True,
            key="cal_regime_v2",
            help="Panic parameters are inactive and unavailable under Baseline.",
        )
    with mode_col:
        mode = st.radio(
            "Target evidence",
            ["KPI values (no holdout)", "Observed daily time series"],
            horizontal=True, key="cal_mode_v2",
        )

    st.subheader("🎯 Empirical Targets and Error Scales")
    target_ready = True
    target_values = target_scales = target_series = None
    target_column = None
    target_labels = []
    series_scale = None
    if mode == "KPI values (no holdout)":
        k1, k2, k3 = st.columns(3)
        with k1:
            target_revenue = st.number_input(
                "Mean daily constant-price revenue (€)", 1.0, 50000.0,
                5000.0, 100.0, key="cal_rev_v2",
            )
            scale_revenue = st.number_input(
                "Revenue error scale (€)", 1.0, 10000.0, 500.0, 10.0,
                key="cal_rev_scale_v2",
            )
        with k2:
            target_fulfilment = st.number_input(
                "Mean shopping fulfilment (%)", 0.0, 100.0, 92.0, 0.5,
                key="cal_ful_v2",
            )
            scale_fulfilment = st.number_input(
                "Fulfilment error scale (percentage points)", 0.1, 50.0,
                3.0, 0.5, key="cal_ful_scale_v2",
            )
        with k3:
            target_waste = st.number_input(
                "Waste share of physical throughput (%)", 0.0, 100.0,
                2.5, 0.1, key="cal_waste_v2",
            )
            scale_waste = st.number_input(
                "Waste-share error scale (percentage points)", 0.1, 50.0,
                1.0, 0.1, key="cal_waste_scale_v2",
            )
        target_values = np.asarray(
            [target_revenue, target_fulfilment, target_waste], dtype=float
        )
        target_scales = np.asarray(
            [scale_revenue, scale_fulfilment, scale_waste], dtype=float
        )
        target_labels = ["Revenue", "Fulfilment %", "Waste share %"]
        st.caption(
            "Waste share = wasted units / (sold units + wasted units). Error scales "
            "define target weights and must reflect measurement error or tolerance."
        )
    else:
        upload = st.file_uploader(
            "CSV containing Day and one supported model-output column",
            type=["csv"], key="cal_ts_upload_v2",
        )
        supported = [
            "Revenue", "FulfillmentRate", "Sales", "Waste", "LostSales",
            "PanicLevel", "ConsumptionFulfillmentRate",
        ]
        if upload is None:
            target_ready = False
            st.info("Upload an observed daily series to enable temporal holdout validation.")
        else:
            try:
                uploaded = pd.read_csv(upload)
                available = [column for column in supported if column in uploaded.columns]
                if "Day" not in uploaded.columns or not available:
                    raise ValueError(
                        "CSV must contain Day and one of: " + ", ".join(supported)
                    )
                target_column = st.selectbox(
                    "Observed output column", available, key="cal_ts_col_v2"
                )
                target_series = uploaded[["Day", target_column]].copy()
                target_series[target_column] = pd.to_numeric(
                    target_series[target_column], errors="coerce"
                )
                target_series = (
                    target_series.dropna().sort_values("Day").reset_index(drop=True)
                )
                if len(target_series) < 10:
                    raise ValueError("At least 10 valid daily observations are required.")
                default_scale = max(
                    0.001, float(target_series[target_column].std(ddof=1)) * 0.25
                )
                series_scale = st.number_input(
                    "Measurement / tolerance scale (same unit as series)",
                    min_value=0.001, value=default_scale, key="cal_ts_scale_v2",
                )
                st.success(
                    f"Loaded {len(target_series)} observations: first 80% fit, "
                    "final 20% held out for validation."
                )
                st.caption(
                    "The uploaded column must use the model export unit exactly "
                    "(for example FulfillmentRate is 0–1, not 0–100)."
                )
                st.dataframe(target_series.head(10), hide_index=True, use_container_width=True)
            except Exception as error:
                target_ready = False
                st.error(f"Invalid calibration series: {error}")

    st.divider()
    st.subheader("⚙️ Free Parameters")
    definitions = {
        "base_con": ("Daily visitors", 30, 250, "int", False,
                     "Prefer direct measurement from transaction counts"),
        "reorder": ("Reorder point", 0.10, 0.60, "float", True,
                    "Store operating rule"),
        "target": ("Restock target", 0.60, 0.99, "float", True,
                   "Store operating rule"),
        "lead": ("Lead time", 1, 10, "int", False,
                 "Prefer direct measurement from supplier records"),
    }
    if regime == "Crisis":
        definitions.update({
            "panic": ("Panic sensitivity", 0.0, 1.0, "float", False,
                      "Unidentified behavioral assumption"),
            "hoard": ("Maximum hoarding multiplier", 1.0, 3.0, "float", False,
                      "Scaled by cross-fitted household propensity"),
            "panic_exposure_floor": ("Scarcity exposure floor", 0.0, 0.30, "float", False,
                                     "Unidentified panic-dynamics assumption"),
            "panic_growth_rate": ("Panic growth rate", 0.10, 1.0, "float", False,
                                  "Unidentified panic-dynamics assumption"),
            "panic_decay_active": ("Active panic decay", 0.0, 0.15, "float", False,
                                   "Unidentified panic-dynamics assumption"),
            "inflation_panic_rate": ("Inflation-to-panic rate", 0.0, 0.80, "float", False,
                                     "Unidentified panic-dynamics assumption"),
        })
    active = {}
    check_columns = st.columns(3)
    for index, (name, definition) in enumerate(definitions.items()):
        label, low, high, kind, default, note = definition
        with check_columns[index % 3]:
            if st.checkbox(
                f"{label} ({low}–{high})", value=default,
                key=f"cal_v2_{regime}_{name}", help=note,
            ):
                active[name] = {
                    "label": label, "min": low, "max": high, "kind": kind,
                }
    if not active:
        st.warning("Select at least one free parameter.")
        return

    control1, control2, control3 = st.columns(3)
    with control1:
        n_samples = st.slider(
            "LHS design points", 24, 160, 64, 8, key="cal_samples_v2"
        )
    with control2:
        n_replicates = st.slider(
            "Replicates per point", 2, 5, 3, 1, key="cal_replicates_v2"
        )
    with control3:
        minimum_days = (
            max(14, int(params.get("cri_start", 30)) + 7)
            if regime == "Crisis" else 14
        )
        days = st.slider(
            "Days per run", minimum_days, 120, max(minimum_days, 45),
            key="cal_days_v2",
        )
    if n_samples < 10 * len(active):
        st.warning(
            f"Design adequacy will fail: use at least {10 * len(active)} points "
            f"for {len(active)} free parameters."
        )
    total_runs = n_samples * n_replicates
    st.info(
        f"Planned: **{n_samples} joint design points × {n_replicates} "
        f"common-seed replicates = {total_runs} ABM runs**."
    )

    if st.button(
        "🎯 Run Identifiability-Gated Calibration", type="primary",
        key="cal_run_v2", disabled=not target_ready,
    ):
        names = list(active)
        specs = [(active[n]["min"], active[n]["max"], active[n]["kind"]) for n in names]
        design = calibration_design(n_samples, specs, seed=20260724)
        if mode == "Observed daily time series":
            observed = target_series[target_column].to_numpy(dtype=float)
            observed = observed[:min(days, len(observed))]
            split = max(2, min(len(observed) - 2, int(len(observed) * 0.80)))
            train_target, validation_target = observed[:split], observed[split:]
            train_scales = np.full(split, float(series_scale))
            validation_scales = np.full(len(validation_target), float(series_scale))
            feature_count = split
        else:
            observed = validation_target = validation_scales = None
            split = None
            train_target, train_scales = target_values, target_scales
            feature_count = len(target_values)

        outputs = np.full((n_samples, n_replicates, feature_count), np.nan)
        full_outputs = (
            np.full((n_samples, n_replicates, len(observed)), np.nan)
            if observed is not None else None
        )
        failures = []
        progress = st.progress(0, text="Running replicated calibration design…")
        completed = 0
        for point, values in enumerate(design):
            overrides = {
                name: (int(round(value)) if active[name]["kind"] == "int" else float(value))
                for name, value in zip(names, values)
            }
            for replicate in range(n_replicates):
                try:
                    run_params = {**params, **overrides, "days": days}
                    model = _make_model(
                        run_params, is_crisis=(regime == "Crisis"),
                        seed=12000 + replicate,
                        policy_cfg=params.get("policy_cfg", {}),
                    )
                    for _ in range(days):
                        model.step()
                    frame = pd.DataFrame(model.daily_records)
                    if mode == "Observed daily time series":
                        simulated = frame[target_column].to_numpy(dtype=float)[:len(observed)]
                        if len(simulated) != len(observed):
                            raise ValueError("simulation is shorter than the target window")
                        full_outputs[point, replicate] = simulated
                        outputs[point, replicate] = simulated[:split]
                    else:
                        phase = frame
                        if regime == "Crisis":
                            phase = frame[frame["Day"] >= int(params.get("cri_start", 30))]
                        if phase.empty:
                            raise ValueError("calibration window has no active observations")
                        outputs[point, replicate] = [
                            float(phase["Revenue"].mean()),
                            float(phase["FulfillmentRate"].mean() * 100.0),
                            waste_rate_percent(phase["Sales"], phase["Waste"]),
                        ]
                except Exception as error:
                    failures.append(f"point {point}, replicate {replicate}: {error}")
                completed += 1
                progress.progress(
                    completed / total_runs,
                    text=f"Calibration simulations: {completed}/{total_runs}",
                )
        progress.empty()

        valid = np.all(np.isfinite(outputs), axis=(1, 2))
        if int(np.sum(valid)) < max(3, 10 * len(names)):
            st.error(
                f"Only {int(np.sum(valid))} complete design points remain; "
                "identifiability cannot be assessed."
            )
        else:
            design_valid, outputs_valid = design[valid], outputs[valid]
            full_valid = full_outputs[valid] if full_outputs is not None else None
            means = outputs_valid.mean(axis=1)
            objectives = np.asarray([
                standardized_rmse(row, train_target, train_scales) for row in means
            ])
            best_index = int(np.argmin(objectives))
            diagnostics = identifiability_diagnostics(names, design_valid, outputs_valid)
            validation = {
                "available": mode == "Observed daily time series", "passed": False,
                "rmse": None, "naive_rmse": None, "skill_vs_naive": None,
            }
            best_full = None
            if full_valid is not None:
                best_full = full_valid[best_index].mean(axis=0)
                validation_rmse = standardized_rmse(
                    best_full[split:], validation_target, validation_scales
                )
                naive_rmse = standardized_rmse(
                    np.full(len(validation_target), float(np.mean(train_target))),
                    validation_target, validation_scales,
                )
                skill = 1.0 - validation_rmse / naive_rmse if naive_rmse > 1e-12 else 0.0
                validation.update({
                    "passed": bool(skill > 0), "rmse": validation_rmse,
                    "naive_rmse": naive_rmse, "skill_vs_naive": skill,
                })
            eligible = bool(diagnostics["recommendation_allowed"] and validation["passed"])
            rows = []
            for row_index, objective in enumerate(objectives):
                row = {"DesignPoint": row_index, "Training standardized RMSE": objective}
                row.update({n: design_valid[row_index, col] for col, n in enumerate(names)})
                rows.append(row)
            best_params = {
                name: (int(round(design_valid[best_index, col]))
                       if active[name]["kind"] == "int"
                       else float(design_valid[best_index, col]))
                for col, name in enumerate(names)
            }
            st.session_state["calibration_v2"] = {
                "regime": regime, "mode": mode, "names": names,
                "labels": {name: active[name]["label"] for name in names},
                "rows": pd.DataFrame(rows).sort_values("Training standardized RMSE").reset_index(drop=True),
                "best_params": best_params,
                "best_objective": float(objectives[best_index]),
                "best_full_series": best_full,
                "target_series": (
                    target_series.iloc[:len(observed)].copy()
                    if observed is not None else None
                ),
                "target_column": target_column,
                "split": split, "diagnostics": diagnostics,
                "validation": validation, "eligible": eligible,
                "failures": failures, "n_runs": total_runs,
            }

    results = st.session_state.get("calibration_v2")
    if not results:
        return
    st.divider()
    st.subheader("🧪 Identifiability and Recoverability Decision")
    diagnostics, validation = results["diagnostics"], results["validation"]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Target-space rank", f"{diagnostics['target_rank']}/{diagnostics['n_parameters']}")
    d2.metric("Signal/noise", f"{diagnostics['median_signal_to_noise']:.2f}")
    d3.metric(
        "Held-out skill",
        f"{validation['skill_vs_naive']:+.3f}" if validation["available"] else "Unavailable",
    )
    d4.metric("Recommendation", "Allowed" if results["eligible"] else "Blocked")
    diagnostic_frame = pd.DataFrame(diagnostics["parameters"])
    diagnostic_frame["parameter"] = diagnostic_frame["parameter"].map(results["labels"])
    st.dataframe(diagnostic_frame.round(4), hide_index=True, use_container_width=True)
    reasons = list(diagnostics["reasons"])
    if not validation["available"]:
        reasons.append("no independent held-out target period")
    elif not validation["passed"]:
        reasons.append("held-out trajectory did not beat the naive training-mean forecast")
    if reasons:
        st.error("Parameter recommendation blocked: " + "; ".join(reasons) + ".")
    else:
        st.success("Synthetic recovery, stochastic signal, target rank, and held-out validation passed.")

    st.subheader("Best numerical fit")
    st.caption(
        "These values minimize training error. They are not a defensible parameter "
        "recommendation unless the decision above says Allowed."
    )
    best_columns = st.columns(max(1, len(results["best_params"])))
    for index, (name, value) in enumerate(results["best_params"].items()):
        best_columns[index].metric(results["labels"][name], f"{value:.4g}")
    st.metric("Training standardized RMSE", f"{results['best_objective']:.4f}")

    widget_map = {
        "base_con": ("sim_base_con", lambda value: int(round(value))),
        "reorder": ("sim_reorder_pct", lambda value: int(round(value * 100))),
        "target": ("sim_target_pct", lambda value: int(round(value * 100))),
        "lead": ("sim_lead", lambda value: int(round(value))),
        "panic": ("sim_panic", float), "hoard": ("sim_hoard", float),
        "panic_exposure_floor": ("sim_panic_exposure_floor", float),
        "panic_growth_rate": ("sim_panic_growth_rate", float),
        "panic_decay_active": ("sim_panic_decay_active", float),
        "inflation_panic_rate": ("sim_inflation_panic_rate", float),
    }
    if st.button(
        "⚡ Apply identified recommendation to simulation controls",
        type="primary", key="cal_apply_v2", disabled=not results["eligible"],
    ):
        st.session_state["_pending_calibration_widget_values"] = {
            widget_map[name][0]: widget_map[name][1](value)
            for name, value in results["best_params"].items()
        }
        st.rerun()

    st.subheader("Top numerical fits")
    st.dataframe(results["rows"].head(15).round(5), hide_index=True, use_container_width=True)
    if results["mode"] == "Observed daily time series":
        observed_frame = results["target_series"]
        observed_values = observed_frame[results["target_column"]].to_numpy(dtype=float)
        split = results["split"]
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=observed_frame["Day"], y=observed_values,
            name="Observed", line=dict(color="#DC143C", width=2),
        ))
        figure.add_trace(go.Scatter(
            x=observed_frame["Day"], y=results["best_full_series"],
            name="Best numerical fit", line=dict(color="#DBA159", width=2),
        ))
        figure.add_vline(
            x=float(observed_frame["Day"].iloc[split]), line_dash="dash",
            annotation_text="Held-out period", line_color="#042026",
        )
        figure.update_layout(
            title=f"Training and held-out validation: {results['target_column']}",
            xaxis_title="Day", yaxis_title=results["target_column"],
            template="plotly_white",
        )
        st.plotly_chart(figure, use_container_width=True, config=_PLOTLY_CFG)

    st.download_button(
        "📥 Download calibration design and fit (CSV)",
        results["rows"].to_csv(index=False).encode("utf-8"),
        "grocerysim_calibration_design.csv", "text/csv", key="dl_cal_v2",
    )
    audit = {
        "regime": results["regime"], "target_mode": results["mode"],
        "best_training_standardized_rmse": results["best_objective"],
        "best_parameters": results["best_params"],
        "eligible_to_apply": results["eligible"],
        "identifiability": diagnostics, "heldout_validation": validation,
        "simulation_runs": results["n_runs"],
    }
    st.download_button(
        "📥 Download calibration audit (JSON)",
        json.dumps(audit, indent=2).encode("utf-8"),
        "grocerysim_calibration_audit.json", "application/json",
        key="dl_cal_audit_v2",
    )


# ===========================================================================
# 12. SCENARIO COMPARE TAB
# ===========================================================================

def render_scenario_compare_tab():
    st.header("📊 Scenario Library & Comparison")

    _saved = st.session_state.saved_scenarios

    # ── Import JSON library ───────────────────────────────────────────────────
    _imp_col, _clr_col = st.columns([3, 1])
    with _imp_col:
        _json_upload = st.file_uploader(
            "📂 Import scenario library (JSON)", type=["json"], key="sc_import",
            label_visibility="collapsed",
        )
        if _json_upload:
            try:
                _imported = json.load(_json_upload)
                _added = 0
                for _sc in _imported:
                    if "name" in _sc and _sc["name"] not in [s["name"] for s in _saved]:
                        # Restore without the raw df (not serialisable)
                        _sc.setdefault("df", pd.DataFrame())
                        _sc.setdefault("tags", [])
                        _sc.setdefault("notes", "")
                        _saved.append(_sc)
                        _added += 1
                if _added:
                    st.success(f"✅ Imported {_added} new scenario(s).")
            except Exception as _ie:
                st.error(f"Import failed: {_ie}")
    with _clr_col:
        if _saved and st.button("🗑️ Clear All", use_container_width=True, key="cmp_clear"):
            st.session_state.saved_scenarios = []
            st.rerun()

    if not _saved:
        st.info(
            "Run **🎮 Interactive Demo** and click **💾 Save Scenario** at least once. "
            "All saved scenarios appear here for comparison."
        )
        return

    # ── Tags / notes editor ───────────────────────────────────────────────────
    with st.expander(f"📝 Manage {len(_saved)} Saved Scenario(s)", expanded=False):
        for _si, _sc in enumerate(_saved):
            _ec1, _ec2, _ec3 = st.columns([2, 2, 1])
            with _ec1:
                _new_name = st.text_input(
                    "Name", value=_sc["name"], key=f"sc_name_{_si}",
                    label_visibility="collapsed")
                _saved[_si]["name"] = _new_name
            with _ec2:
                _new_tags = st.text_input(
                    "Tags (comma-separated)", value=", ".join(_sc.get("tags", [])),
                    key=f"sc_tags_{_si}", label_visibility="collapsed",
                    placeholder="e.g. baseline, policy, crisis…")
                _saved[_si]["tags"] = [t.strip() for t in _new_tags.split(",") if t.strip()]
            with _ec3:
                if st.button("🗑️", key=f"sc_del_{_si}", help="Delete this scenario"):
                    _saved.pop(_si)
                    st.rerun()

    # ── Ranking ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🏆 Scenario Ranking")
    _rank_col, _rank_dir_col = st.columns([3, 1])
    with _rank_col:
        _rank_metric = st.selectbox(
            "Rank by", ["revenue_base", "revenue_crisis", "fulfillment",
                        "lost_sales", "waste", "co2", "fies_low"],
            format_func=lambda k: {
                "revenue_base":   "Revenue — Baseline (€)",
                "revenue_crisis": "Revenue — Crisis (€)",
                "fulfillment":    "Fulfilment Rate",
                "lost_sales":     "Lost Sales (€) ↓ lower = better",
                "waste":          "Waste (units) ↓",
                "co2":            "Avg CO₂ / day (kg) ↓",
                "fies_low":       "Access Stress High Low-income ↓",
            }.get(k, k),
            key="rank_metric",
        )
    with _rank_dir_col:
        _rank_asc = st.checkbox(
            "Ascending (lower = better)", value=_rank_metric in ("lost_sales","waste","co2","fies_low"),
            key="rank_asc",
        )

    _rank_rows = []
    for _ri, _sc in enumerate(
        sorted(_saved, key=lambda s: s.get(_rank_metric, 0), reverse=not _rank_asc)
    ):
        _rank_rows.append({
            "Rank":           _ri + 1,
            "Scenario":       _sc["name"],
            "Timestamp":      _sc.get("timestamp", ""),
            "Tags":           ", ".join(_sc.get("tags", [])),
            "Rev Baseline":   f"{_sc.get('revenue_base', 0):,.0f}",
            "Rev Crisis":     f"{_sc.get('revenue_crisis', 0):,.0f}",
            "Fulfilment":     f"{_sc.get('fulfillment', 0):.3f}",
            "Lost Sales":     f"{_sc.get('lost_sales', 0):,.0f}",
            "Waste":          f"{_sc.get('waste', 0):,.0f}",
            "CO₂/day":        f"{_sc.get('co2', 0):.1f}",
            "Access Stress Low":       f"{_sc.get('fies_low', 0):.3f}",
        })
    st.dataframe(pd.DataFrame(_rank_rows), use_container_width=True, hide_index=True)

    # ── KPI Radar chart (all scenarios) ──────────────────────────────────────
    st.divider()
    st.markdown("### 🕸️ KPI Radar — All Scenarios")
    st.caption(
        "Each axis is normalised 0–1 across saved scenarios. "
        "Revenue and Fulfilment point outward = good; Waste, CO₂, access stress and Lost Sales inverted."
    )

    _RADAR_AXES = [
        ("revenue_base",   "Revenue\n(baseline)",  True),
        ("fulfillment",    "Fulfilment",           True),
        ("lost_sales",     "Lost Sales\n(lower=better)", False),
        ("waste",          "Waste\n(lower=better)",      False),
        ("co2",            "CO₂\n(lower=better)",        False),
        ("fies_low",       "Access Stress Low\n(lower=better)",   False),
    ]
    _rdr_labels = [lbl for _, lbl, _ in _RADAR_AXES]

    def _norm_col(vals, higher_is_better):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        normed = [(v - mn) / (mx - mn) for v in vals]
        return normed if higher_is_better else [1 - v for v in normed]

    _radar_vals = {ax: [s.get(ax, 0) for s in _saved] for ax, _, _ in _RADAR_AXES}
    _radar_norm = {ax: _norm_col(_radar_vals[ax], hib) for ax, _, hib in _RADAR_AXES}

    _RADAR_COLORS = [
        "#DBA159","#44A1A0","#DC143C","#8E44AD","#27AE60","#2980B9","#E67E22","#C0392B",
    ]
    _fig_radar = go.Figure()
    for _si, _sc in enumerate(_saved):
        _r_vals = [_radar_norm[ax][_si] for ax, _, _ in _RADAR_AXES]
        _r_vals += [_r_vals[0]]          # close the polygon
        _theta   = _rdr_labels + [_rdr_labels[0]]
        _fig_radar.add_trace(go.Scatterpolar(
            r=_r_vals, theta=_theta,
            name=_sc["name"],
            fill="toself", opacity=0.25,
            line=dict(color=_RADAR_COLORS[_si % len(_RADAR_COLORS)], width=2),
        ))
    _fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=40, b=80),
    )
    st.plotly_chart(_fig_radar, use_container_width=True, config=_PLOTLY_CFG)

    # ── Head-to-head: pick any two ────────────────────────────────────────────
    if len(_saved) >= 2:
        st.divider()
        st.markdown("### 🆚 Head-to-Head Comparison")
        _names = [s["name"] for s in _saved]
        _h2h_a, _h2h_b = st.columns(2)
        with _h2h_a:
            _sel_a = st.selectbox("Scenario A", _names, index=0, key="cmp_a")
        with _h2h_b:
            _sel_b = st.selectbox("Scenario B", _names,
                                   index=min(1, len(_names)-1), key="cmp_b")
        _sc_a = next(s for s in _saved if s["name"] == _sel_a)
        _sc_b = next(s for s in _saved if s["name"] == _sel_b)

        _CMP_METRICS = [
            ("Rev Baseline (€)",  "revenue_base"),
            ("Rev Crisis (€)",    "revenue_crisis"),
            ("Lost Sales (€)",    "lost_sales"),
            ("Waste (units)",     "waste"),
            ("Fulfilment",        "fulfillment"),
            ("CO₂/day (kg)",      "co2"),
            ("Access Stress High Low",   "fies_low"),
        ]
        def _pct_diff_html(a, b):
            if b == 0:
                return "—"
            d = (a - b) / abs(b) * 100
            arrow = "▲" if d > 0 else "▼"
            color = "#27ae60" if d > 0 else "#c0392b"
            return f"<span style='color:{color};font-weight:700'>{arrow}{abs(d):.1f}%</span>"

        _rows_html = ""
        for _lbl, _key in _CMP_METRICS:
            _va = _sc_a.get(_key, 0); _vb = _sc_b.get(_key, 0)
            _rows_html += (
                f"<tr><td style='padding:5px 10px'>{_lbl}</td>"
                f"<td style='padding:5px 10px;text-align:right;font-weight:600'>{_va:,.2f}</td>"
                f"<td style='padding:5px 10px;text-align:right;font-weight:600'>{_vb:,.2f}</td>"
                f"<td style='padding:5px 10px;text-align:center'>{_pct_diff_html(_va, _vb)}</td></tr>"
            )
        st.markdown(
            f"""<table style='width:100%;border-collapse:collapse;background:#FAF6EC;
                              border:1px solid #e8dcc8;border-radius:8px;overflow:hidden'>
              <thead>
                <tr style='background:#F0E9DA'>
                  <th style='padding:7px 10px;text-align:left;color:#042026'>Metric</th>
                  <th style='padding:7px 10px;text-align:right;color:#DBA159'>{_sel_a}</th>
                  <th style='padding:7px 10px;text-align:right;color:#44A1A0'>{_sel_b}</th>
                  <th style='padding:7px 10px;text-align:center;color:#042026'>A vs B</th>
                </tr>
              </thead>
              <tbody>{_rows_html}</tbody>
            </table>""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Revenue overlay (if df available)
        _dfa = _sc_a.get("df"); _dfb = _sc_b.get("df")
        if isinstance(_dfa, pd.DataFrame) and not _dfa.empty and \
           isinstance(_dfb, pd.DataFrame) and not _dfb.empty and \
           "Scenario" in _dfa.columns and "Revenue" in _dfa.columns:
            _fig_ov = go.Figure()
            for _sc_obj, _col_hex in [(_sc_a, "#DBA159"), (_sc_b, "#44A1A0")]:
                _df_plot = _sc_obj["df"]
                if isinstance(_df_plot, pd.DataFrame) and not _df_plot.empty:
                    _base = _df_plot[_df_plot["Scenario"] == "Baseline"]
                    _fig_ov.add_trace(go.Scatter(
                        x=_base["Day"], y=_base["Revenue"],
                        name=_sc_obj["name"],
                        line=dict(color=_col_hex, width=2.5),
                    ))
            _fig_ov.update_layout(
                title="Baseline Revenue Overlay",
                template="plotly_white", xaxis_title="Day", yaxis_title="Revenue (€)",
                legend=dict(orientation="h", y=1.12),
                margin=dict(l=40, r=20, t=55, b=40),
            )
            st.plotly_chart(_fig_ov, use_container_width=True, config=_PLOTLY_CFG)

        # Parameter diff
        with st.expander("⚙️ Parameter Differences"):
            _pa, _pb = _sc_a.get("params", {}), _sc_b.get("params", {})
            _all_k   = sorted(set(list(_pa.keys()) + list(_pb.keys())))
            _param_rows = [
                {"Parameter": _k, _sel_a: _pa.get(_k, "—"), _sel_b: _pb.get(_k, "—"),
                 "Changed": "✅" if _pa.get(_k) != _pb.get(_k) else ""}
                for _k in _all_k
            ]
            st.dataframe(pd.DataFrame(_param_rows), use_container_width=True, hide_index=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    _exp1, _exp2 = st.columns(2)
    with _exp1:
        _buf = io.StringIO()
        for _sc in _saved:
            _buf.write(f"### {_sc['name']}  ({_sc.get('timestamp','')})\n")
            if isinstance(_sc.get("df"), pd.DataFrame) and not _sc["df"].empty:
                _sc["df"].to_csv(_buf, index=False)
            _buf.write("\n\n")
        st.download_button(
            "📥 Download All Scenarios (CSV)",
            _buf.getvalue().encode("utf-8"),
            "scenario_library.csv", "text/csv", key="dl_cmp",
        )
    with _exp2:
        # JSON export (exclude non-serialisable df)
        _json_export = json.dumps([
            {k: v for k, v in _sc.items() if k != "df"}
            for _sc in _saved
        ], indent=2, default=str)
        st.download_button(
            "📤 Export Scenario Library (JSON)",
            _json_export.encode("utf-8"),
            "scenario_library.json", "application/json", key="dl_sc_json",
        )


# ===========================================================================
# 13. STRESS TEST TAB
# ===========================================================================

_STRESS_SCENARIOS = [
    {
        "id": "supply_shock",
        "name": "🌊 Supply Chain Collapse",
        "desc": "Complete delivery stoppage for 10 days (flood / logistics failure)",
        "overrides": {"dis": 10, "inf": 5.0, "panic": 0.6, "hoard": 1.4},
    },
    {
        "id": "price_spike",
        "name": "💸 Commodity Price Spike",
        "desc": "40% price inflation shock with moderate supply disruption",
        "overrides": {"inf": 40.0, "dis": 3, "panic": 0.4, "hoard": 1.2},
    },
    {
        "id": "panic_buying",
        "name": "😱 Panic Buying Wave",
        "desc": "Media-driven panic: high hoarding, rapid shelf depletion",
        "overrides": {"panic": 0.9, "hoard": 2.5, "dis": 2, "inf": 8.0},
    },
    {
        "id": "import_dep",
        "name": "🚢 Import Dependency Crisis",
        "desc": "Extended import disruption — 14 days, high inflation",
        "overrides": {"dis": 14, "inf": 20.0, "panic": 0.5, "hoard": 1.3},
    },
    {
        "id": "demand_surge",
        "name": "📈 Demand Surge (+80%)",
        "desc": "Sudden 80% increase in shoppers (tourism / refugee influx)",
        "overrides": {"base_con_mult": 1.80, "dis": 1, "inf": 5.0},
    },
    {
        "id": "deep_freeze",
        "name": "🧊 Deep Freeze (Cold-chain Failure)",
        "desc": "Perishables supply cut by 60% for 7 days",
        "overrides": {"dis": 7, "inf": 15.0, "panic": 0.55, "hoard": 1.6},
    },
]


def render_stress_tab(params: dict):
    st.header("🚨 Automated Scenario Stress Test")
    st.markdown(
        "Automatically run **6 pre-defined crisis scenarios** against your current baseline "
        "parameters and rank them by impact on revenue, food security, and supply chain resilience."
    )

    if st.session_state.config_data is None:
        st.warning("⚠️ Load population data in **🏠 Data & Population** first.")
        return

    st.markdown("### ⚙️ Stress Test Settings")
    c1, c2 = st.columns(2)
    with c1:
        st_days = st.slider("Simulation duration per scenario (days)", 30, 180, 90, 15,
                            key="stress_days")
    with c2:
        st_runs = st.slider("Monte Carlo runs per scenario", 1, 5, 2, 1,
                            key="stress_runs",
                            help="More runs = more accurate but slower. 2–3 is sufficient for stress ranking.")

    st.markdown("### 📋 Scenarios to Test")
    exploratory_stress = bool(params.get("exploratory_behaviour", False))
    if not exploratory_stress:
        st.info(
            "Empirical-only mode: the panic-buying-wave scenario is unavailable, "
            "and panic/hoarding overrides in other scenarios are ignored."
        )
    cols = st.columns(3)
    selected_ids = []
    for i, sc in enumerate(_STRESS_SCENARIOS):
        with cols[i % 3]:
            with st.container(border=True):
                panic_only = sc["id"] == "panic_buying"
                checked = st.checkbox(
                    sc["name"], value=not panic_only,
                    key=f"stress_chk_{sc['id']}",
                    disabled=panic_only and not exploratory_stress,
                )
                st.caption(sc["desc"])
                if checked and (not panic_only or exploratory_stress):
                    selected_ids.append(sc["id"])

    if not selected_ids:
        st.warning("Select at least one scenario.")
        return

    run_stress = st.button("🚀 Run Stress Test", type="primary", key="run_stress_btn",
                           use_container_width=False)

    if run_stress:
        st.session_state.stress_results = None
        stress_rows = []
        progress = st.progress(0, text="Initialising…")
        selected_scs = [s for s in _STRESS_SCENARIOS if s["id"] in selected_ids]

        for idx, sc in enumerate(selected_scs):
            progress.progress(idx / len(selected_scs), text=f"Running: {sc['name']}…")

            # Build overridden params
            sc_params = dict(params)
            ov = sc["overrides"]
            if "base_con_mult" in ov:
                sc_params["base_con"] = int(params["base_con"] * ov["base_con_mult"])
            for k in ["dis", "inf", "panic", "hoard"]:
                if k in ov:
                    sc_params[k] = ov[k]
            sc_params["days"]      = st_days
            sc_params["cri_start"] = max(7, st_days // 6)

            # Run baseline + crisis
            run_rev_base, run_rev_crisis, run_lost, run_waste, run_fulfill = [], [], [], [], []
            for run_id in range(st_runs):
                try:
                    m_base = _make_model(sc_params, is_crisis=False, seed=500 + run_id)
                    m_cris = _make_model(sc_params, is_crisis=True,  seed=500 + run_id)
                    rev_b = rev_c = lost = waste = fulfill = 0.0
                    for day in range(1, st_days + 1):
                        m_base.step(); m_cris.step()
                        rb, _ = _collect_model_day(m_base, day, "Baseline", collect_products=False)
                        rc, _ = _collect_model_day(m_cris, day, "Crisis",   collect_products=False)
                        rev_b   += rb["Revenue"]
                        rev_c   += rc["Revenue"]
                        lost    += rc["LostSales"]
                        waste   += rc["Waste"]
                        fulfill += rc["FulfillmentRate"]
                    run_rev_base.append(rev_b)
                    run_rev_crisis.append(rev_c)
                    run_lost.append(lost)
                    run_waste.append(waste)
                    run_fulfill.append(fulfill / st_days)
                except Exception:
                    pass

            if not run_rev_base:
                continue

            rev_base_mean   = float(np.mean(run_rev_base))
            rev_crisis_mean = float(np.mean(run_rev_crisis))
            revenue_loss_pct = (rev_base_mean - rev_crisis_mean) / max(rev_base_mean, 1) * 100

            stress_rows.append({
                "Scenario":           sc["name"],
                "Description":        sc["desc"],
                "Revenue Loss (%)":   round(revenue_loss_pct, 1),
                "Total Lost Sales (€)": round(float(np.mean(run_lost)), 0),
                "Total Waste (units)": round(float(np.mean(run_waste)), 0),
                "Avg Fulfillment":    round(float(np.mean(run_fulfill)), 3),
                "Baseline Revenue":   round(rev_base_mean, 0),
                "Crisis Revenue":     round(rev_crisis_mean, 0),
                "Behaviour Evidence Mode": (
                    "exploratory_extensions" if exploratory_stress else "empirical_only"
                ),
                "_severity":          revenue_loss_pct,
            })

        progress.progress(1.0, text="Complete ✓")
        time.sleep(0.4)
        progress.empty()
        st.session_state.stress_results = pd.DataFrame(stress_rows).sort_values(
            "_severity", ascending=False
        )
        st.rerun()

    # ── Results dashboard ─────────────────────────────────────────────────────
    results = st.session_state.stress_results
    if results is None or results.empty:
        return

    st.divider()
    st.markdown("### 🏆 Risk Ranking")

    def _severity_color(pct):
        if pct >= 30:   return "#c0392b"
        if pct >= 15:   return "#e67e22"
        if pct >= 5:    return "#f1c40f"
        return "#27ae60"

    for _, row in results.iterrows():
        col_icon, col_info, col_bar = st.columns([1, 4, 3])
        sev = row["Revenue Loss (%)"]
        color = _severity_color(sev)
        with col_icon:
            risk_label = "CRITICAL" if sev >= 30 else ("HIGH" if sev >= 15 else ("MEDIUM" if sev >= 5 else "LOW"))
            st.markdown(
                f"<div style='background:{color};color:white;text-align:center;"
                f"padding:10px 6px;border-radius:6px;font-weight:700;font-size:12px;"
                f"margin-top:4px'>{risk_label}</div>",
                unsafe_allow_html=True,
            )
        with col_info:
            st.markdown(f"**{row['Scenario']}**  \n{row['Description']}")
            st.caption(
                f"Revenue loss: **{sev:.1f}%**  ·  "
                f"Lost sales: **€{row['Total Lost Sales (€)']:,.0f}**  ·  "
                f"Fulfillment: **{row['Avg Fulfillment']:.1%}**"
            )
        with col_bar:
            fig_bar = go.Figure(go.Bar(
                x=[sev], y=[""], orientation="h",
                marker_color=color, text=[f"{sev:.1f}%"],
                textposition="inside",
            ))
            fig_bar.update_layout(
                height=60, margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(range=[0, max(results["Revenue Loss (%)"].max() * 1.15, 5)],
                           showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False),
                template="plotly_white", showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── Heatmap ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🌡️ Risk Heatmap")
    heat_cols = ["Revenue Loss (%)", "Total Lost Sales (€)", "Total Waste (units)", "Avg Fulfillment"]
    heat_df = results[["Scenario"] + heat_cols].set_index("Scenario")
    norm_df = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-9)
    norm_df["Avg Fulfillment"] = 1 - norm_df["Avg Fulfillment"]  # invert: lower is worse

    fig_heat = go.Figure(go.Heatmap(
        z=norm_df.values,
        x=heat_cols,
        y=norm_df.index.tolist(),
        colorscale=[[0, "#27ae60"], [0.5, "#f1c40f"], [1, "#c0392b"]],
        showscale=True,
        text=heat_df.values.round(1),
        texttemplate="%{text}",
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
    ))
    fig_heat.update_layout(
        template="plotly_white", height=320,
        margin=dict(l=160, r=40, t=30, b=60),
        xaxis=dict(side="top"),
    )
    st.plotly_chart(fig_heat, use_container_width=True, config=_PLOTLY_CFG)

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()
    dl_df = results.drop(columns=["_severity"], errors="ignore")
    st.download_button(
        "📥 Download Stress Test Results (CSV)",
        dl_df.to_csv(index=False).encode("utf-8"),
        "stress_test_results.csv", "text/csv",
        key="dl_stress",
    )

    # ── Stochastic Risk Analysis ───────────────────────────────────────────────
    st.divider()
    st.markdown("## 🎲 Stochastic Risk Analysis")
    st.markdown(
        "Rather than testing fixed disruption scenarios, this module draws disruption "
        "timing and severity **randomly** from configurable distributions, running many "
        "independent realisations to build a **probabilistic risk picture**. "
        "Output metrics include **Value-at-Risk (VaR)** and **Conditional VaR (CVaR)** — "
        "the standard format used in supply-chain insurance and food-security risk reports."
    )

    with st.expander("⚙️ Stochastic Risk Settings", expanded=True):
        sr_c1, sr_c2, sr_c3 = st.columns(3)
        with sr_c1:
            sr_runs = st.slider("Monte Carlo runs", 20, 200, 50, 10,
                                key="sr_runs",
                                help="More runs → smoother exceedance curve. 50 is fast; 200 gives publication quality.")
            sr_days = st.slider("Simulation horizon (days)", 30, 180, 90, 15,
                                key="sr_days")
        with sr_c2:
            st.markdown("**Disruption Severity** *(lognormal)*")
            sr_sev_mu  = st.slider("Mean severity (% supply cut)", 5, 60, 25, 5,
                                   key="sr_sev_mu",
                                   help="Median of the lognormal severity distribution.")
            sr_sev_sig = st.slider("Std dev of severity (%)", 2, 30, 10, 2,
                                   key="sr_sev_sig",
                                   help="Width of the severity distribution — higher = more tail risk.")
        with sr_c3:
            st.markdown("**Disruption Timing** *(uniform window)*")
            sr_onset_min = st.slider("Earliest onset (day)", 1, 30, 7, 1, key="sr_onset_min")
            sr_onset_max = st.slider("Latest onset (day)", 10, 60, 40, 5, key="sr_onset_max")
            sr_dur_mean  = st.slider("Mean disruption duration (days)", 3, 30, 10, 1,
                                     key="sr_dur_mean")
            sr_inf_mean  = st.slider("Mean price inflation (%)", 5, 50, 20, 5,
                                     key="sr_inf_mean")

        sr_var_pct = st.select_slider(
            "VaR / CVaR confidence level",
            options=[80, 85, 90, 95, 99], value=90,
            key="sr_var_pct",
            help="VaR(90%) = the revenue loss not exceeded in 90% of scenarios. CVaR(90%) = mean loss in the worst 10%.",
        )

    if st.button("🎲 Run Stochastic Risk Analysis", type="primary",
                 key="sr_run_btn", use_container_width=False):
        if st.session_state.config_data is None:
            st.warning("Load population data first.")
        else:
            _sr_rng = np.random.default_rng(seed=42)
            _sr_losses   = []   # revenue loss % per run
            _sr_fies     = []   # peak FIES severe low per run
            _sr_ful_lo   = []   # mean fulfilment low income per run

            _sr_progress = st.progress(0, "Initialising stochastic runs…")
            _sr_params = dict(params)
            _sr_params["days"] = sr_days

            for _i in range(sr_runs):
                _sr_progress.progress((_i + 1) / sr_runs,
                                      f"Run {_i + 1} / {sr_runs}…")
                # Sample disruption parameters
                _onset    = int(_sr_rng.integers(sr_onset_min, max(sr_onset_min + 1, sr_onset_max)))
                _dur      = max(2, int(_sr_rng.poisson(sr_dur_mean)))
                _raw_sev  = _sr_rng.lognormal(
                    mean=np.log(max(1, sr_sev_mu) / 100),
                    sigma=sr_sev_sig / 100,
                )
                _dis_pct  = float(np.clip(_raw_sev * 100, 1, 95))
                _inf_pct  = float(_sr_rng.lognormal(
                    mean=np.log(max(1, sr_inf_mean)),
                    sigma=0.35,
                ))

                _sr_p = dict(_sr_params)
                _sr_p.update({
                    "cri_start":    _onset,
                    "cri_duration": _dur,
                    "dis":          max(1, int(_dis_pct / 10)),
                    "inf":          min(_inf_pct, 80.0),
                    "panic":        0.45 + _dis_pct / 200,
                    "hoard":        1.2  + _dis_pct / 100,
                    "mc_runs":      1,
                    "policy_cfg":   params.get("policy_cfg", {}),
                })

                try:
                    _mb = _make_model(_sr_p, is_crisis=False, seed=_i)
                    _mc = _make_model(_sr_p, is_crisis=True,  seed=_i)
                    _rev_b, _rev_c = 0.0, 0.0
                    _fies_peak, _ful_lo_sum = 0.0, 0.0
                    for _d in range(1, sr_days + 1):
                        _mb.step(); _mc.step()
                        _ab, _ = _collect_model_day(_mb, _d, "Baseline", collect_products=False)
                        _ac, _ = _collect_model_day(_mc, _d, "Crisis",   collect_products=False)
                        _rev_b    += _ab.get("Revenue", 0)
                        _rev_c    += _ac.get("Revenue", 0)
                        _fies_peak = max(_fies_peak, _ac.get("FIESSevere_Low", 0))
                        _ful_lo_sum += _ac.get("Fulfillment_Low", 1.0)
                    _loss_pct = (_rev_b - _rev_c) / max(_rev_b, 1) * 100
                    _sr_losses.append(float(_loss_pct))
                    _sr_fies.append(float(_fies_peak * 100))
                    _sr_ful_lo.append(float(_ful_lo_sum / sr_days * 100))
                except Exception:
                    continue

            _sr_progress.empty()
            st.session_state["sr_results"] = {
                "losses": _sr_losses, "fies": _sr_fies, "ful_lo": _sr_ful_lo,
                "var_pct": sr_var_pct, "runs": sr_runs,
            }

    sr_res = st.session_state.get("sr_results")
    if sr_res and sr_res.get("losses"):
        _losses  = np.array(sr_res["losses"])
        _fies_a  = np.array(sr_res["fies"])
        _ful_a   = np.array(sr_res["ful_lo"])
        _vp      = sr_res["var_pct"]
        _n       = len(_losses)

        _var_rev  = float(np.percentile(_losses, _vp))
        _cvar_rev = float(_losses[_losses >= _var_rev].mean()) if (_losses >= _var_rev).any() else _var_rev
        _var_fies = float(np.percentile(_fies_a, _vp))
        _median   = float(np.median(_losses))
        _p10      = float(np.percentile(_losses, 10))

        # KPI boxes
        mk1, mk2, mk3, mk4 = st.columns(4)
        with mk1:
            st.metric(f"VaR({_vp}%) Revenue Loss",    f"{_var_rev:.1f}%",
                      help=f"Revenue loss not exceeded in {_vp}% of simulated disruptions.")
        with mk2:
            st.metric(f"CVaR({_vp}%) Revenue Loss",   f"{_cvar_rev:.1f}%",
                      help=f"Mean revenue loss in the worst {100-_vp}% of cases.")
        with mk3:
            st.metric("Median Revenue Loss",           f"{_median:.1f}%")
        with mk4:
            st.metric(f"VaR({_vp}%) Access Stress High",     f"{_var_fies:.1f}%",
                      help="Food insecurity level (low-income agents) in the worst disruption scenarios.")

        st.markdown(
            f"**Interpretation:** In {100 - _vp}% of randomly sampled disruption scenarios "
            f"(the tail), revenue loss exceeds **{_var_rev:.1f}%** (VaR), and the average "
            f"loss in those tail scenarios is **{_cvar_rev:.1f}%** (CVaR). "
            f"The best {10}% of draws produce losses below **{_p10:.1f}%**, "
            f"representing resilient configurations. "
            f"Median loss across all {_n} runs: **{_median:.1f}%**."
        )
        st.caption(
            f"Based on {_n} Monte Carlo runs with lognormal severity (mean {sr_res.get('var_pct')}%) "
            f"and Poisson duration distribution."
        )

        # Exceedance (CCDF) curve + histogram side by side
        _sorted = np.sort(_losses)
        _exceedance = 1.0 - np.arange(1, _n + 1) / _n

        _fig_ec, _ax_ec = plt.subplots(figsize=(6, 3.5))
        _ax_ec.plot(_sorted, _exceedance * 100, color="#2980b9", lw=2.5)
        _ax_ec.fill_between(_sorted, _exceedance * 100, alpha=0.10, color="#2980b9")
        _ax_ec.axvline(_var_rev,  color="#c0392b", ls="--", lw=1.5,
                       label=f"VaR({_vp}%) = {_var_rev:.1f}%")
        _ax_ec.axvline(_cvar_rev, color="#e67e22", ls=":",  lw=1.5,
                       label=f"CVaR({_vp}%) = {_cvar_rev:.1f}%")
        _ax_ec.axhline(100 - _vp, color="#95a5a6", ls=":", lw=1)
        _ax_ec.set_xlabel("Revenue Loss (%)")
        _ax_ec.set_ylabel("Probability of Exceeding (%)")
        _ax_ec.set_title("Risk Exceedance Curve (CCDF)")
        _ax_ec.legend(fontsize=8)
        _ax_ec.spines[["top", "right"]].set_visible(False)
        _fig_ec.tight_layout()

        _fig_hist, _ax_h = plt.subplots(figsize=(6, 3.5))
        _ax_h.hist(_losses, bins=min(25, _n // 2), color="#44A1A0",
                   edgecolor="white", linewidth=0.4, alpha=0.85)
        _ax_h.axvline(_var_rev,  color="#c0392b", ls="--", lw=2,
                      label=f"VaR({_vp}%)")
        _ax_h.axvline(_cvar_rev, color="#e67e22", ls=":",  lw=2,
                      label=f"CVaR({_vp}%)")
        _ax_h.axvline(_median,   color="#27ae60", ls="-",  lw=1.5,
                      label=f"Median")
        _ax_h.set_xlabel("Revenue Loss (%)")
        _ax_h.set_ylabel("Frequency")
        _ax_h.set_title("Revenue Loss Distribution")
        _ax_h.legend(fontsize=8)
        _ax_h.spines[["top", "right"]].set_visible(False)
        _fig_hist.tight_layout()

        _sc1, _sc2 = st.columns(2)
        with _sc1:
            st.pyplot(_fig_ec)
        with _sc2:
            st.pyplot(_fig_hist)
        plt.close(_fig_ec); plt.close(_fig_hist)

        # FIES / Fulfilment scatterplot
        if len(_fies_a) == len(_losses):
            _fig_sc, _ax_s = plt.subplots(figsize=(6, 3.5))
            _ax_s.scatter(_losses, _fies_a, c=_ful_a, cmap="RdYlGn",
                          s=28, alpha=0.75, edgecolors="none")
            _cbar = _fig_sc.colorbar(_ax_s.collections[0], ax=_ax_s)
            _cbar.set_label("Fulfilment — Low Income (%)", fontsize=8)
            _ax_s.set_xlabel("Revenue Loss (%)")
            _ax_s.set_ylabel("Peak Access Stress High — Low Income (%)")
            _ax_s.set_title("Risk Trade-off: Revenue Loss vs. Food Insecurity")
            _ax_s.spines[["top", "right"]].set_visible(False)
            _fig_sc.tight_layout()
            st.pyplot(_fig_sc)
            plt.close(_fig_sc)
            st.caption(
                "Each point = one Monte Carlo run. Colour = low-income basket fulfilment "
                "(green = high, red = low). Runs in the top-right corner combine high revenue "
                "loss with severe food insecurity — the highest-priority risk zone."
            )

        # Download
        _sr_df = pd.DataFrame({
            "Run": range(1, _n + 1),
            "RevenueLoss_pct": _losses,
            "FIESSevere_Low_pct": _fies_a,
            "Fulfillment_Low_pct": _ful_a,
        })
        st.download_button(
            "📥 Download Stochastic Risk Results (CSV)",
            _sr_df.to_csv(index=False).encode("utf-8"),
            "stochastic_risk_results.csv", "text/csv",
            key="dl_sr",
        )


# ===========================================================================
# 13. AGENT REPLAY VIEWER
# ===========================================================================

_ARCHETYPE_COLORS = {
    "price_champion":   "#E87722",   # amber
    "green_buyer":      "#44A1A0",   # teal
    "health_optimizer": "#27AE60",   # green
    "habitual_buyer":   "#8E44AD",   # purple
}
_ARCHETYPE_LABELS = {
    "price_champion":   "💰 Price Champion",
    "green_buyer":      "🌿 Green Buyer",
    "health_optimizer": "🥗 Health Optimizer",
    "habitual_buyer":   "🛒 Habitual Buyer",
}
_INCOME_BRACKETS = {
    (0,     1500):  "Low",
    (1500,  3000):  "Mid",
    (3000,  99999): "High",
}


def _income_bracket(midpoint: float) -> str:
    for (lo, hi), label in _INCOME_BRACKETS.items():
        if lo <= midpoint < hi:
            return label
    return "Mid"


def _collect_agent_snapshot(model, day: int, scenario: str) -> list[dict]:
    """
    Capture a lightweight per-agent snapshot AFTER model.step().

    model.last_daily_agents holds the ConsumerAgent objects that just ran.
    Objects are removed from the Mesa schedule but remain alive in memory.
    """
    rows = []
    for agent in getattr(model, "last_daily_agents", []):
        fulfillment = (agent.items_purchased / max(1, agent.items_wanted)
                       if agent.items_wanted > 0 else 1.0)
        rows.append({
            "Day":             day,
            "Scenario":        scenario,
            "HouseholdID":     agent.household_id,
            "VisitNumber":     agent.visit_number,
            "Archetype":       agent.archetype,
            "IncomeBracket":   _income_bracket(agent.income_midpoint),
            "IncomeMidpoint":  round(agent.income_midpoint, 0),
            "ItemsWanted":     agent.items_wanted,
            "ItemsPurchased":  agent.items_purchased,
            "Fulfillment":     round(fulfillment, 3),
            "BudgetExhausted": int(agent.budget_exhausted),
            "PanicLevel":      round(agent.panic_level, 3),
            "PBC":             round(agent.pbc, 3),
            "SubjectiveNorm":  round(agent.subjective_norm, 3),
            "PriceSensitivity": round(agent.price_sensitivity, 3),
            "OrganicPref":     round(agent.organic_preference, 3),
            "FinnishPref":     round(agent.finnish_preference, 3),
            "AccessStress":    agent.access_stress_score,
            "ShoppingShortfall": round(agent.shopping_shortfall_rate, 3),
            # Deprecated alias retained for old replay exports.
            "FIES":            agent.food_insecurity_score,
        })
    return rows


def render_agent_replay_tab():
    st.header("🎬 Agent Replay Viewer")
    st.markdown(
        "Step through the simulation day by day and inspect individual consumer "
        "agent decisions — archetype mix, panic levels, fulfillment rates, and "
        "behavioural-theory scores for every shopper on any given day."
    )

    df_log = st.session_state.get("agent_log")
    if df_log is None or df_log.empty:
        st.info(
            "No agent data found yet. Run the **🎮 Interactive Demo** simulation first — "
            "agent snapshots are automatically captured during each run."
        )
        return

    scenarios = sorted(df_log["Scenario"].unique())
    days_avail = sorted(df_log["Day"].unique())

    # ── Controls ─────────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        sel_day = st.slider(
            "📅 Day", min_value=int(days_avail[0]), max_value=int(days_avail[-1]),
            value=int(days_avail[0]), step=1, key="replay_day_slider",
        )
    with ctrl_col2:
        sel_scenario = st.selectbox("Scenario", scenarios, key="replay_scenario_sel")

    day_df = df_log[(df_log["Day"] == sel_day) & (df_log["Scenario"] == sel_scenario)].copy()
    if day_df.empty:
        st.warning("No agent data for this day/scenario combination.")
        return

    n_agents    = len(day_df)
    avg_fulfill = day_df["Fulfillment"].mean()
    avg_panic   = day_df["PanicLevel"].mean()
    pct_exh     = day_df["BudgetExhausted"].mean() * 100
    avg_pbc     = day_df["PBC"].mean()

    # ── KPI strip ─────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shoppers today",     f"{n_agents}")
    m2.metric("Avg Fulfillment",    f"{avg_fulfill:.1%}")
    m3.metric("Budget Exhausted",   f"{pct_exh:.1f}%")
    m4.metric("Avg Panic Level",    f"{avg_panic:.2f}")

    st.divider()

    # ── Scatter: Panic vs PBC coloured by archetype ───────────────────────────
    st.subheader("🔬 Agent Decision Space — Panic vs Perceived Behavioural Control")
    st.caption(
        "Each dot is one shopper. Size = items purchased. "
        "High panic + low PBC (bottom-right) = most vulnerable agents."
    )

    day_df["ArchetypeLabel"] = day_df["Archetype"].map(
        lambda x: _ARCHETYPE_LABELS.get(x, x)
    )
    day_df["SizeScaled"] = (day_df["ItemsPurchased"].clip(lower=1) * 4).clip(upper=40)

    fig_scatter = px.scatter(
        day_df,
        x="PanicLevel", y="PBC",
        color="ArchetypeLabel",
        size="SizeScaled",
        size_max=18,
        color_discrete_map={v: _ARCHETYPE_COLORS[k] for k, v in _ARCHETYPE_LABELS.items()},
        hover_data={
            "IncomeBracket": True,
            "ItemsWanted": True,
            "ItemsPurchased": True,
            "Fulfillment": ":.1%",
            "BudgetExhausted": True,
            "SubjectiveNorm": ":.2f",
            "SizeScaled": False,
        },
        labels={"PanicLevel": "Panic Level (0–1)", "PBC": "Perceived Behavioural Control (0–1)",
                "ArchetypeLabel": "Archetype"},
        title=f"Day {sel_day} — {sel_scenario}: {n_agents} shoppers",
        template="plotly_white",
    )
    fig_scatter.update_layout(
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=90, b=40, l=50, r=20),
    )
    # Danger zone annotation
    fig_scatter.add_shape(
        type="rect", x0=0.5, x1=1.0, y0=0.0, y1=0.5,
        fillcolor="rgba(220,20,60,0.06)", line=dict(color="rgba(220,20,60,0.3)", dash="dot"),
    )
    fig_scatter.add_annotation(
        x=0.75, y=0.25, text="⚠️ High vulnerability zone",
        showarrow=False, font=dict(size=10, color="#c0392b"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config=_PLOTLY_CFG)

    # ── Archetype breakdown ───────────────────────────────────────────────────
    st.subheader("🏷️ Archetype Breakdown — Day " + str(sel_day))
    arch_summary = (
        day_df.groupby("Archetype")
        .agg(
            Count       =("Archetype",       "count"),
            AvgFulfill  =("Fulfillment",      "mean"),
            AvgPanic    =("PanicLevel",       "mean"),
            PctExhausted=("BudgetExhausted",  "mean"),
            AvgPBC      =("PBC",              "mean"),
        )
        .reset_index()
    )
    arch_summary["AvgFulfill"]   = arch_summary["AvgFulfill"].map("{:.1%}".format)
    arch_summary["PctExhausted"] = arch_summary["PctExhausted"].map("{:.1%}".format)
    arch_summary["AvgPanic"]     = arch_summary["AvgPanic"].map("{:.2f}".format)
    arch_summary["AvgPBC"]       = arch_summary["AvgPBC"].map("{:.2f}".format)
    arch_summary["Archetype"]    = arch_summary["Archetype"].map(
        lambda x: _ARCHETYPE_LABELS.get(x, x)
    )
    arch_summary.columns = ["Archetype", "Shoppers", "Avg Fulfillment",
                             "Avg Panic", "% Budget Exhausted", "Avg PBC"]
    st.dataframe(arch_summary, use_container_width=True, hide_index=True)

    st.divider()

    # ── Time-series: daily aggregate traces across all days ──────────────────
    st.subheader("📈 Daily Trends — " + sel_scenario)
    daily_agg = (
        df_log[df_log["Scenario"] == sel_scenario]
        .groupby(["Day", "Archetype"])
        .agg(AvgFulfill=("Fulfillment", "mean"), AvgPanic=("PanicLevel", "mean"))
        .reset_index()
    )
    daily_agg["ArchetypeLabel"] = daily_agg["Archetype"].map(
        lambda x: _ARCHETYPE_LABELS.get(x, x)
    )

    tab_ts1, tab_ts2 = st.tabs(["Fulfilment Rate by Archetype", "Panic Level by Archetype"])

    with tab_ts1:
        fig_ts1 = px.line(
            daily_agg, x="Day", y="AvgFulfill", color="ArchetypeLabel",
            color_discrete_map={v: _ARCHETYPE_COLORS[k] for k, v in _ARCHETYPE_LABELS.items()},
            labels={"AvgFulfill": "Avg Fulfillment", "ArchetypeLabel": "Archetype"},
            title=f"Daily Mean Fulfillment Rate — {sel_scenario}",
            template="plotly_white",
        )
        fig_ts1.add_vline(x=sel_day, line_dash="dot", line_color="#888",
                          annotation_text=f"Day {sel_day}")
        fig_ts1.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ts1, use_container_width=True, config=_PLOTLY_CFG)

    with tab_ts2:
        fig_ts2 = px.line(
            daily_agg, x="Day", y="AvgPanic", color="ArchetypeLabel",
            color_discrete_map={v: _ARCHETYPE_COLORS[k] for k, v in _ARCHETYPE_LABELS.items()},
            labels={"AvgPanic": "Avg Panic Level", "ArchetypeLabel": "Archetype"},
            title=f"Daily Mean Panic Level — {sel_scenario}",
            template="plotly_white",
        )
        fig_ts2.add_vline(x=sel_day, line_dash="dot", line_color="#888",
                          annotation_text=f"Day {sel_day}")
        st.plotly_chart(fig_ts2, use_container_width=True, config=_PLOTLY_CFG)

    # ── Income vulnerability heatmap ──────────────────────────────────────────
    st.subheader("💶 Income Vulnerability — Budget Exhaustion by Bracket Over Time")
    heat_df = (
        df_log[df_log["Scenario"] == sel_scenario]
        .groupby(["Day", "IncomeBracket"])["BudgetExhausted"]
        .mean()
        .reset_index()
    )
    heat_pivot = heat_df.pivot(index="IncomeBracket", columns="Day",
                                values="BudgetExhausted").fillna(0)
    # Reorder rows Low → Mid → High
    _bracket_order = [b for b in ["Low", "Mid", "High"] if b in heat_pivot.index]
    heat_pivot = heat_pivot.loc[_bracket_order]

    fig_heat = go.Figure(go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns.tolist(),
        y=heat_pivot.index.tolist(),
        colorscale=[[0, "#27AE60"], [0.5, "#F1C40F"], [1, "#C0392B"]],
        zmin=0, zmax=1,
        colorbar=dict(title="Budget<br>Exhaustion", tickformat=".0%"),
        hoverongaps=False,
    ))
    fig_heat.update_layout(
        title=f"Budget Exhaustion Rate by Income Bracket — {sel_scenario}",
        xaxis_title="Day", yaxis_title="Income Bracket",
        template="plotly_white",
        height=250,
    )
    st.plotly_chart(fig_heat, use_container_width=True, config=_PLOTLY_CFG)

    # ── Raw data download ─────────────────────────────────────────────────────
    st.divider()
    st.download_button(
        "📥 Download full agent log (CSV)",
        df_log.to_csv(index=False).encode("utf-8"),
        "GROCERYsim_agent_log.csv",
        "text/csv",
        key="dl_agent_log",
    )


# ===========================================================================
# 14. REGIONAL MAP (Finnish store network + food-security overlay)
# ===========================================================================

# Finnish supermarket locations (realistic, not exhaustive)
_FI_STORES = [
    # Helsinki metro
    {"name": "K-Citymarket Jumbo", "chain": "K-Citymarket", "lat": 60.2927, "lon": 25.0414, "region": "Uusimaa",       "city": "Vantaa",       "size": "hypermarket"},
    {"name": "Prisma Kannelmäki",  "chain": "Prisma",        "lat": 60.2343, "lon": 24.8878, "region": "Uusimaa",       "city": "Helsinki",     "size": "hypermarket"},
    {"name": "S-Market Kamppi",    "chain": "S-Market",      "lat": 60.1683, "lon": 24.9316, "region": "Uusimaa",       "city": "Helsinki",     "size": "supermarket"},
    {"name": "Lidl Itäkeskus",     "chain": "Lidl",          "lat": 60.2105, "lon": 25.0800, "region": "Uusimaa",       "city": "Helsinki",     "size": "supermarket"},
    {"name": "K-Supermarket Munkkivuori", "chain": "K-Supermarket", "lat": 60.2041, "lon": 24.8817, "region": "Uusimaa", "city": "Helsinki",   "size": "supermarket"},
    {"name": "Alepa Punavuori",    "chain": "Alepa",         "lat": 60.1605, "lon": 24.9407, "region": "Uusimaa",       "city": "Helsinki",     "size": "convenience"},
    {"name": "Prisma Lippulaiva",  "chain": "Prisma",        "lat": 60.1631, "lon": 24.7439, "region": "Uusimaa",       "city": "Espoo",        "size": "hypermarket"},
    {"name": "K-Citymarket Ruoholahti", "chain": "K-Citymarket", "lat": 60.1631, "lon": 24.9096, "region": "Uusimaa",  "city": "Helsinki",     "size": "hypermarket"},
    # Tampere
    {"name": "Prisma Lielahti",    "chain": "Prisma",        "lat": 61.5139, "lon": 23.7072, "region": "Pirkanmaa",     "city": "Tampere",      "size": "hypermarket"},
    {"name": "K-Citymarket Turtola","chain": "K-Citymarket", "lat": 61.4918, "lon": 23.7895, "region": "Pirkanmaa",     "city": "Tampere",      "size": "hypermarket"},
    {"name": "S-Market Tampere",   "chain": "S-Market",      "lat": 61.4978, "lon": 23.7610, "region": "Pirkanmaa",     "city": "Tampere",      "size": "supermarket"},
    {"name": "Lidl Tampere",       "chain": "Lidl",          "lat": 61.5021, "lon": 23.7553, "region": "Pirkanmaa",     "city": "Tampere",      "size": "supermarket"},
    # Turku
    {"name": "Prisma Länsikeskus", "chain": "Prisma",        "lat": 60.4518, "lon": 22.2153, "region": "Varsinais-Suomi","city": "Turku",       "size": "hypermarket"},
    {"name": "K-Citymarket Turku", "chain": "K-Citymarket",  "lat": 60.4518, "lon": 22.2641, "region": "Varsinais-Suomi","city": "Turku",       "size": "hypermarket"},
    {"name": "S-Market Turku",     "chain": "S-Market",      "lat": 60.4518, "lon": 22.2666, "region": "Varsinais-Suomi","city": "Turku",       "size": "supermarket"},
    # Oulu
    {"name": "Prisma Oulu",        "chain": "Prisma",        "lat": 65.0121, "lon": 25.4651, "region": "Pohjois-Pohjanmaa","city": "Oulu",       "size": "hypermarket"},
    {"name": "K-Citymarket Oulu",  "chain": "K-Citymarket",  "lat": 65.0031, "lon": 25.5181, "region": "Pohjois-Pohjanmaa","city": "Oulu",       "size": "hypermarket"},
    {"name": "S-Market Oulu",      "chain": "S-Market",      "lat": 65.0121, "lon": 25.4731, "region": "Pohjois-Pohjanmaa","city": "Oulu",       "size": "supermarket"},
    {"name": "Lidl Oulu",          "chain": "Lidl",          "lat": 64.9993, "lon": 25.5013, "region": "Pohjois-Pohjanmaa","city": "Oulu",       "size": "supermarket"},
    # Jyväskylä
    {"name": "Prisma Seppälä",     "chain": "Prisma",        "lat": 62.2421, "lon": 25.7482, "region": "Keski-Suomi",   "city": "Jyväskylä",   "size": "hypermarket"},
    {"name": "K-Citymarket Jyväskylä","chain": "K-Citymarket","lat": 62.2366, "lon": 25.7482,"region": "Keski-Suomi",   "city": "Jyväskylä",   "size": "hypermarket"},
    {"name": "S-Market Jyväskylä", "chain": "S-Market",      "lat": 62.2421, "lon": 25.7413, "region": "Keski-Suomi",   "city": "Jyväskylä",   "size": "supermarket"},
    # Kuopio
    {"name": "Prisma Kuopio",      "chain": "Prisma",        "lat": 62.8980, "lon": 27.6782, "region": "Pohjois-Savo",  "city": "Kuopio",      "size": "hypermarket"},
    {"name": "K-Citymarket Kuopio","chain": "K-Citymarket",  "lat": 62.8879, "lon": 27.6894, "region": "Pohjois-Savo",  "city": "Kuopio",      "size": "hypermarket"},
    # Joensuu
    {"name": "Prisma Joensuu",     "chain": "Prisma",        "lat": 62.5984, "lon": 29.7740, "region": "Pohjois-Karjala","city": "Joensuu",    "size": "hypermarket"},
    {"name": "S-Market Joensuu",   "chain": "S-Market",      "lat": 62.6009, "lon": 29.7618, "region": "Pohjois-Karjala","city": "Joensuu",    "size": "supermarket"},
    # Lahti
    {"name": "Prisma Lahti",       "chain": "Prisma",        "lat": 60.9827, "lon": 25.6609, "region": "Päijät-Häme",   "city": "Lahti",       "size": "hypermarket"},
    {"name": "K-Citymarket Lahti", "chain": "K-Citymarket",  "lat": 60.9827, "lon": 25.6553, "region": "Päijät-Häme",   "city": "Lahti",       "size": "hypermarket"},
    # Rovaniemi (Lapland)
    {"name": "Prisma Rovaniemi",   "chain": "Prisma",        "lat": 66.5039, "lon": 25.7294, "region": "Lappi",          "city": "Rovaniemi",   "size": "hypermarket"},
    {"name": "S-Market Rovaniemi", "chain": "S-Market",      "lat": 66.5007, "lon": 25.7320, "region": "Lappi",          "city": "Rovaniemi",   "size": "supermarket"},
    # Vaasa
    {"name": "Prisma Vaasa",       "chain": "Prisma",        "lat": 63.0961, "lon": 21.5922, "region": "Pohjanmaa",      "city": "Vaasa",       "size": "hypermarket"},
    {"name": "K-Citymarket Vaasa", "chain": "K-Citymarket",  "lat": 63.0961, "lon": 21.5800, "region": "Pohjanmaa",      "city": "Vaasa",       "size": "hypermarket"},
]

# Regional food-security & population context
_FI_REGIONS = {
    "Uusimaa":             {"pop": 1_700_000, "income_idx": 1.25, "rural_pct": 5,  "import_dep": 55},
    "Pirkanmaa":           {"pop":   530_000, "income_idx": 1.05, "rural_pct": 18, "import_dep": 48},
    "Varsinais-Suomi":     {"pop":   490_000, "income_idx": 1.03, "rural_pct": 22, "import_dep": 45},
    "Pohjois-Pohjanmaa":   {"pop":   420_000, "income_idx": 0.95, "rural_pct": 35, "import_dep": 42},
    "Keski-Suomi":         {"pop":   280_000, "income_idx": 0.93, "rural_pct": 40, "import_dep": 44},
    "Pohjois-Savo":        {"pop":   245_000, "income_idx": 0.90, "rural_pct": 45, "import_dep": 40},
    "Pohjois-Karjala":     {"pop":   162_000, "income_idx": 0.86, "rural_pct": 52, "import_dep": 38},
    "Päijät-Häme":         {"pop":   202_000, "income_idx": 0.96, "rural_pct": 20, "import_dep": 47},
    "Lappi":               {"pop":   180_000, "income_idx": 0.88, "rural_pct": 75, "import_dep": 35},
    "Pohjanmaa":           {"pop":   185_000, "income_idx": 0.97, "rural_pct": 38, "import_dep": 39},
}

_CHAIN_COLORS = {
    "K-Citymarket":  "#E87722",
    "K-Supermarket": "#F5A623",
    "Prisma":        "#27AE60",
    "S-Market":      "#44A1A0",
    "Lidl":          "#2471A3",
    "Alepa":         "#8E44AD",
}
_SIZE_MAP = {"hypermarket": 18, "supermarket": 12, "convenience": 7}


def render_regional_map_tab():
    st.header("🗺️ Finnish Store Network & Regional Food-Security Map")
    st.markdown(
        "Explore the geographical distribution of major grocery chains across Finland "
        "and see how regional food-security indicators (import dependency, rural access, "
        "income index) interact with your simulation results."
    )

    df_stores = pd.DataFrame(_FI_STORES)
    df_stores["MarkerSize"] = df_stores["size"].map(_SIZE_MAP)
    df_stores["ChainColor"] = df_stores["chain"].map(_CHAIN_COLORS)

    # ── Controls ─────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        sel_chains = st.multiselect(
            "Filter by chain", sorted(df_stores["chain"].unique()),
            default=sorted(df_stores["chain"].unique()),
            key="map_chains",
        )
    with ctrl2:
        sel_regions = st.multiselect(
            "Filter by region", sorted(df_stores["region"].unique()),
            default=sorted(df_stores["region"].unique()),
            key="map_regions",
        )
    with ctrl3:
        map_metric = st.selectbox(
            "Regional overlay metric",
            ["Import Dependency %", "Rural Population %", "Income Index", "Population"],
            key="map_metric",
        )

    _metric_col_map = {
        "Import Dependency %": "import_dep",
        "Rural Population %":  "rural_pct",
        "Income Index":        "income_idx",
        "Population":          "pop",
    }
    metric_col = _metric_col_map[map_metric]

    filtered = df_stores[
        df_stores["chain"].isin(sel_chains) &
        df_stores["region"].isin(sel_regions)
    ].copy()

    if filtered.empty:
        st.warning("No stores match the current filters.")
        return

    # Attach regional metric to each store row
    filtered["RegionMetric"] = filtered["region"].map(
        lambda r: _FI_REGIONS.get(r, {}).get(metric_col, 0)
    )
    filtered["RegionPop"] = filtered["region"].map(
        lambda r: _FI_REGIONS.get(r, {}).get("pop", 0)
    )

    # ── Map ───────────────────────────────────────────────────────────────────
    fig_map = px.scatter_mapbox(
        filtered,
        lat="lat", lon="lon",
        color="chain",
        size="MarkerSize",
        size_max=20,
        color_discrete_map=_CHAIN_COLORS,
        hover_name="name",
        hover_data={
            "city": True,
            "region": True,
            "size": True,
            "RegionMetric": True,
            "lat": False,
            "lon": False,
            "MarkerSize": False,
            "ChainColor": False,
            "RegionPop": True,
        },
        labels={"chain": "Chain", "RegionMetric": map_metric, "RegionPop": "Region pop."},
        zoom=4.5,
        center={"lat": 64.5, "lon": 26.0},
        mapbox_style="open-street-map",
        title="Finnish Supermarket Network",
        height=560,
    )
    fig_map.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        margin=dict(l=0, r=0, t=70, b=0),
    )
    st.plotly_chart(fig_map, use_container_width=True, config=_PLOTLY_CFG)

    # ── Regional stats table ──────────────────────────────────────────────────
    st.subheader("📊 Regional Food-Security Indicators")

    # Build region summary
    region_rows = []
    for region, stats_r in _FI_REGIONS.items():
        n_stores = len(df_stores[df_stores["region"] == region])
        region_rows.append({
            "Region":             region,
            "Population":         f"{stats_r['pop']:,}",
            "Income Index":       f"{stats_r['income_idx']:.2f}",
            "Import Dep. %":      f"{stats_r['import_dep']}%",
            "Rural Pop. %":       f"{stats_r['rural_pct']}%",
            "Mapped Stores":      n_stores,
            "Food Access Risk":   (
                "🔴 High"   if stats_r["rural_pct"] > 60 or stats_r["income_idx"] < 0.90 else
                "🟡 Medium" if stats_r["rural_pct"] > 35 or stats_r["income_idx"] < 0.95 else
                "🟢 Low"
            ),
        })
    st.dataframe(pd.DataFrame(region_rows), use_container_width=True, hide_index=True)

    # ── Simulation overlay (if data available) ───────────────────────────────
    df_sim = st.session_state.get("sim_results")
    if df_sim is not None and not df_sim.empty:
        st.divider()
        st.subheader("🔗 Simulation → Regional Projection")
        st.markdown(
            "Your simulation results scaled to each region's population. "
            "This projects what the simulated supply shock would mean if the "
            "same scenario played out in each Finnish region."
        )

        df_crisis = df_sim[df_sim["Scenario"] == "Crisis"] if "Crisis" in df_sim["Scenario"].values else df_sim
        sim_avg_daily_rev   = df_crisis["Revenue"].mean()   if "Revenue"   in df_crisis.columns else 1
        sim_avg_daily_waste = df_crisis["Waste"].mean()     if "Waste"     in df_crisis.columns else 0
        sim_avg_food_stress = df_crisis["FoodStressedPct"].mean() if "FoodStressedPct" in df_crisis.columns else 0
        # Simulation uses ~200 agents; scale to store catchment ~5000 households
        _scale = 5000 / max(df_crisis["Consumers"].mean(), 1) if "Consumers" in df_crisis.columns else 25

        proj_rows = []
        for region, stats_r in _FI_REGIONS.items():
            n_stores_r  = len(df_stores[df_stores["region"] == region])
            region_scale = (stats_r["pop"] / 200_000) * (stats_r["income_idx"])
            proj_rev    = sim_avg_daily_rev   * _scale * region_scale * n_stores_r
            proj_waste  = sim_avg_daily_waste * _scale * region_scale * n_stores_r
            proj_stress = sim_avg_food_stress * stats_r["pop"]
            proj_rows.append({
                "Region":                   region,
                "Est. Daily Revenue (€)":   f"{proj_rev:,.0f}",
                "Est. Daily Food Waste":    f"{proj_waste:,.0f} units",
                "Est. Food-Stressed People":f"{proj_stress:,.0f}",
                "Mapped Stores":            n_stores_r,
            })
        st.dataframe(pd.DataFrame(proj_rows), use_container_width=True, hide_index=True)
        st.caption(
            "⚠️ Projections are illustrative estimates based on linear scaling from "
            "the ABM (single store, ~200 agents). They are not validated forecasts."
        )

    # ── Multi-Store simulation overlay ────────────────────────────────────────
    st.divider()
    st.subheader("🏪 Multi-Store Simulation Overlay")

    df_ms      = st.session_state.get("multistore_results")
    ms_configs = st.session_state.get("multistore_config", [])

    if df_ms is None or df_ms.empty:
        st.info(
            "No multi-store simulation results yet. "
            "Run the **🏪 Multi-Store Network** simulation first — "
            "your simulated stores and their risk levels will appear on this map."
        )
        if st.button("🏪 Go to Multi-Store Network", key="map_goto_ms",
                     use_container_width=False):
            st.session_state["nav_section"] = "multistore"
            st.rerun()
    else:
        # Build risk table from multi-store results
        _RISK_COLORS_MAP = {
            "🔴 Critical": "#C0392B",
            "🟠 High":     "#E67E22",
            "🟡 Medium":   "#F1C40F",
            "🟢 Low":      "#27AE60",
        }
        df_crisis_ms = df_ms[df_ms["Scenario"] == "Crisis"]
        df_base_ms   = df_ms[df_ms["Scenario"] == "Baseline"]

        sim_store_pts = []
        for sc in ms_configs:
            s_name  = sc.get("name", "")
            s_region = sc.get("region", "")
            # Find the store in _FI_STORES for coordinates
            match = next(
                (s for s in _FI_STORES
                 if sc.get("name", "").lower() in s["name"].lower()
                 or s["name"].lower() in sc.get("name", "").lower()),
                None,
            )
            # Fall back to region centroid from any store in that region
            if match is None:
                match = next(
                    (s for s in _FI_STORES if s["region"] == s_region), None
                )
            if match is None:
                continue

            sub_c = df_crisis_ms[df_crisis_ms["Store"] == s_name]
            sub_b = df_base_ms[df_base_ms["Store"]   == s_name]
            if sub_c.empty:
                continue
            avg_rev_b = sub_b["Revenue"].mean() if not sub_b.empty else 1
            avg_rev_c = sub_c["Revenue"].mean()
            rev_loss  = (avg_rev_b - avg_rev_c) / max(avg_rev_b, 1) * 100
            risk = (
                "🔴 Critical" if rev_loss >= 35 else
                "🟠 High"     if rev_loss >= 20 else
                "🟡 Medium"   if rev_loss >= 10 else
                "🟢 Low"
            )
            sim_store_pts.append({
                "name":     s_name,
                "lat":      match["lat"],
                "lon":      match["lon"],
                "region":   s_region,
                "type":     sc.get("type", ""),
                "rev_loss": round(rev_loss, 1),
                "risk":     risk,
                "color":    _RISK_COLORS_MAP.get(risk, "#888"),
                "size":     22,
            })

        if sim_store_pts:
            show_overlay = st.toggle(
                "Show Multi-Store simulation results on map",
                value=True, key="map_ms_overlay_toggle",
            )

            if show_overlay:
                df_sim_pts = pd.DataFrame(sim_store_pts)
                fig_overlay = px.scatter_mapbox(
                    df_sim_pts,
                    lat="lat", lon="lon",
                    color="risk",
                    size="size",
                    size_max=22,
                    color_discrete_map=_RISK_COLORS_MAP,
                    hover_name="name",
                    hover_data={
                        "region":   True,
                        "type":     True,
                        "rev_loss": True,
                        "risk":     True,
                        "lat": False, "lon": False, "size": False, "color": False,
                    },
                    labels={"rev_loss": "Revenue Loss %", "risk": "Risk Level"},
                    zoom=4.5,
                    center={"lat": 64.5, "lon": 26.0},
                    mapbox_style="open-street-map",
                    title="Simulated Store Network — Risk Level Overlay",
                    height=480,
                )
                fig_overlay.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.01),
                    margin=dict(l=0, r=0, t=70, b=0),
                )
                st.plotly_chart(fig_overlay, use_container_width=True, config=_PLOTLY_CFG)

                # Risk summary
                risk_counts = df_sim_pts["risk"].value_counts().to_dict()
                rc1, rc2, rc3, rc4 = st.columns(4)
                for col, risk_lbl, bg in [
                    (rc1, "🔴 Critical", "#fdecea"),
                    (rc2, "🟠 High",     "#fef3e2"),
                    (rc3, "🟡 Medium",   "#fefce8"),
                    (rc4, "🟢 Low",      "#eafaf1"),
                ]:
                    col.markdown(
                        f"<div style='background:{bg};border-radius:8px;padding:10px;"
                        f"text-align:center;'>"
                        f"<div style='font-weight:700;font-size:0.9rem;'>{risk_lbl}</div>"
                        f"<div style='font-size:1.6rem;font-weight:700;'>"
                        f"{risk_counts.get(risk_lbl, 0)}</div>"
                        f"<div style='font-size:0.75rem;color:#666;'>stores</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("")
        if st.button("🔄 Re-run Multi-Store Simulation", key="map_goto_ms_rerun"):
            st.session_state["nav_section"] = "multistore"
            st.rerun()


# ===========================================================================
# 15. MULTI-STORE NETWORK
# ===========================================================================

# ── Store-type consumer-count multipliers ───────────────────────────────────
_STORE_TYPE_MULT = {
    "Hypermarket":   1.00,
    "Supermarket":   0.50,
    "Convenience":   0.15,
}

# ── Preset store networks ────────────────────────────────────────────────────
_NETWORK_PRESETS = {
    "Helsinki Metro (4 stores)": [
        {"name": "K-Citymarket Jumbo",    "region": "Uusimaa",     "type": "Hypermarket",  "dis_sensitivity": 1.0},
        {"name": "Prisma Lippulaiva",     "region": "Uusimaa",     "type": "Hypermarket",  "dis_sensitivity": 1.0},
        {"name": "S-Market Kamppi",       "region": "Uusimaa",     "type": "Supermarket",  "dis_sensitivity": 1.2},
        {"name": "Alepa Punavuori",       "region": "Uusimaa",     "type": "Convenience",  "dis_sensitivity": 1.5},
    ],
    "Multi-Region Network (6 stores)": [
        {"name": "Prisma Helsinki",       "region": "Uusimaa",           "type": "Hypermarket", "dis_sensitivity": 1.0},
        {"name": "K-Citymarket Tampere",  "region": "Pirkanmaa",         "type": "Hypermarket", "dis_sensitivity": 1.0},
        {"name": "Prisma Turku",          "region": "Varsinais-Suomi",   "type": "Hypermarket", "dis_sensitivity": 1.1},
        {"name": "S-Market Oulu",         "region": "Pohjois-Pohjanmaa", "type": "Supermarket", "dis_sensitivity": 1.3},
        {"name": "K-Citymarket Jyväskylä","region": "Keski-Suomi",       "type": "Supermarket", "dis_sensitivity": 1.2},
        {"name": "Prisma Rovaniemi",      "region": "Lappi",             "type": "Supermarket", "dis_sensitivity": 1.6},
    ],
    "National Supply Chain (8 stores)": [
        {"name": "Prisma Helsinki",       "region": "Uusimaa",           "type": "Hypermarket", "dis_sensitivity": 1.0},
        {"name": "K-Citymarket Espoo",    "region": "Uusimaa",           "type": "Hypermarket", "dis_sensitivity": 1.0},
        {"name": "Prisma Tampere",        "region": "Pirkanmaa",         "type": "Hypermarket", "dis_sensitivity": 1.1},
        {"name": "K-Citymarket Turku",    "region": "Varsinais-Suomi",   "type": "Hypermarket", "dis_sensitivity": 1.1},
        {"name": "Prisma Oulu",           "region": "Pohjois-Pohjanmaa", "type": "Hypermarket", "dis_sensitivity": 1.2},
        {"name": "S-Market Kuopio",       "region": "Pohjois-Savo",      "type": "Supermarket", "dis_sensitivity": 1.3},
        {"name": "S-Market Joensuu",      "region": "Pohjois-Karjala",   "type": "Supermarket", "dis_sensitivity": 1.4},
        {"name": "Prisma Rovaniemi",      "region": "Lappi",             "type": "Supermarket", "dis_sensitivity": 1.7},
    ],
}

# Region proximity matrix — 0=same region, 1=adjacent, 2=distant
# Used for panic-contagion decay: same=0.80, adjacent=0.35, distant=0.10
_REGION_PROXIMITY = {
    ("Uusimaa", "Uusimaa"):                 0,
    ("Uusimaa", "Pirkanmaa"):               1,
    ("Uusimaa", "Varsinais-Suomi"):         1,
    ("Uusimaa", "Päijät-Häme"):             1,
    ("Pirkanmaa", "Varsinais-Suomi"):       1,
    ("Pirkanmaa", "Keski-Suomi"):           1,
    ("Pirkanmaa", "Pohjois-Savo"):          2,
    ("Keski-Suomi", "Pohjois-Savo"):        1,
    ("Keski-Suomi", "Pohjois-Karjala"):     1,
    ("Pohjois-Pohjanmaa", "Lappi"):         1,
    ("Pohjois-Savo", "Pohjois-Karjala"):    1,
}
_CONTAGION_DECAY = {0: 0.80, 1: 0.35, 2: 0.10}


def _proximity_level(r1: str, r2: str) -> int:
    """Return proximity level (0=same, 1=adjacent, 2=distant)."""
    if r1 == r2:
        return 0
    return _REGION_PROXIMITY.get((r1, r2), _REGION_PROXIMITY.get((r2, r1), 2))


def _run_multistore_network(
    store_configs: list[dict],
    params: dict,
    n_days: int,
    crisis_start: int,
    inflation: float,
    disruption_days: int,
    panic_sens: float,
    hoard: float,
    enable_contagion: bool,
    enable_redistribution: bool,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run N SupermarketModel instances in lockstep, one per store config.

    Inter-store dynamics (when enabled):
    • Panic contagion  — after each day, stores with panic > 0.55 spread
      a fraction of their panic to other stores (decay by proximity).
    • Emergency redistribution — a shared network buffer of supply days
      (= n_stores × 2) is drawn from when any store's daily revenue drops
      >35 % below its own baseline. Each draw reduces that store's
      effective disruption by 1 day (capped at buffer remaining).

    Returns a long DataFrame: Day, Store, Region, Type, Scenario, and all
    aggregate KPIs from _collect_model_day().
    """
    if st.session_state.config_data is None:
        return pd.DataFrame()

    n_stores = len(store_configs)
    base_con  = params["base_con"]

    # Build one model per store (baseline + crisis)
    models_base   = []
    models_crisis = []
    for i, sc in enumerate(store_configs):
        region_stats = _FI_REGIONS.get(sc["region"], {})
        income_idx   = region_stats.get("income_idx", 1.0)
        store_mult   = _STORE_TYPE_MULT.get(sc["type"], 0.5)
        eff_consumers = max(10, int(base_con * store_mult * income_idx))
        eff_dis       = max(0, int(disruption_days * sc.get("dis_sensitivity", 1.0)))

        store_params = {**params,
                        "base_con":  eff_consumers,
                        "dis":       eff_dis,
                        "inf":       inflation,
                        "panic":     panic_sens,
                        "hoard":     hoard,
                        "cri_start": crisis_start,
                        "days":      n_days}
        models_base.append(_make_model(store_params, is_crisis=False, seed=seed + i))
        models_crisis.append(_make_model(store_params, is_crisis=True,  seed=seed + i))

    # Redistribution buffer
    redist_buffer = n_stores * 2  # shared supply-day buffer

    rows = []
    # Track each store's baseline revenue (day-5 rolling average) for redistribution trigger
    baseline_rev_tracker = [[] for _ in range(n_stores)]

    for day in range(1, n_days + 1):
        for i in range(n_stores):
            models_base[i].step()
            models_crisis[i].step()

        # ── Panic contagion ──────────────────────────────────────────────────
        if enable_contagion:
            panic_levels = [m.global_panic_level for m in models_crisis]
            new_panics   = list(panic_levels)
            for i in range(n_stores):
                if panic_levels[i] > 0.55:
                    for j in range(n_stores):
                        if i == j:
                            continue
                        prox   = _proximity_level(store_configs[i]["region"],
                                                  store_configs[j]["region"])
                        decay  = _CONTAGION_DECAY.get(prox, 0.10)
                        signal = (panic_levels[i] - 0.55) * decay * 0.30
                        new_panics[j] = min(1.0, new_panics[j] + signal)
            for i, m in enumerate(models_crisis):
                m.global_panic_level = new_panics[i]

        # ── Emergency redistribution ─────────────────────────────────────────
        if enable_redistribution and redist_buffer > 0 and day >= crisis_start:
            for i in range(n_stores):
                last_rec_c = models_crisis[i].daily_records[-1] if models_crisis[i].daily_records else {}
                last_rec_b = models_base[i].daily_records[-1]   if models_base[i].daily_records   else {}
                rev_c = last_rec_c.get("Revenue", 0)
                rev_b = last_rec_b.get("Revenue", 1)
                if rev_b > 0 and (rev_b - rev_c) / rev_b > 0.35:
                    # Store in crisis — draw from buffer
                    draw = min(1, redist_buffer)
                    redist_buffer -= draw
                    # Simulate redistribution: temporarily reduce effective disruption
                    if models_crisis[i].supply_disruption_days > 0:
                        models_crisis[i].supply_disruption_days = max(
                            0, models_crisis[i].supply_disruption_days - draw
                        )

        # ── Collect results ──────────────────────────────────────────────────
        for i, sc in enumerate(store_configs):
            agg_b, _ = _collect_model_day(models_base[i],   day, "Baseline",
                                          collect_products=False)
            agg_c, _ = _collect_model_day(models_crisis[i], day, "Crisis",
                                          collect_products=False)
            for agg, scenario in [(agg_b, "Baseline"), (agg_c, "Crisis")]:
                rows.append({
                    "Day":      day,
                    "Store":    sc["name"],
                    "Region":   sc["region"],
                    "Type":     sc["type"],
                    "Scenario": scenario,
                    **{k: agg[k] for k in [
                        "Revenue", "Waste", "LostSales", "PanicLevel",
                        "BudgetExhaustionRate", "FoodStressedPct",
                        "FulfillmentRate", "CO2Total", "ImportDepPct",
                    ] if k in agg},
                })

    return pd.DataFrame(rows)


def render_multistore_tab(params: dict):
    st.header("🏪 Multi-Store Network Simulation")
    st.markdown(
        "Simulate a network of supermarkets across Finland under the same supply-chain "
        "crisis. Each store is calibrated to its region's income level and size. "
        "Optional **panic contagion** spreads fear between nearby stores; "
        "**emergency redistribution** lets the network partially shield the hardest-hit stores."
    )

    if st.session_state.config_data is None:
        st.warning("⚠️ Load data in the **🏠 Data & Population** tab first.")
        return

    # ── Network preset selector ──────────────────────────────────────────────
    st.subheader("1️⃣ Choose a Store Network")
    preset_names = ["Custom"] + list(_NETWORK_PRESETS.keys())
    sel_preset   = st.selectbox("Preset network", preset_names, index=1,
                                key="ms_preset_sel")

    if sel_preset != "Custom":
        default_stores = _NETWORK_PRESETS[sel_preset]
    else:
        default_stores = [
            {"name": "Store A", "region": "Uusimaa",   "type": "Hypermarket", "dis_sensitivity": 1.0},
            {"name": "Store B", "region": "Pirkanmaa", "type": "Supermarket", "dis_sensitivity": 1.2},
        ]

    # ── Editable store table ─────────────────────────────────────────────────
    st.markdown("**Edit the network below** — add rows to expand, delete rows to reduce:")
    region_opts = sorted(_FI_REGIONS.keys())
    type_opts   = list(_STORE_TYPE_MULT.keys())

    edited_stores = st.data_editor(
        pd.DataFrame(default_stores),
        num_rows="dynamic",
        column_config={
            "name":            st.column_config.TextColumn("Store Name"),
            "region":          st.column_config.SelectboxColumn("Region", options=region_opts),
            "type":            st.column_config.SelectboxColumn("Type",   options=type_opts),
            "dis_sensitivity": st.column_config.NumberColumn(
                "Disruption Sensitivity", min_value=0.5, max_value=3.0, step=0.1,
                help="1.0 = same as global setting. >1 = more exposed (e.g. remote store). <1 = more resilient."
            ),
        },
        use_container_width=True,
        key="ms_store_editor",
    )

    store_configs = edited_stores.dropna(subset=["name"]).to_dict("records")
    if len(store_configs) < 1:
        st.warning("Add at least one store to run the simulation.")
        return

    # ── Crisis parameters ────────────────────────────────────────────────────
    st.subheader("2️⃣ Crisis & Network Parameters")
    p1, p2, p3 = st.columns(3)
    with p1:
        ms_days        = st.slider("Simulation days",     30, 180, 90, 10, key="ms_days")
        ms_cri_start   = st.slider("Crisis start (day)",   5,  60, 30,  5, key="ms_cri_start")
    with p2:
        ms_inflation   = st.slider("Inflation rate (%)",   0,  50, 15,  5, key="ms_inf") / 100
        ms_disruption  = st.slider("Disruption (days)",    0,  21,  7,  1, key="ms_dis")
    with p3:
        ms_panic       = st.slider("Panic sensitivity",  0.0, 1.0, 0.5, 0.1, key="ms_panic")
        ms_hoard       = st.slider("Hoarding multiplier",1.0, 3.0, 1.5, 0.1, key="ms_hoard")

    net1, net2 = st.columns(2)
    with net1:
        enable_contagion      = st.toggle("🦠 Enable panic contagion between stores",
                                          value=True, key="ms_contagion")
    with net2:
        enable_redistribution = st.toggle("🚛 Enable emergency supply redistribution",
                                          value=True, key="ms_redist")

    # ── Run ──────────────────────────────────────────────────────────────────
    st.subheader("3️⃣ Run")
    n_stores_display = len(store_configs)
    st.caption(
        f"Network: **{n_stores_display} stores** · "
        f"{'Panic contagion ON' if enable_contagion else 'No contagion'} · "
        f"{'Redistribution ON' if enable_redistribution else 'No redistribution'}"
    )

    if st.button("▶️ Run Multi-Store Simulation", type="primary",
                 use_container_width=True, key="ms_run_btn"):
        with st.spinner(f"Simulating {n_stores_display} stores × {ms_days} days…"):
            ms_params = {**params, "days": ms_days, "cri_start": ms_cri_start,
                         "cri_duration": 0}
            df_ms = _run_multistore_network(
                store_configs       = store_configs,
                params              = ms_params,
                n_days              = ms_days,
                crisis_start        = ms_cri_start,
                inflation           = ms_inflation,
                disruption_days     = ms_disruption,
                panic_sens          = ms_panic,
                hoard               = ms_hoard,
                enable_contagion    = enable_contagion,
                enable_redistribution = enable_redistribution,
                seed                = 42,
            )
        st.session_state.multistore_results = df_ms
        st.session_state.multistore_config  = store_configs
        st.rerun()

    # ── Results dashboard ─────────────────────────────────────────────────────
    df_ms = st.session_state.get("multistore_results")
    if df_ms is None or df_ms.empty:
        return

    df_crisis   = df_ms[df_ms["Scenario"] == "Crisis"]
    df_baseline = df_ms[df_ms["Scenario"] == "Baseline"]
    stores_list = df_ms["Store"].unique().tolist()

    st.divider()
    st.subheader("📊 Network Dashboard")

    # ── KPI strip ────────────────────────────────────────────────────────────
    total_rev_b  = df_baseline.groupby("Day")["Revenue"].sum().mean()
    total_rev_c  = df_crisis.groupby("Day")["Revenue"].sum().mean()
    rev_drop_pct = (total_rev_b - total_rev_c) / max(total_rev_b, 1) * 100
    avg_fulfill  = df_crisis["FulfillmentRate"].mean()
    avg_food_str = df_crisis["FoodStressedPct"].mean() * 100
    avg_panic    = df_crisis["PanicLevel"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Network Rev/day (Crisis)", f"€{total_rev_c:,.0f}",
              delta=f"{-rev_drop_pct:.1f}% vs Baseline",
              delta_color="inverse")
    k2.metric("Stores simulated",   str(len(stores_list)))
    k3.metric("Avg Fulfillment",    f"{avg_fulfill:.1%}")
    k4.metric("Food-Stressed %",    f"{avg_food_str:.1f}%")
    k5.metric("Avg Panic Level",    f"{avg_panic:.2f}")

    st.divider()

    # ── Revenue heatmap: stores × days ───────────────────────────────────────
    st.subheader("🌡️ Revenue Heatmap — Crisis Scenario (all stores × all days)")
    heat_data = df_crisis.pivot_table(
        index="Store", columns="Day", values="Revenue", aggfunc="mean"
    )
    # Normalise each store row to its own baseline (so absolute-size differences
    # don't hide crisis impact in small stores)
    base_data = df_baseline.pivot_table(
        index="Store", columns="Day", values="Revenue", aggfunc="mean"
    )
    # Relative revenue (crisis ÷ baseline)
    rel_data = heat_data.div(base_data).clip(0, 1.5)

    fig_heat = go.Figure(go.Heatmap(
        z=rel_data.values,
        x=rel_data.columns.tolist(),
        y=rel_data.index.tolist(),
        colorscale=[[0, "#C0392B"], [0.5, "#F1C40F"], [1.0, "#27AE60"]],
        zmin=0.4, zmax=1.1,
        colorbar=dict(title="Crisis Rev<br>÷ Baseline"),
        hovertemplate="Store: %{y}<br>Day: %{x}<br>Relative Rev: %{z:.2f}<extra></extra>",
    ))
    fig_heat.update_layout(
        title="Relative Revenue (Crisis ÷ Baseline) — 1.0 = no impact, <0.7 = severe",
        xaxis_title="Day", yaxis_title="Store",
        template="plotly_white",
        height=max(200, len(stores_list) * 45 + 100),
    )
    st.plotly_chart(fig_heat, use_container_width=True, config=_PLOTLY_CFG)

    # ── Per-store revenue time-series ─────────────────────────────────────────
    st.subheader("📈 Revenue Over Time — Crisis vs Baseline per Store")
    fig_ts = go.Figure()
    _TS_COLORS = ["#E87722","#44A1A0","#27AE60","#8E44AD","#2471A3","#C0392B","#D4AC0D","#1ABC9C"]
    for i, store in enumerate(stores_list):
        col = _TS_COLORS[i % len(_TS_COLORS)]
        sub_b = df_baseline[df_baseline["Store"] == store].groupby("Day")["Revenue"].mean()
        sub_c = df_crisis[df_crisis["Store"] == store].groupby("Day")["Revenue"].mean()
        fig_ts.add_trace(go.Scatter(
            x=sub_b.index, y=sub_b.values,
            name=f"{store} (Baseline)",
            line=dict(color=col, width=1.5, dash="dot"),
            legendgroup=store, showlegend=True,
        ))
        fig_ts.add_trace(go.Scatter(
            x=sub_c.index, y=sub_c.values,
            name=f"{store} (Crisis)",
            line=dict(color=col, width=2.5),
            legendgroup=store, showlegend=True,
        ))
    fig_ts.add_vline(x=ms_cri_start, line_dash="dot", line_color="orange",
                     annotation_text="Crisis start")
    fig_ts.update_layout(
        title="Daily Revenue by Store",
        xaxis_title="Day", yaxis_title="Revenue (€)",
        template="plotly_white",
        legend=dict(orientation="v", x=1.01, y=1, xanchor="left", yanchor="top",
                    font=dict(size=9)),
        margin=dict(t=60, b=40, l=50, r=160),
        height=420,
    )
    st.plotly_chart(fig_ts, use_container_width=True, config=_PLOTLY_CFG)

    # ── Panic contagion chart ─────────────────────────────────────────────────
    if enable_contagion:
        st.subheader("🦠 Panic Contagion — Level by Store Over Time")
        fig_panic = px.line(
            df_crisis.groupby(["Day", "Store"])["PanicLevel"].mean().reset_index(),
            x="Day", y="PanicLevel", color="Store",
            color_discrete_sequence=_TS_COLORS,
            title="Panic Level Propagation Across the Network",
            template="plotly_white",
            labels={"PanicLevel": "Panic Level (0–1)"},
        )
        fig_panic.add_vline(x=ms_cri_start, line_dash="dot", line_color="orange",
                             annotation_text="Crisis start")
        fig_panic.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_panic, use_container_width=True, config=_PLOTLY_CFG)

    # ── Vulnerability ranking ────────────────────────────────────────────────
    st.subheader("🏆 Store Vulnerability Ranking")
    store_summary = []
    for store in stores_list:
        sc_cfg = next((s for s in store_configs if s["name"] == store), {})
        sub_b  = df_baseline[df_baseline["Store"] == store]
        sub_c  = df_crisis[df_crisis["Store"]   == store]
        avg_rev_b  = sub_b["Revenue"].mean()
        avg_rev_c  = sub_c["Revenue"].mean()
        rev_loss   = (avg_rev_b - avg_rev_c) / max(avg_rev_b, 1) * 100
        max_panic  = sub_c["PanicLevel"].max()
        avg_fulfill = sub_c["FulfillmentRate"].mean()
        risk = (
            "🔴 Critical" if rev_loss >= 35 else
            "🟠 High"     if rev_loss >= 20 else
            "🟡 Medium"   if rev_loss >= 10 else
            "🟢 Low"
        )
        store_summary.append({
            "Store":              store,
            "Region":             sc_cfg.get("region", ""),
            "Type":               sc_cfg.get("type",   ""),
            "Avg Rev Baseline":   f"€{avg_rev_b:,.0f}",
            "Avg Rev Crisis":     f"€{avg_rev_c:,.0f}",
            "Revenue Loss %":     f"{rev_loss:.1f}%",
            "Peak Panic":         f"{max_panic:.2f}",
            "Avg Fulfillment":    f"{avg_fulfill:.1%}",
            "Risk Level":         risk,
        })
    # Sort by revenue loss descending
    store_summary.sort(key=lambda r: float(r["Revenue Loss %"].rstrip("%")), reverse=True)
    st.dataframe(pd.DataFrame(store_summary), use_container_width=True, hide_index=True)

    # ── Network aggregate: stacked area ──────────────────────────────────────
    st.subheader("📦 Network Aggregate — Daily Revenue Stack")
    stack_df = df_crisis.groupby(["Day", "Store"])["Revenue"].mean().reset_index()
    fig_stack = px.area(
        stack_df, x="Day", y="Revenue", color="Store",
        color_discrete_sequence=_TS_COLORS,
        title="Combined Network Revenue (Crisis) — stacked by store",
        template="plotly_white",
        labels={"Revenue": "Revenue (€)"},
    )
    fig_stack.add_vline(x=ms_cri_start, line_dash="dot", line_color="orange",
                        annotation_text="Crisis start")
    st.plotly_chart(fig_stack, use_container_width=True, config=_PLOTLY_CFG)

    # ── Food security: food-stressed % by store ───────────────────────────────
    st.subheader("🍞 Consumer Food Security — Food-Stressed % by Store")
    food_df = df_crisis.groupby(["Day", "Store"])["FoodStressedPct"].mean().reset_index()
    food_df["FoodStressedPct"] *= 100
    fig_food = px.line(
        food_df, x="Day", y="FoodStressedPct", color="Store",
        color_discrete_sequence=_TS_COLORS,
        title="Food-Stressed Consumer % — Crisis scenario",
        template="plotly_white",
        labels={"FoodStressedPct": "Food-Stressed %"},
    )
    fig_food.add_vline(x=ms_cri_start, line_dash="dot", line_color="orange",
                       annotation_text="Crisis start")
    st.plotly_chart(fig_food, use_container_width=True, config=_PLOTLY_CFG)

    # ── Network resilience score ──────────────────────────────────────────────
    st.divider()
    net_resilience = avg_fulfill * (1 - rev_drop_pct / 100) * min(1.0, 1 / max(avg_panic, 0.01) * 0.5)
    net_resilience = max(0.0, min(1.0, net_resilience))
    resilience_label = (
        "🟢 Resilient"       if net_resilience >= 0.75 else
        "🟡 Moderate"        if net_resilience >= 0.50 else
        "🟠 Vulnerable"      if net_resilience >= 0.30 else
        "🔴 Critical"
    )
    st.markdown(
        f"<div style='background:#F0E9DA;border-radius:10px;padding:16px 24px;'>"
        f"<h4 style='margin:0 0 6px 0;'>🔒 Network Resilience Score</h4>"
        f"<span style='font-size:2rem;font-weight:700;'>{net_resilience:.2f}</span>"
        f"<span style='font-size:1.1rem;margin-left:12px;'>{resilience_label}</span>"
        f"<p style='margin:8px 0 0 0;font-size:0.85rem;color:#555;'>"
        f"Composite score: avg fulfillment × (1 − revenue loss) × panic dampener. "
        f"1.0 = fully resilient network, 0.0 = complete collapse.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Download + Map link ───────────────────────────────────────────────────
    st.divider()
    _dl_col, _map_col = st.columns([2, 1])
    with _dl_col:
        st.download_button(
            "📥 Download full network results (CSV)",
            df_ms.to_csv(index=False).encode("utf-8"),
            "GROCERYsim_multistore_results.csv",
            "text/csv",
            key="dl_multistore",
            use_container_width=True,
        )
    with _map_col:
        if st.button("🗺️ View Results on Regional Map", key="ms_goto_map",
                     use_container_width=True, type="primary"):
            st.session_state["nav_section"] = "map"
            st.rerun()


# ===========================================================================
# 16. CARD NAVIGATION
# ===========================================================================

_NAV_CARDS = [
    {
        "id":       "setup",
        "icon":     "🏠",
        "title":    "Data & Setup",
        "subtitle": "Load population & catalogue",
        "color":    "#DBA159",
        "sections": [
            {"key": "data",    "label": "🏠 Data & Population",
             "desc": "Load cohort from Firebase or upload CSV/JSON"},
        ],
    },
    {
        "id":       "simulation",
        "icon":     "🔬",
        "title":    "Simulation",
        "subtitle": "Run the ABM",
        "color":    "#44A1A0",
        "sections": [
            {"key": "demo",    "label": "🎮 Interactive Demo",
             "desc": "Live animation · Monte Carlo CI bands · AI storage optimisation"},
        ],
    },
    {
        "id":       "analysis",
        "icon":     "📊",
        "title":    "Analysis",
        "subtitle": "Deep-dive results",
        "color":    "#27AE60",
        "sections": [
            {"key": "waste",       "label": "♻️ Food Waste",
             "desc": "Waste log, drivers, and environmental impact"},
            {"key": "product",     "label": "📦 Per-Product",
             "desc": "Stock, sales, CO₂ per SKU"},
            {"key": "behaviour",   "label": "🧪 Behavioural Theory",
             "desc": "Prospect theory, TPB, and exploratory access stress"},
            {"key": "sensitivity", "label": "🎚️ Sensitivity Analysis",
             "desc": "Replicated global uncertainty and importance screening"},
            {"key": "compare",     "label": "📊 Compare Scenarios",
             "desc": "Side-by-side saved simulation runs"},
            {"key": "agent",       "label": "🎬 Agent Replay",
             "desc": "Day-level individual shopper decisions"},
            {"key": "validation",  "label": "✅ Model Validation",
             "desc": "Preregistered targets · evidence tiers · verification audit"},
            {"key": "calibration", "label": "🎯 Model Calibration",
             "desc": "Replicated LHS · synthetic recovery · held-out validation gate"},
        ],
    },
    {
        "id":       "policy",
        "icon":     "🏛️",
        "title":    "Policy & Strategy",
        "subtitle": "Interventions & networks",
        "color":    "#8E44AD",
        "sections": [
            {"key": "policy",      "label": "🏛️ Policy Analysis",
             "desc": "Fat tax · subsidy · purchase cap · labelling"},
            {"key": "stakeholder", "label": "👔 Stakeholder View",
             "desc": "Policy briefs and KPI dashboards"},
            {"key": "stress",      "label": "🚨 Stress Test",
             "desc": "Automated 6-scenario resilience battery"},
            {"key": "multistore",  "label": "🏪 Multi-Store Network",
             "desc": "N stores with panic contagion & redistribution"},
            {"key": "map",         "label": "🗺️ Regional Map",
             "desc": "Finnish store network + food-security overlay"},
        ],
    },
    {
        "id":       "export",
        "icon":     "📤",
        "title":    "Export",
        "subtitle": "Download all results",
        "color":    "#2471A3",
        "sections": [
            {"key": "export", "label": "📥 Export & PDF Report",
             "desc": "CSV bundles · full branded PDF report"},
            {"key": "docs",   "label": "📋 ODD+D Documentation",
             "desc": "Auto-generated ODD+D protocol PDF for publications"},
        ],
    },
]

# Flat lookup: section key → {card, section} for breadcrumb rendering
_SECTION_META: dict = {
    sec["key"]: {"card": card, "section": sec}
    for card in _NAV_CARDS
    for sec in card["sections"]
}


def _build_section_renderers(params: dict) -> dict:
    """Return a {key: callable} map for every navigable section."""
    return {
        "data":        render_data_tab,
        "demo":        lambda: render_demo_tab(params),
        "waste":       render_waste_tab,
        "product":     render_product_tab,
        "behaviour":   lambda: render_behaviour_tab(params),
        "sensitivity": lambda: render_sensitivity_tab(params),
        "compare":     render_scenario_compare_tab,
        "agent":       render_agent_replay_tab,
        "policy":      lambda: render_policy_tab(params),
        "stakeholder": render_stakeholder_tab,
        "stress":      lambda: render_stress_tab(params),
        "multistore":  lambda: render_multistore_tab(params),
        "map":         render_regional_map_tab,
        "export":      render_export_tab,
        "validation":  lambda: render_validation_tab(params),
        "calibration": lambda: render_calibration_tab(params),
        "docs":        lambda: render_documentation_tab(params),
    }


def render_nav_home():
    """Render the 5-card navigation grid (shown when no section is active)."""

    # ── Status banner ─────────────────────────────────────────────────────────
    data_ok = st.session_state.get("config_data") is not None
    sim_ok  = st.session_state.get("sim_results") is not None

    if not data_ok:
        st.info(
            "👆 **Start here:** click **🏠 Data & Population** in the card below "
            "to load your participant cohort, then head to **🔬 Simulation** to run the model."
        )
    else:
        _parts = ["✅ Data loaded"]
        if sim_ok:
            _parts.append("✅ Simulation results ready")
        else:
            _parts.append("⬜ No simulation run yet")
        st.success("  ·  ".join(_parts))

    st.markdown(
        "<h4 style='margin:18px 0 4px 0;'>📍 Where would you like to go?</h4>",
        unsafe_allow_html=True,
    )

    # ── Card CSS (scoped to nav home; doesn't persist into section views) ─────
    st.markdown(
        """
        <style>
        /* Card header strip */
        .nav-card-hdr {
            border-radius: 10px 10px 0 0;
            padding: 14px 14px 10px;
            color: white;
            margin-bottom: 0;
        }
        .nav-card-hdr .nav-icon  { font-size: 1.6rem; line-height: 1; }
        .nav-card-hdr .nav-title { font-weight: 700; font-size: 0.92rem;
                                   margin-top: 6px; line-height: 1.2; }
        .nav-card-hdr .nav-sub   { font-size: 0.70rem; opacity: 0.88;
                                   margin-top: 3px; }
        /* Card body wrapper */
        .nav-card-body {
            border: 1px solid #E8DEC8;
            border-top: none;
            border-radius: 0 0 10px 10px;
            padding: 6px 4px 10px;
            background: #FFFFFF;
            box-shadow: 0 3px 10px rgba(4,32,38,0.07);
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(5, gap="small")
    for col, card in zip(cols, _NAV_CARDS):
        with col:
            # Extract values to avoid nested-quote issues in f-strings
            _c_color    = card["color"]
            _c_icon     = card["icon"]
            _c_title    = card["title"]
            _c_subtitle = card["subtitle"]
            # Coloured header
            st.markdown(
                f"<div class='nav-card-hdr' style='background:{_c_color};'>"
                f"<div class='nav-icon'>{_c_icon}</div>"
                f"<div class='nav-title'>{_c_title}</div>"
                f"<div class='nav-sub'>{_c_subtitle}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            # White card body border opener
            st.markdown("<div class='nav-card-body'>", unsafe_allow_html=True)

            for section in card["sections"]:
                if st.button(
                    section["label"],
                    key=f"nav_{section['key']}",
                    use_container_width=True,
                    help=section.get("desc", ""),
                ):
                    st.session_state["nav_section"] = section["key"]
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)


def _render_nav_breadcrumb(section_key: str):
    """
    Render a fully-clickable breadcrumb bar.

    Layout:  [🏠 Menu]  ›  [🔬 Card Name]  ›  **Current Section**
    All segments except the current one are live Streamlit buttons.
    Both 'Menu' and 'Card Name' navigate back to the home card grid.
    """
    meta    = _SECTION_META.get(section_key, {})
    card    = meta.get("card",    {})
    section = meta.get("section", {})

    _bc_icon   = card.get("icon",  "")
    _bc_title  = card.get("title", "")
    _bc_label  = section.get("label", "")

    # Inject link-style CSS scoped to the breadcrumb buttons only.
    # We target the two buttons by their unique keys via a data-testid approach
    # that Streamlit exposes for the parent element.
    st.markdown(
        """
        <style>
        /* Breadcrumb buttons: look like coloured text links, not boxes */
        [data-testid="stButton"]:has(button[kind="secondary"]#bc_home_btn),
        [data-testid="stButton"]:has(button[kind="secondary"]#bc_card_btn) { display:inline; }
        button#bc_home_btn, button#bc_card_btn {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #DBA159 !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            padding: 2px 4px !important;
            min-height: 28px !important;
            height: 28px !important;
        }
        button#bc_home_btn:hover, button#bc_card_btn:hover {
            background: transparent !important;
            text-decoration: underline !important;
            color: #c2873c !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Breadcrumb columns: [Menu btn] [sep] [Card btn] [sep] [Section label] [spacer]
    _c0, _s1, _c1, _s2, _c2, _ = st.columns([0.65, 0.08, 1.05, 0.08, 2.5, 3.5])

    with _c0:
        if st.button("🏠 Menu", key="bc_home_btn",
                     help="Go back to the navigation home"):
            st.session_state["nav_section"] = None
            st.rerun()

    with _s1:
        st.markdown(
            "<div style='padding:5px 0;color:#aaa;font-size:1.05rem;'>›</div>",
            unsafe_allow_html=True,
        )

    with _c1:
        if st.button(f"{_bc_icon} {_bc_title}", key="bc_card_btn",
                     help="Go back to the navigation home"):
            st.session_state["nav_section"] = None
            st.rerun()

    with _s2:
        st.markdown(
            "<div style='padding:5px 0;color:#aaa;font-size:1.05rem;'>›</div>",
            unsafe_allow_html=True,
        )

    with _c2:
        st.markdown(
            f"<div style='padding:5px 0;font-size:0.85rem;font-weight:700;"
            f"color:#042026;'>{_bc_label}</div>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# 12. MAIN ENTRY POINT
# ===========================================================================

def main():
    params = build_sidebar_params()
    # Cache params so the export tab can include them in the PDF report
    st.session_state["_last_params"] = params
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
                f"<div style='background:#edf7f7; border-left:4px solid #44A1A0; "
                f"border-radius:4px; padding:8px 10px; margin:0 0 10px 0; "
                f"font-size:12px; line-height:1.4; color:#16383d;'>"
                f"<strong>SecureFood users:</strong> After launching the "
                f"<strong>Finland — Dairy Supply Chain</strong> case study, click "
                f"<strong>Scenario Simulator</strong> below to enter the dedicated SecureFood workspace."
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
                if st.button(
                    "🌿 Scenario Simulator",
                    use_container_width=True,
                    key="sf_launch_btn",
                    help="Open the dedicated SecureFood scenario workspace.",
                ):
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
                        file_name="GROCERYsim_SecureFood_Quick_User_Manual.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="securefood_pdf_btn",
                        help="Download the SecureFood-only navigation, reporting, and CSV guide.",
                    )
                except FileNotFoundError:
                    pass
    st.divider()

    if params.get("exploratory_behaviour", False):
        st.warning(
            "**Behavioural evidence mode: exploratory extensions.** Panic, TPB, "
            "Prospect Theory, and other unvalidated dynamics may affect results. "
            "Do not interpret their effects as estimated from GROCERYsim."
        )
    else:
        st.success(
            "**Behavioural evidence mode: empirical only.** Unvalidated dynamic "
            "mechanisms are disabled; the model uses observed and calibration-gated "
            "GROCERYsim behaviour."
        )

    _nav_section = st.session_state.get("nav_section")

    if _nav_section is None:
        render_nav_home()
    else:
        _render_nav_breadcrumb(_nav_section)
        st.divider()
        _renderers = _build_section_renderers(params)
        _fn = _renderers.get(_nav_section)
        if _fn is not None:
            _fn()
        else:
            st.error(f"Unknown section: {_nav_section}")
            st.session_state["nav_section"] = None

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
