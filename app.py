"""
AI Impact on Students: Major Category Predictor
====================================================
An ML-Powered Web Application that predicts a student's Major Category
based on their AI usage patterns and academic characteristics.

Built as part of IT325 - Performance Innovative Task 2
Extends Laboratory Activity 1 (Logistic Regression Classification)

Model: Logistic Regression Pipeline (preprocessor + classifier)
Target: Major_Category (Arts, Business, Humanities, Medical, STEM)
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Impact on Students – Major Category Predictor",
    page_icon="ml-icon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load SVG content for logo and headers
try:
    with open("ml-icon.svg", "r") as f:
        svg_content = f.read()
except Exception:
    svg_content = ""

# ──────────────────────────────────────────────────────────
# SVG Icon System (Lucide-style Inline SVGs)
# ──────────────────────────────────────────────────────────
SVG_ICONS = {
    "compass": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
    "info": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
    "user": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
    "chart-simple": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>',
    "award": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="6"/><path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/></svg>',
    "chart-pie": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>',
    "sliders": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/><line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="16" x2="16" y1="18" y2="22"/></svg>',
    "layer-group": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-10 5 10 5 10-5-10-5Z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/></svg>',
    "list-check": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="6" height="6" rx="1"/><path d="m3 17 2 2 4-4"/><path d="M13 6h8"/><path d="M13 12h8"/><path d="M13 18h8"/></svg>',
    "border-all": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/><path d="M15 3v18"/></svg>',
    "database": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/></svg>',
    "diagram-project": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="16" y="16" width="6" height="6" rx="1"/><rect x="2" y="16" width="6" height="6" rx="1"/><rect x="9" y="2" width="6" height="6" rx="1"/><path d="M12 8v4"/><path d="M12 12H5v4"/><path d="M12 12h7v4"/></svg>',
    "chart-line": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>',
}

MAJOR_SVGS = {
    "Arts": '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="13.5" cy="6.5" r=".5"/><circle cx="17.5" cy="10.5" r=".5"/><circle cx="8.5" cy="7.5" r=".5"/><circle cx="6.5" cy="12.5" r=".5"/><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.92 0 1.63-.77 1.63-1.7 0-.43-.16-.83-.41-1.16a.78.78 0 0 1-.18-.51c0-.43.35-.78.78-.78H15c4.97 0 9-4.03 9-9 0-4.97-4.03-9-9-9Z"/></svg>',
    "Business": '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>',
    "Humanities": '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>',
    "Medical": '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l1.5-2 2 4 1.5-2h3.78"/></svg>',
    "STEM": '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="1"/><path d="M16 12c0 2.2-1.8 4-4 4s-4-1.8-4-4 1.8-4 4-4 4 1.8 4 4Z"/><path d="M12 2c5.5 0 10 4.5 10 10S17.5 22 12 22 2 17.5 2 12 6.5 2 12 2Z"/></svg>'
}

# ──────────────────────────────────────────────────────────
# Custom CSS for a premium, polished look
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global styling */
html, body, [class*="css"] {
    font-family: 'Inter', var(--font, sans-serif);
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', var(--font, sans-serif) !important;
    font-weight: 700 !important;
    color: var(--text-color) !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: color-mix(in srgb, var(--secondary-background-color) 80%, transparent) !important;
    border-right: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
    font-family: 'Plus Jakarta Sans', var(--font, sans-serif) !important;
}
/* Sidebar Project Info card */
.sidebar-project-info {
    background-color: color-mix(in srgb, currentColor 6%, transparent) !important;
    padding: 1rem 1.15rem !important;
    border-radius: 10px !important;
    border: 1px solid color-mix(in srgb, currentColor 10%, transparent) !important;
    margin-top: 1.5rem !important;
}
.sidebar-project-info h4 {
    margin-top: 0 !important;
    margin-bottom: 0.4rem !important;
    font-size: 0.85rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.4rem !important;
    font-weight: 600 !important;
}
.sidebar-project-info h4 svg {
    color: #38bdf8 !important;
    width: 14px !important;
    height: 14px !important;
}
.sidebar-project-info p {
    font-size: 0.75rem !important;
    margin-bottom: 0.35rem !important;
    line-height: 1.45 !important;
    opacity: 0.75 !important;
}
.sidebar-project-info .subtext {
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    opacity: 0.5 !important;
}

/* Navigation section title in sidebar */
.sidebar-nav-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-color);
    margin-bottom: 0.75rem;
    padding-left: 0.85rem;
}
.sidebar-nav-title svg {
    color: #38bdf8;
}

/* Active menu item styling in sidebar */
div[data-testid="stRadio"] > div, div.row-widget.stRadio > div {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"], div.row-widget.stRadio label[data-baseweb="radio"] {
    display: flex !important;
    align-items: center !important;
    padding: 0.65rem 0.85rem !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    margin-bottom: 0.50rem !important;
    border: 1px solid transparent !important;
    cursor: pointer !important;
    color: var(--text-color) !important;
    background-color: transparent !important;
}
/* Hide the default radio dot indicator container */
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child,
div.row-widget.stRadio label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}
/* Style hover state */
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover, div.row-widget.stRadio label[data-baseweb="radio"]:hover {
    background-color: color-mix(in srgb, var(--text-color) 6%, transparent) !important;
}
/* Style active state */
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked), div.row-widget.stRadio label[data-baseweb="radio"]:has(input:checked) {
    background-color: color-mix(in srgb, #4f46e5 12%, transparent) !important;
    border-color: color-mix(in srgb, #4f46e5 25%, transparent) !important;
    color: #4f46e5 !important;
    font-weight: 600 !important;
}
/* Support dark mode theme override for active state */
@media (prefers-color-scheme: dark) {
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked), div.row-widget.stRadio label[data-baseweb="radio"]:has(input:checked) {
        background-color: color-mix(in srgb, #6366f1 18%, transparent) !important;
        border-color: color-mix(in srgb, #6366f1 35%, transparent) !important;
        color: #818cf8 !important;
    }
}
/* Add custom mask icons to the sidebar items */
div[data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(1)::before,
div.row-widget.stRadio label[data-baseweb="radio"]:nth-of-type(1)::before {
    content: "";
    display: inline-block;
    width: 18px;
    height: 18px;
    margin-right: 10px;
    background-color: currentColor;
    flex-shrink: 0;
    -webkit-mask: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M21 12H3"/><path d="M12 3v18"/></svg>') no-repeat center;
    mask: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M21 12H3"/><path d="M12 3v18"/></svg>') no-repeat center;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:nth-of-type(2)::before,
div.row-widget.stRadio label[data-baseweb="radio"]:nth-of-type(2)::before {
    content: "";
    display: inline-block;
    width: 18px;
    height: 18px;
    margin-right: 10px;
    background-color: currentColor;
    flex-shrink: 0;
    -webkit-mask: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>') no-repeat center;
    mask: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>') no-repeat center;
}

/* Header styling */
.main-header {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 1.5rem;
    padding: 2rem;
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.header-logo-container {
    width: 64px;
    height: 64px;
    flex-shrink: 0;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 8px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    display: flex;
    align-items: center;
    justify-content: center;
}
.header-logo-container svg {
    width: 100%;
    height: 100%;
}
.header-text-container {
    flex-grow: 1;
}
.header-text-container h1 {
    color: #ffffff !important;
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0 !important;
    padding: 0 !important;
    letter-spacing: -0.5px;
}
.header-text-container p {
    color: #93c5fd !important;
    font-size: 0.95rem;
    margin: 0.25rem 0 0 0 !important;
    font-weight: 400;
}

/* Info box */
.info-box {
    background: rgba(16, 185, 129, 0.08);
    border-left: 4px solid #10b981;
    padding: 1.25rem;
    border-radius: 8px;
    margin: 1.5rem 0;
    font-size: 0.95rem;
    color: var(--text-color);
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
}
.info-box svg {
    flex-shrink: 0;
    margin-top: 0.15rem;
    color: #10b981;
}

/* Metric cards */
.metric-card {
    background: color-mix(in srgb, var(--text-color) 4%, var(--background-color));
    border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.08);
    border-color: color-mix(in srgb, var(--text-color) 30%, transparent);
}
.metric-card .icon-container {
    font-size: 1.75rem;
    color: #4f46e5;
    background: color-mix(in srgb, #4f46e5 10%, transparent);
    width: 50px;
    height: 50px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.25rem;
}
.metric-card .icon-container svg {
    color: #4f46e5;
}
.metric-card .label {
    font-size: 0.8rem;
    color: color-mix(in srgb, var(--text-color) 70%, transparent);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-color);
    margin: 0;
}

/* Prediction result card */
.prediction-card {
    background: color-mix(in srgb, var(--text-color) 4%, var(--background-color));
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05), 0 8px 10px -6px rgba(0,0,0,0.05);
    margin: 2rem 0;
    border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
    transition: all 0.3s;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.prediction-icon-wrapper {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 84px;
    height: 84px;
    border-radius: 50%;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.03);
}
.prediction-label-title {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: color-mix(in srgb, var(--text-color) 70%, transparent);
    margin-top: 0.5rem;
    font-weight: 600;
}
.prediction-card .predicted-label {
    font-size: 2.75rem;
    font-weight: 800;
    margin: 0.25rem 0;
    letter-spacing: -0.5px;
}
.confidence-badge {
    display: inline-block;
    background: var(--background-color);
    color: var(--text-color);
    padding: 0.4rem 1.25rem;
    border-radius: 9999px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 1rem;
    border: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
}

/* Section headers */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-color);
    border-bottom: 2px solid color-mix(in srgb, var(--text-color) 10%, transparent);
    padding-bottom: 0.75rem;
    margin: 2.5rem 0 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.section-header svg {
    color: #4f46e5;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0;
    color: color-mix(in srgb, var(--text-color) 60%, transparent);
    font-size: 0.85rem;
    margin-top: 4rem;
    border-top: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
}

/* Streamlit Form Overrides */
div[data-testid="stForm"] {
    background-color: color-mix(in srgb, var(--text-color) 4%, var(--background-color));
    border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent) !important;
    border-radius: 16px;
    padding: 2rem !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
}

.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.3) !important;
    transform: translateY(-1.5px) !important;
}

@media (max-width: 768px) {
    .main-header {
        flex-direction: column;
        text-align: center;
        padding: 1.5rem;
        gap: 1rem;
    }
    .header-logo-container {
        margin: 0 auto;
    }
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Load model and metrics (cached)
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the pre-trained ML pipeline from disk."""
    return joblib.load("major_category_model_for_task2.pkl")


@st.cache_data
def load_metrics():
    """Load pre-computed model performance metrics."""
    with open("model_metrics.json", "r") as f:
        return json.load(f)


@st.cache_data
def load_dataset():
    """Load the original dataset for visualizations."""
    return pd.read_csv("data/ai_student_impact_dataset (1).csv")


model = load_model()
metrics = load_metrics()

# Try loading dataset; if unavailable, set to None
try:
    dataset = load_dataset()
except Exception:
    dataset = None


# ──────────────────────────────────────────────────────────
# Feature metadata used by the prediction form
# ──────────────────────────────────────────────────────────
FEATURE_META = {
    "Year_of_Study": {
        "type": "categorical",
        "options": ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
        "label": "Year of Study",
        "help": "The student's current academic year.",
    },
    "Pre_Semester_GPA": {
        "type": "numerical",
        "min": 1.0,
        "max": 4.0,
        "step": 0.01,
        "default": 3.0,
        "label": "Pre-Semester GPA",
        "help": "The student's GPA before the semester (1.0 – 4.0).",
    },
    "Weekly_GenAI_Hours": {
        "type": "numerical",
        "min": 0.0,
        "max": 40.0,
        "step": 0.5,
        "default": 8.0,
        "label": "Weekly GenAI Hours",
        "help": "Hours per week spent using Generative AI tools.",
    },
    "Primary_Use_Case": {
        "type": "categorical",
        "options": [
            "Copywriting/Drafting",
            "Debugging/Troubleshooting",
            "Direct_Answer_Generation",
            "Ideation",
            "Summarizing_Reading",
        ],
        "label": "Primary AI Use Case",
        "help": "The main way the student uses AI tools.",
    },
    "Prompt_Engineering_Skill": {
        "type": "categorical",
        "options": ["Beginner", "Intermediate", "Advanced"],
        "label": "Prompt Engineering Skill",
        "help": "Self-assessed skill level in crafting AI prompts.",
    },
    "Tool_Diversity": {
        "type": "numerical",
        "min": 1,
        "max": 5,
        "step": 1,
        "default": 3,
        "label": "Tool Diversity",
        "help": "Number of distinct AI tools the student regularly uses (1-5).",
    },
    "Traditional_Study_Hours": {
        "type": "numerical",
        "min": 1.0,
        "max": 36.0,
        "step": 0.5,
        "default": 11.0,
        "label": "Traditional Study Hours",
        "help": "Hours per week spent studying without AI tools.",
    },
    "Perceived_AI_Dependency": {
        "type": "numerical",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 4,
        "label": "Perceived AI Dependency",
        "help": "Self-rated dependency on AI (1 = low, 10 = high).",
    },
    "Institutional_Policy": {
        "type": "categorical",
        "options": ["Actively_Encouraged", "Allowed_With_Citation", "Strict_Ban"],
        "label": "Institutional AI Policy",
        "help": "The institution's policy regarding AI use.",
    },
    "Anxiety_Level_During_Exams": {
        "type": "numerical",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 4,
        "label": "Anxiety Level During Exams",
        "help": "Self-rated anxiety level during exams (1 = low, 10 = high).",
    },
    "Post_Semester_GPA": {
        "type": "numerical",
        "min": 1.0,
        "max": 4.0,
        "step": 0.01,
        "default": 3.3,
        "label": "Post-Semester GPA",
        "help": "The student's GPA after the semester (1.0 – 4.0).",
    },
    "Skill_Retention_Score": {
        "type": "numerical",
        "min": 10.0,
        "max": 100.0,
        "step": 0.5,
        "default": 75.0,
        "label": "Skill Retention Score",
        "help": "Score measuring how well the student retains skills (10-100).",
    },
    "Burnout_Risk_Level": {
        "type": "categorical",
        "options": ["Low", "Medium", "High"],
        "label": "Burnout Risk Level",
        "help": "Assessed burnout risk level.",
    },
}

MAJOR_COLOR = {
    "Arts": "#6366f1",
    "Business": "#0ea5e9",
    "Humanities": "#10b981",
    "Medical": "#8b5cf6",
    "STEM": "#f59e0b",
}


# ──────────────────────────────────────────────────────────
# Sidebar – Navigation
# ──────────────────────────────────────────────────────────
st.sidebar.markdown(f'<div class="sidebar-nav-title">{SVG_ICONS["compass"]} Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Go to",
    ["Prediction Hub", "Model Analysis"],
    label_visibility="collapsed",
)

st.sidebar.markdown(
    f'<div class="sidebar-project-info">'
    f'<h4>{SVG_ICONS["info"]} Project Info</h4>'
    f'<p>This application extends <strong>Laboratory Activity 1</strong> to predict student major categories using GenAI footprints.</p>'
    f'<div class="subtext">IT325 Final Term Activity</div>'
    f'</div>',
    unsafe_allow_html=True
)


# ══════════════════════════════════════════════════════════
# PAGE 1: Prediction Hub
# ══════════════════════════════════════════════════════════
if page == "Prediction Hub":
    # Header
    st.markdown(f"""
<div class="main-header">
<div class="header-logo-container">
{svg_content}
</div>
<div class="header-text-container">
<h1>AI Student Major Category Predictor</h1>
<p>Predicting academic paths from student Generative AI footprint</p>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        f"""
<div class="info-box">
{SVG_ICONS["info"]}
<div>
<strong>How it works:</strong> Fill in the student's academic and AI-usage
profile below. The model — a Logistic Regression pipeline trained on
50,000 student records — will predict the most likely <em>Major Category</em>
along with confidence scores for every class.
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Input Form ────────────────────────────────────────
    with st.form("prediction_form"):
        st.markdown(f'<div class="section-header">{SVG_ICONS["user"]} Student Profile</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        input_data = {}
        feature_keys = list(FEATURE_META.keys())

        for idx, feat in enumerate(feature_keys):
            meta = FEATURE_META[feat]
            target_col = [col1, col2, col3][idx % 3]
            with target_col:
                if meta["type"] == "categorical":
                    input_data[feat] = st.selectbox(
                        meta["label"],
                        options=meta["options"],
                        help=meta["help"],
                        key=f"input_{feat}",
                    )
                else:
                    if isinstance(meta["step"], int):
                        input_data[feat] = st.number_input(
                            meta["label"],
                            min_value=meta["min"],
                            max_value=meta["max"],
                            value=meta["default"],
                            step=meta["step"],
                            help=meta["help"],
                            key=f"input_{feat}",
                        )
                    else:
                        input_data[feat] = st.number_input(
                            meta["label"],
                            min_value=float(meta["min"]),
                            max_value=float(meta["max"]),
                            value=float(meta["default"]),
                            step=float(meta["step"]),
                            help=meta["help"],
                            key=f"input_{feat}",
                        )

        submitted = st.form_submit_button("Run Prediction Analysis")

    # ── Prediction Output ─────────────────────────────────
    if submitted:
        # Build DataFrame
        input_df = pd.DataFrame([input_data])

        # Validate
        errors = []
        if input_data["Pre_Semester_GPA"] < 1.0 or input_data["Pre_Semester_GPA"] > 4.0:
            errors.append("Pre-Semester GPA must be between 1.0 and 4.0.")
        if input_data["Post_Semester_GPA"] < 1.0 or input_data["Post_Semester_GPA"] > 4.0:
            errors.append("Post-Semester GPA must be between 1.0 and 4.0.")
        if input_data["Weekly_GenAI_Hours"] < 0:
            errors.append("Weekly GenAI Hours cannot be negative.")

        if errors:
            for e in errors:
                st.error(f"⚠️ {e}")
        else:
            try:
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)
                confidence = max(probabilities) * 100

                icon_svg = MAJOR_SVGS.get(prediction, SVG_ICONS["user"])
                color = MAJOR_COLOR.get(prediction, "#4f46e5")

                st.markdown(
                    f"""
<div class="prediction-card" style="border-top: 6px solid {color};">
<div class="prediction-icon-wrapper" style="background: {color}15; color: {color};">
{icon_svg}
</div>
<div class="prediction-label-title">Predicted Major Category</div>
<div class="predicted-label" style="color: {color};">{prediction}</div>
<div class="confidence-badge">Confidence: {confidence:.1f}%</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                # Probability bar chart
                st.markdown(f'<div class="section-header">{SVG_ICONS["chart-simple"]} Prediction Probabilities</div>', unsafe_allow_html=True)

                prob_df = pd.DataFrame(
                    {"Major Category": classes, "Probability (%)": [p * 100 for p in probabilities]}
                ).sort_values("Probability (%)", ascending=True)

                colors = [MAJOR_COLOR.get(c, "#888") for c in prob_df["Major Category"]]

                fig = go.Figure(
                    go.Bar(
                        x=prob_df["Probability (%)"],
                        y=prob_df["Major Category"],
                        orientation="h",
                        marker_color=colors,
                        text=[f"{p:.1f}%" for p in prob_df["Probability (%)"]],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=40, t=10, b=10),
                    xaxis_title="Probability (%)",
                    yaxis_title="",
                    xaxis=dict(range=[0, 100]),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif"),
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as exc:
                st.error(f"❌ Prediction failed: {exc}")


# ══════════════════════════════════════════════════════════
# PAGE 2: Model Analysis
# ══════════════════════════════════════════════════════════
elif page == "Model Analysis":
    st.markdown(f"""
<div class="main-header">
<div class="header-logo-container">
{svg_content}
</div>
<div class="header-text-container">
<h1>Model Performance & Analytics</h1>
<p>Comprehensive evaluations and distributions for the Student Major Category model</p>
</div>
</div>
""", unsafe_allow_html=True)

    # ── Model performance metrics ─────────────────────────
    st.markdown(f'<div class="section-header">{SVG_ICONS["award"]} Model Performance</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
<div class="metric-card">
<div class="icon-container">{SVG_ICONS["award"]}</div>
<div class="label">Accuracy</div>
<div class="value">{metrics['accuracy']*100:.1f}%</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
<div class="metric-card">
<div class="icon-container">{SVG_ICONS["chart-pie"]}</div>
<div class="label">F1 (Weighted)</div>
<div class="value">{metrics['f1_weighted']*100:.1f}%</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
<div class="metric-card">
<div class="icon-container">{SVG_ICONS["sliders"]}</div>
<div class="label">F1 (Macro)</div>
<div class="value">{metrics['f1_macro']*100:.1f}%</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
<div class="metric-card">
<div class="icon-container">{SVG_ICONS["layer-group"]}</div>
<div class="label">Num Classes</div>
<div class="value">{len(metrics['classes'])}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ── Per-class metrics table ───────────────────────────
    st.markdown(f'<div class="section-header">{SVG_ICONS["list-check"]} Per-Class Metrics</div>', unsafe_allow_html=True)

    report = metrics["report"]
    class_rows = []
    for cls in metrics["classes"]:
        if cls in report:
            class_rows.append(
                {
                    "Major Category": cls,
                    "Precision": f"{report[cls]['precision']:.2f}",
                    "Recall": f"{report[cls]['recall']:.2f}",
                    "F1-Score": f"{report[cls]['f1-score']:.2f}",
                    "Support": int(report[cls]["support"]),
                }
            )
    class_df = pd.DataFrame(class_rows)
    st.dataframe(class_df, use_container_width=True, hide_index=True)

    # ── Confusion Matrix ──────────────────────────────────
    st.markdown(f'<div class="section-header">{SVG_ICONS["border-all"]} Confusion Matrix</div>', unsafe_allow_html=True)

    cm = np.array(metrics["confusion_matrix"])
    classes = metrics["classes"]

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12, "family": "Inter, sans-serif"},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig_cm.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Dataset Info ──────────────────────────────────────
    st.markdown(f'<div class="section-header">{SVG_ICONS["database"]} Dataset Information</div>', unsafe_allow_html=True)

    st.markdown(
        """
    | Property | Value |
    |----------|-------|
    | **Dataset** | AI Impact on Students Dataset |
    | **Source** | [Kaggle – laveshjadon/ai-impact-on-students](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students) |
    | **Records** | 50,000 |
    | **Features** | 16 (13 used for prediction) |
    | **Target** | `Major_Category` (5 classes: Arts, Business, Humanities, Medical, STEM) |
    | **Train/Test Split** | 80% / 20% |
    """
    )

    # ── Feature list ──────────────────────────────────────
    st.markdown(f'<div class="section-header">{SVG_ICONS["sliders"]} Features Used</div>', unsafe_allow_html=True)

    num_features = [
        "Pre_Semester_GPA",
        "Weekly_GenAI_Hours",
        "Tool_Diversity",
        "Traditional_Study_Hours",
        "Perceived_AI_Dependency",
        "Anxiety_Level_During_Exams",
        "Post_Semester_GPA",
        "Skill_Retention_Score",
    ]
    cat_features = [
        "Year_of_Study",
        "Primary_Use_Case",
        "Prompt_Engineering_Skill",
        "Institutional_Policy",
        "Burnout_Risk_Level",
    ]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Numerical Features** (StandardScaler)")
        for f in num_features:
            st.markdown(f"- `{f}`")
    with c2:
        st.markdown("**Categorical Features** (OneHotEncoder)")
        for f in cat_features:
            st.markdown(f"- `{f}`")

    # ── Model Architecture ────────────────────────────────
    st.markdown(f'<div class="section-header">{SVG_ICONS["diagram-project"]} Model Architecture</div>', unsafe_allow_html=True)

    st.markdown(
        """
    The model is a **scikit-learn Pipeline** with two main stages:

    1. **Preprocessor** (`ColumnTransformer`)
       - *Numerical pipeline*: `SimpleImputer` (median) → `StandardScaler`
       - *Categorical pipeline*: `SimpleImputer` (most frequent) → `OneHotEncoder`
    2. **Classifier** — `LogisticRegression`
       - Solver: `lbfgs`
       - Penalty: `L2`
       - Max iterations: `2000`
       - Multi-class strategy: One-vs-Rest (OvR)

    The model was trained and evaluated as part of **Laboratory Activity 1**
    and saved using `joblib` / `pickle`.
    """
    )

    # ── Dataset Visualizations (if dataset available) ─────
    if dataset is not None:
        st.markdown(f'<div class="section-header">{SVG_ICONS["chart-line"]} Dataset Visualizations</div>', unsafe_allow_html=True)

        # Class distribution
        class_counts = dataset["Major_Category"].value_counts().reset_index()
        class_counts.columns = ["Major Category", "Count"]
        colors = [MAJOR_COLOR.get(c, "#888") for c in class_counts["Major Category"]]

        fig_dist = go.Figure(
            go.Bar(
                x=class_counts["Major Category"],
                y=class_counts["Count"],
                marker_color=colors,
                text=class_counts["Count"],
                textposition="outside",
            )
        )
        fig_dist.update_layout(
            title="Class Distribution (Major Category)",
            height=400,
            xaxis_title="Major Category",
            yaxis_title="Count",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # GPA distribution by major
        fig_gpa = px.box(
            dataset,
            x="Major_Category",
            y="Pre_Semester_GPA",
            color="Major_Category",
            color_discrete_map=MAJOR_COLOR,
            title="Pre-Semester GPA Distribution by Major",
        )
        fig_gpa.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_gpa, use_container_width=True)

        # Weekly GenAI Hours by major
        fig_ai = px.violin(
            dataset,
            x="Major_Category",
            y="Weekly_GenAI_Hours",
            color="Major_Category",
            color_discrete_map=MAJOR_COLOR,
            title="Weekly GenAI Hours by Major",
            box=True,
        )
        fig_ai.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_ai, use_container_width=True)


# ──────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────
st.markdown(
    """
<div class="footer">
    <p>AI Impact on Students – Major Category Predictor &nbsp;|&nbsp;
    IT325 Final Term – Performance Innovative Task 2 &nbsp;|&nbsp;
    Built with Streamlit + scikit-learn</p>
</div>
""",
    unsafe_allow_html=True,
)
