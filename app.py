"""
TruthPulse: Live Fake News Verification Engine.
Real-time NLP Misinformation Screening & Explainable AI (XAI) Analysis.
"""

import os
import sys
import pandas as pd
import streamlit as st

# Configure sys.path so src imports work reliably
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import extract_linguistic_features
from model import FakeNewsModelManager
from explainability import explain_prediction, highlight_html_text

# Page Config
st.set_page_config(
    page_title="TruthPulse | Fake News Verification Engine",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme-Adaptive CSS (Dark & Light Mode Compatible)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 700;
    }
    .metric-lbl {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    .badge-fake {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1.5px solid #ef4444;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.3rem;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-real {
        background-color: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1.5px solid #22c55e;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.3rem;
        display: inline-block;
        margin-bottom: 8px;
    }
    .highlight-box {
        background-color: rgba(148, 163, 184, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 8px;
        padding: 18px;
        font-size: 1rem;
        line-height: 1.75;
        margin-top: 10px;
    }
    .keyword-chip-fake {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 4px 11px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .keyword-chip-real {
        display: inline-block;
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 16px;
        padding: 4px 11px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Model Manager
@st.cache_resource
def get_manager():
    manager = FakeNewsModelManager()
    manager.load_artifacts()
    return manager

manager = get_manager()

# Sidebar Information & Sample Article Presets
with st.sidebar:
    st.image("https://img.icons8.com/color/96/news.png", width=70)
    st.markdown("### **TruthPulse AI**")
    st.markdown("**Live Fake News Verification Engine**")
    st.divider()

    st.markdown("#### **Active ML Pipeline**")
    st.write(f"• **Classifier**: `{manager.best_model_name}`")
    st.write("• **Vectorizer**: `TF-IDF (1, 2) n-grams`")
    st.write("• **Features**: `5,000 max vocabulary`")
    st.write("• **Explainability**: Model Weight Attribution")
    st.divider()

    st.markdown("#### **Sample Article Presets**")
    sample_presets = {
        "Select a preset...": {"title": "", "text": ""},
        "Fake: Secret Kitchen Lemon Juice Cures All Cancers": {
            "title": "SHOCKING SECRET: Common Kitchen Lemon Juice Completely Cures All Cancers in 48 Hours!",
            "text": "Big Pharma doesn't want you to know this suppressed natural secret! Top whistleblowers inside classified medical institutions have revealed that drinking hot boiled lemon water infused with baking soda destroys 100% of malignant cancer cells within exactly 48 hours. Medical doctors are secretly sworn to secrecy under threat of license revocation because chemotherapy brings in trillions in profits. Share this urgent message before the mainstream media wipes this website from the internet forever!"
        },
        "Real: FDA Approves Breakthrough Gene Therapy": {
            "title": "FDA Approves Breakthrough Gene Therapy for Rare Sickle Cell Disease",
            "text": "The Food and Drug Administration has granted regulatory approval to a groundbreaking CRISPR-based gene therapy designed to treat patients aged 12 and older suffering from severe sickle cell disease. Clinical trials demonstrated that over 90% of participating patients were completely free of severe vaso-occlusive pain crises for at least 12 consecutive months following the single-dose cellular infusion. Health authorities praised the milestone as a transformative step for genomic medicine."
        },
        "Fake: NASA Photos Prove Earth is Flat": {
            "title": "Leaked NASA Satellite Photos Prove Earth Is Actually Completely Flat and Surrounded by Giant Ice Wall",
            "text": "A brave anonymous cybersecurity specialist has breached NASA's classified supercomputer servers, downloading unaltered panoramic imagery showing our planet is an expansive flat plane enclosed by an impenetrable 300-foot ice wall guarded by armed international naval warships! The round globe narrative was concocted by the military-industrial complex to extract trillions in fabricated satellite research funding. NASA's composite spherical images are 100% computer-generated CGI artwork!"
        },
        "Real: JWST Discovers Carbon Dioxide on Exoplanet": {
            "title": "James Webb Space Telescope Identifies Atmospheric Carbon Dioxide on Distant Exoplanet",
            "text": "Astrophysicists analyzing transmission spectroscopy data from the James Webb Space Telescope have confirmed clear chemical signatures of carbon dioxide in the upper atmosphere of gas giant exoplanet WASP-39b, situated roughly 700 light-years from Earth. The observation marks the first definitive detection of CO2 on an extrasolar planet, providing crucial observational benchmarks for atmospheric modeling and opening new avenues to examine planetary formation dynamics across stellar systems."
        },
        "Fake: Glitch Allows Unlimited ATM Cash Withdrawals": {
            "title": "INSANE GLITCH: Secret Banking Code Allows Anyone to Withdraw Unlimited Cash From Any ATM",
            "text": "A disgruntled banking software developer has leaked an emergency master maintenance passcode that forces any ATM machine in the country to spit out thousands in twenties without charging your debit card! The secret override was created by financial elites for emergency cash liquidity during military crises. Try typing *99# followed by your birth year twice at any terminal tonight before the central bank patches the mainframe network! Share this viral secret now!"
        }
    }

    sample_choice = st.selectbox(
        "Load demo article:",
        list(sample_presets.keys())
    )

# Manage input values
if "curr_title" not in st.session_state:
    st.session_state["curr_title"] = ""
if "curr_text" not in st.session_state:
    st.session_state["curr_text"] = ""

if sample_choice != "Select a preset...":
    st.session_state["curr_title"] = sample_presets[sample_choice]["title"]
    st.session_state["curr_text"] = sample_presets[sample_choice]["text"]

# Main Page Header
st.markdown('<div class="main-title">📰 TruthPulse: Live Fake News Verification Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-Time Natural Language Misinformation Screening & Explainable AI (XAI) Analysis</div>', unsafe_allow_html=True)

# Input Section
st.markdown("### 📝 Input Article for Credibility Screening")
st.write("Enter the headline and full body text of any news article, or load a preset sample from the sidebar.")

col_in1, col_in2 = st.columns([4, 1])
with col_in1:
    input_title = st.text_input(
        "Article Headline / Title:",
        value=st.session_state["curr_title"],
        placeholder="e.g. Health Authorities Announce New Seasonal Guidelines..."
    )
with col_in2:
    st.write("")
    st.write("")
    if st.button("🔄 Clear Input", use_container_width=True):
        st.session_state["curr_title"] = ""
        st.session_state["curr_text"] = ""
        st.rerun()

input_text = st.text_area(
    "Article Body Text:",
    value=st.session_state["curr_text"],
    height=170,
    placeholder="Paste the complete news story, claim, or social media post here..."
)

analyze_clicked = st.button("🚀 Analyze Article Credibility", type="primary", use_container_width=True)

# Trigger analysis on button click or when sample is loaded
should_analyze = analyze_clicked or (input_title and input_text)

if should_analyze:
    if not input_text.strip() and not input_title.strip():
        st.warning("⚠️ Please provide an article title or body text to analyze.")
    else:
        with st.spinner("Analyzing text signals, calculating TF-IDF features, and running classifier..."):
            # Model prediction
            pred_result = manager.predict(input_title, input_text)
            is_fake = (pred_result["label"] == 1)

            # Linguistic & explainability analysis
            ling_features = extract_linguistic_features(input_text, input_title)
            explained_words = explain_prediction(f"{input_title} {input_text}", manager.best_model, manager.vectorizer, top_k=12)

        st.divider()

        # Verdict and Stylistic Cards
        col_v1, col_v2 = st.columns([1.4, 2])
        with col_v1:
            st.markdown("#### **Verification Verdict**")
            if is_fake:
                st.markdown('<div class="badge-fake">🚨 FAKE NEWS DETECTED</div>', unsafe_allow_html=True)
                st.markdown(f"**Confidence:** `{pred_result['confidence']}%` | **Risk Category:** High Misinformation")
            else:
                st.markdown('<div class="badge-real">✅ VERIFIED CREDIBLE NEWS</div>', unsafe_allow_html=True)
                st.markdown(f"**Confidence:** `{pred_result['confidence']}%` | **Risk Category:** Low / Reliable")

            st.progress(pred_result['fake_probability'] / 100.0)
            st.caption(f"Estimated Probability of Misinformation: {pred_result['fake_probability']}%")

        with col_v2:
            st.markdown("#### **Stylistic & Clickbait Indicators**")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{ling_features['sensationalism_score']}</div>
                    <div class="metric-lbl">Sensationalism (0-100)</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{ling_features['uppercase_ratio']}%</div>
                    <div class="metric-lbl">All-Caps Ratio</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{ling_features['exclamation_density']}%</div>
                    <div class="metric-lbl">Exclamation Density</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Explainable AI (XAI) Section
        col_x1, col_x2 = st.columns([1.2, 1.8])
        with col_x1:
            st.markdown("#### 🔬 **Key Trigger Keywords Identified**")
            st.write("Words detected in this text with strong model weight attribution:")

            if explained_words:
                for item in explained_words:
                    if item["indicates"].startswith("Fake"):
                        st.markdown(f'<span class="keyword-chip-fake">⚠️ {item["word"]} (+{item["impact"]})</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="keyword-chip-real">✓ {item["word"]} (-{item["impact"]})</span>', unsafe_allow_html=True)
            else:
                st.info("No specific high-leverage vocabulary outliers detected.")

            if ling_features["flagged_keywords"]:
                st.markdown("##### 🚩 Sensational Buzzwords Triggered:")
                st.write(", ".join([f"`{k}`" for k in ling_features["flagged_keywords"]]))

        with col_x2:
            st.markdown("#### 🎯 **Contextual Text Highlighting**")
            st.caption(
                "Colored tokens reflect statistical vocabulary weighting: "
                "<span style='color:#ef4444; font-weight:bold;'>Red = Fake-indicative</span> | "
                "<span style='color:#22c55e; font-weight:bold;'>Green = Credible-indicative</span>",
                unsafe_allow_html=True
            )
            highlighted_html = highlight_html_text(f"<strong>{input_title}</strong><br><br>{input_text}", explained_words)
            st.markdown(f'<div class="highlight-box">{highlighted_html}</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("TruthPulse AI | Live Fake News Verification Engine | Machine Learning & Explainable AI")
