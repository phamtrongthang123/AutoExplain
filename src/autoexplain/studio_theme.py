"""Local studio styling; no remote fonts, copied assets or model dependencies."""
import streamlit as st


def apply():
    # Static CSS only. User/model content must never be interpolated here.
    # Keep Streamlit's toolbar and Stop control visible and usable.
    st.markdown("""
<style>
.stApp {
    --studio-accent: #087e73;
    --studio-line: #dbe5e3;
    background: #f8faf9;
    color: #182d2a;
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
[data-testid="stMainBlockContainer"] {
    max-width: 1440px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #edf3f1;
    border-right: 1px solid var(--studio-line);
}
h1, h2, h3 { letter-spacing: -0.035em; color: #182d2a; }
h1 { font-weight: 700; }
[data-testid="stCaptionContainer"] { color: #526965; }
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 20px !important;
    border-color: var(--studio-line) !important;
    background: #ffffff;
    box-shadow: 0 3px 14px rgba(24, 45, 42, 0.035);
}
[data-testid="stExpander"] {
    background: #ffffff;
    border-radius: 16px;
    border-color: var(--studio-line);
}
[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button,
[data-testid="stLinkButton"] a { border-radius: 12px; }
[data-testid="stButton"] button[kind="primary"] {
    background: var(--studio-accent);
    border-color: var(--studio-accent);
    color: #ffffff;
}
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input { border-radius: 10px; }
[data-testid="stMetric"] {
    padding: 0.8rem;
    border: 1px solid var(--studio-line);
    border-radius: 14px;
    background: #f6faf8;
}
[data-testid="stMetricValue"] { font-size: 1.15rem; }
button[role="tab"][aria-selected="true"] { color: #087e73 !important; }
[data-baseweb="tab-highlight"] { background-color: #087e73 !important; }
[data-testid="stCaptionContainer"] { color: #425c56 !important; }
[data-testid="stSidebar"] [data-testid="stButton"] { margin-bottom: -0.4rem; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 0.55rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 10px;
    padding: 0.45rem 0.65rem;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #d8ebe5;
    color: #12574d;
}
@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"] { padding: 1.5rem 1rem; }
}
</style>
""", unsafe_allow_html=True)
