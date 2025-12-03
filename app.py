# ============================================================
#  app.py — AI-KnowMap (Final fixed + mixed UI A+B)
# ============================================================

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import jwt
import time
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt

# Optional parse import retained for compatibility if used in modules
from dateutil import parser as dateparser

# ---------------------------
# Module imports (your modules)
# ---------------------------
sys.path.append(str(Path(__file__).parent / "modules"))

from modules.dataset_loader import load_dataset
from modules.nlp_pipeline import extract_entities_from_data, extract_relations_from_data
from modules.graph_builder import build_knowledge_graph, get_graph_stats, visualize_graph, search_subgraph
from modules.cache import cache
from modules.semantic_search import SemanticSearch
from modules.graph_builder import search_subgraph, visualize_graph


# ---------------------------
# Data & persistence paths
# ---------------------------
# Root of the app — used for static assets (read-only)
ROOT = Path(__file__).parent
# Use Render-safe writable directory (ephemeral)
DATA_DIR = Path("/tmp/ai_knowmap_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


USERS_FILE = DATA_DIR / "users.json"
GRAPHS_FILE = DATA_DIR / "saved_graphs.json"
TRIPLES_FILE = DATA_DIR / "triples.json"
ENTITIES_FILE = DATA_DIR / "entities.json"
LOGS_FILE = DATA_DIR / "logs.json"

for p in [USERS_FILE, GRAPHS_FILE, TRIPLES_FILE, ENTITIES_FILE, LOGS_FILE]:
    if not p.exists():
        p.write_text(json.dumps({} if p in [USERS_FILE, GRAPHS_FILE] else []))

def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {} if path in [USERS_FILE, GRAPHS_FILE] else []

def write_json(path: Path, data):
    path.write_text(json.dumps(data, default=str, indent=2))

# ---------------------------
# JWT / Auth config
# ---------------------------
JWT_SECRET = os.environ.get("AI_KNOWMAP_JWT_SECRET", "change_this_secret")
JWT_ALGO = "HS256"
JWT_EXP_MINUTES = 90

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_jwt(username: str) -> str:
    payload = {
        "sub": username,
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "exp": int((datetime.now(timezone.utc) + timedelta(minutes=JWT_EXP_MINUTES)).timestamp())
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    return token if isinstance(token, str) else token.decode()

def verify_jwt(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return True, payload.get("sub")
    except jwt.ExpiredSignatureError:
        return False, "Token expired"
    except Exception as e:
        return False, str(e)

# ---------------------------
# Users management (JSON)
# ---------------------------
def initialize_users_file():
    users = read_json(USERS_FILE)
    if not users:
        users = {
            "admin": {
                "password": hash_password("admin123"),
                "name": "Admin User",
                "role": "admin",
                "created": datetime.now(timezone.utc).isoformat(),
                "saved_graphs": [],
                "preferences": {}
            },
            "demo": {
                "password": hash_password("demo123"),
                "name": "Demo User",
                "role": "user",
                "created": datetime.now(timezone.utc).isoformat(),
                "saved_graphs": [],
                "preferences": {}
            }
        }
        write_json(USERS_FILE, users)
    return users

def register_user(username, password, name):
    users = read_json(USERS_FILE)
    if username in users:
        return False, "Username already exists"
    users[username] = {
        "password": hash_password(password),
        "name": name,
        "role": "user",
        "created": datetime.now(timezone.utc).isoformat(),
        "saved_graphs": [],
        "preferences": {}
    }
    write_json(USERS_FILE, users)
    add_log("INFO", f"User registered: {username}")
    return True, "Registered successfully"

def authenticate_user_jwt(username, password):
    users = read_json(USERS_FILE)
    if username not in users:
        return False, "User not found"
    if users[username]["password"] != hash_password(password):
        return False, "Invalid password"
    token = create_jwt(username)
    add_log("INFO", f"User logged in: {username}")
    return True, token

# ---------------------------
# Simple logging
# ---------------------------
def add_log(level, message):
    logs = read_json(LOGS_FILE) or []
    logs.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message
    })
    write_json(LOGS_FILE, logs)

# ---------------------------
# Triples/entities persistence
# ---------------------------
def save_triples(triples_df: pd.DataFrame):
    write_json(TRIPLES_FILE, triples_df.to_dict(orient="records"))
    add_log("INFO", f"Saved {len(triples_df)} triples")

def load_triples() -> pd.DataFrame:
    records = read_json(TRIPLES_FILE) or []
    return pd.DataFrame(records)

def save_entities(entities_df: pd.DataFrame):
    write_json(ENTITIES_FILE, entities_df.to_dict(orient="records"))
    add_log("INFO", f"Saved {len(entities_df)} entities")

def load_entities() -> pd.DataFrame:
    records = read_json(ENTITIES_FILE) or []
    return pd.DataFrame(records)

def save_graph_to_file(graph: nx.Graph, name: str):
    graphs = read_json(GRAPHS_FILE) or {}
    graphs[name] = nx.readwrite.json_graph.node_link_data(graph)
    write_json(GRAPHS_FILE, graphs)
    add_log("INFO", f"Graph saved: {name}")

def save_graph_png(graph: nx.Graph, filename: str):
    png_path = ROOT / "saved_graphs" / f"{filename}.png"
    png_path.parent.mkdir(exist_ok=True)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    return png_path

def save_pyvis_html(graph: nx.Graph, filename: str):
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(graph)
    html_path = ROOT / "saved_graphs" / f"{filename}.html"
    html_path.parent.mkdir(exist_ok=True)
    net.save_graph(str(html_path))
    return html_path

def load_graph_from_file(name: str):
    graphs = read_json(GRAPHS_FILE) or {}
    if name not in graphs:
        return None
    return nx.readwrite.json_graph.node_link_graph(graphs[name])

# === SAFE LOADER FOR ENTITIES (fixes KeyError forever) ===
def safe_load_entities():
    if df_ok(st.session_state.entities_data):
        # If it's already a proper DataFrame with 'entity' column → good
        if 'entity' in st.session_state.entities_data.columns:
            return st.session_state.entities_data
        # If it's a list of dicts (from saved graph) → convert
        elif isinstance(st.session_state.entities_data, list):
            df = pd.DataFrame(st.session_state.entities_data)
            if not df.empty and 'entity' in df.columns:
                st.session_state.entities_data = df
                return df
        # Fallback: try to extract from any column that looks like entities
        for col in st.session_state.entities_data.columns:
            if 'entit' in col.lower():
                return st.session_state.entities_data
    return pd.DataFrame(columns=["entity"])  # empty safe version


def get_document_chat_answer(question: str):
    if not df_ok(st.session_state.data):
        return "No documents loaded yet. Please upload a dataset first."

    question_lower = question.lower()
    answer = "Based on your uploaded documents and extracted knowledge:\n\n"
    found = 0

    if df_ok(st.session_state.triples_data):
        relevant = st.session_state.triples_data[
            st.session_state.triples_data.apply(
                lambda row: any(word in " ".join(row.astype(str)).lower() for word in question_lower.split()),
                axis=1
            )
        ]
        if not relevant.empty:
            for _, row in relevant.head(10).iterrows():
                answer += f"• {row['Subject']} → {row['Relation']} → {row['Object']}\n"
                found += 1

    if found > 0:
        return answer + f"\n\n(Found {found} matching facts)"
    else:
        return "I couldn't find specific information matching your question in the current dataset. Try running the full NLP pipeline first or ask a more specific question."
# ---------------------------
# Streamlit page config + CSS
# ---------------------------
st.set_page_config(page_title="AI-KnowMap", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* background & header */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg,#f7f8ff 0%, #ffffff 100%);
}
.main-header {
  background: linear-gradient(90deg,#5b2ee1,#2fa4f8);
  color: white;
  font-weight: 700;
  padding: 18px;
  border-radius: 10px;
  margin-bottom: 18px;
  box-shadow: 0 6px 18px rgba(43,58,120,0.06);
}
.card {
  background: white;
  padding: 12px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(43,58,120,0.04);
}
.small-card {
  background: #f6f8ff;
  padding: 12px;
  border-radius: 10px;
}
.feature-chip {
  display:inline-block;
  padding:6px 10px;
  margin:4px;
  border-radius:18px;
  background:#eef4ff;
  color:#27408b;
}
.footer-note {text-align:center;color:gray;padding:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;600&display=swap');
    
    /* Animated gradient background */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.5),
                        0 0 40px rgba(59, 130, 246, 0.3),
                        0 0 60px rgba(59, 130, 246, 0.2);
        }
        50% {
            box-shadow: 0 0 30px rgba(59, 130, 246, 0.8),
                        0 0 60px rgba(59, 130, 246, 0.5),
                        0 0 90px rgba(59, 130, 246, 0.3);
        }
    }
    
    @keyframes particleFloat {
        0% {
            transform: translateY(100vh) translateX(0) rotate(0deg);
            opacity: 0;
        }
        10% {
            opacity: 1;
        }
        90% {
            opacity: 1;
        }
        100% {
            transform: translateY(-100vh) translateX(100px) rotate(360deg);
            opacity: 0;
        }
    }
    
    /* Particle system */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
    }
    
    .particle {
        position: absolute;
        width: 4px;
        height: 4px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.8), transparent);
        border-radius: 50%;
        animation: particleFloat linear infinite;
    }
    
    .particle:nth-child(1) { left: 10%; animation-duration: 15s; animation-delay: 0s; }
    .particle:nth-child(2) { left: 20%; animation-duration: 18s; animation-delay: 2s; }
    .particle:nth-child(3) { left: 30%; animation-duration: 12s; animation-delay: 4s; }
    .particle:nth-child(4) { left: 40%; animation-duration: 20s; animation-delay: 1s; }
    .particle:nth-child(5) { left: 50%; animation-duration: 16s; animation-delay: 3s; }
    .particle:nth-child(6) { left: 60%; animation-duration: 14s; animation-delay: 5s; }
    .particle:nth-child(7) { left: 70%; animation-duration: 19s; animation-delay: 2.5s; }
    .particle:nth-child(8) { left: 80%; animation-duration: 13s; animation-delay: 4.5s; }
    .particle:nth-child(9) { left: 90%; animation-duration: 17s; animation-delay: 1.5s; }
    .particle:nth-child(10) { left: 15%; animation-duration: 21s; animation-delay: 3.5s; }
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(-45deg, #0a0a0f, #1a0e2e, #0f0a1f, #1e1430);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Title styling with enhanced glow */
    .login-title {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
        margin-bottom: 1rem;
        animation: float 3s ease-in-out infinite, glow 2s ease-in-out infinite;
        letter-spacing: 2px;
        position: relative;
        z-index: 2;
    }
    
    /* Subtitle with animation */
    .login-subtitle {
        text-align: center;
        color: #000000;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 3px;
        animation: slideIn 1s ease-out 0.3s both;
        position: relative;
        z-index: 2;
    }
    
    /* Tab styling with glassmorphism */
    .stTabs {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2rem;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        animation: slideIn 1s ease-out 0.5s both;
        position: relative;
        z-index: 2;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 15px;
        padding: 0.5rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: #000000;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.2);
        color: #fff;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.4), rgba(30, 64, 175, 0.4));
        color: black !important;
        border: 1px solid rgba(59, 130, 246, 0.5);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    
    /* Form styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 12px !important;
        color: black !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input:focus {
        border: 1px solid rgba(59, 130, 246, 0.8) !important;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.4) !important;
        background: rgba(255, 255, 255, 0.08) !important;
        transform: scale(1.02);
    }
    
    .stTextInput > label {
        color: #00000 !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Button styling with enhanced effects */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover:before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 25px rgba(59, 130, 246, 0.6) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Form submit button */
    .stForm button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
    }
    
    /* Subheader styling */
    .stMarkdown h3 {
        color: black !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
        font-size: 1.5rem !important;
    }
    
    /* Alert boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        border-left: 4px solid !important;
        animation: slideIn 0.5s ease-out;
    }
    
    /* Success alert */
    div[data-baseweb="notification"][kind="success"] {
        background: rgba(16, 185, 129, 0.1) !important;
        border-left-color: #10b981 !important;
    }
    
    /* Error alert */
    div[data-baseweb="notification"][kind="error"] {
        background: rgba(239, 68, 68, 0.1) !important;
        border-left-color: #ef4444 !important;
    }
    
    /* Warning alert */
    div[data-baseweb="notification"][kind="warning"] {
        background: rgba(245, 158, 11, 0.1) !important;
        border-left-color: #f59e0b !important;
    }
    
    /* Footer text */
    .footer-text {
        text-align: center;
        color: #9ca3af;
        font-size: 0.9rem;
        margin-top: 2rem;
        animation: slideIn 1s ease-out 0.7s both;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6, #1e40af);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
    }
</style>

<div class="particles">
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Session init helpers
# ---------------------------
def initialize_session():
    keys = ["jwt_token", "username", "user_data", "data", "entities_data", "triples_data", "graph", "semantic_engine"]
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = None

initialize_users_file()
initialize_session()

# ---------------------------
# Helper to check if DataFrame exists and non-empty
# ---------------------------
def df_ok(obj):
    return (obj is not None) and isinstance(obj, pd.DataFrame) and (not obj.empty)

# ---------------------------
# LOGIN UI (fixed: single form, demo button outside)
# ---------------------------
def login_page():
    # Apply the mesmerizing style first (make sure to call the style function before this)
    
    # Title with subtitle
    st.markdown('<h1 class="login-title">🧠 AI-KnowMap</h1>', unsafe_allow_html=True)
    st.markdown('<p class="login-subtitle">INTERDISCIPLINARY KNOWLEDGE MAPPER</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tabs = st.tabs(["🔐 Sign In", "📝 Register"])
        
        # SIGN IN TAB
        with tabs[0]:
            st.subheader("Welcome back")
            with st.form("login_form"):
                user = st.text_input("Username or Email", value="")
                pwd = st.text_input("Password", type="password")
                submit = st.form_submit_button("Sign In")
            
            demo = st.button("Demo Login")
            
            if submit:
                if user and pwd:
                    ok, res = authenticate_user_jwt(user, pwd)
                    if ok:
                        st.session_state.jwt_token = res
                        st.session_state.username = user
                        st.session_state.user_data = read_json(USERS_FILE).get(user, {})
                        st.success("Login successful")
                        st.rerun()
                    else:
                        st.error(res)
                else:
                    st.warning("Enter both username and password")
            
            if demo:
                ok, res = authenticate_user_jwt("demo", "demo123")
                if ok:
                    st.session_state.jwt_token = res
                    st.session_state.username = "demo"
                    st.session_state.user_data = read_json(USERS_FILE).get("demo", {})
                    st.success("Demo login successful")
                    st.rerun()
                else:
                    st.error("Demo login failed")
        
        # REGISTER TAB
        with tabs[1]:
            st.subheader("Create account")
            with st.form("reg_form"):
                new_user = st.text_input("Username")
                new_name = st.text_input("Full name")
                new_pwd = st.text_input("Password", type="password")
                new_pwd2 = st.text_input("Confirm password", type="password")
                reg_submit = st.form_submit_button("Register")
            
            if reg_submit:
                if not all([new_user, new_name, new_pwd, new_pwd2]):
                    st.warning("Fill all fields")
                elif new_pwd != new_pwd2:
                    st.error("Passwords do not match")
                else:
                    ok, msg = register_user(new_user, new_pwd, new_name)
                    if ok:
                        st.success("Account created. Please sign in.")
                    else:
                        st.error(msg)
        
        st.markdown('<p class="footer-text">Powered by AI • Built by TEAM-1</p>', unsafe_allow_html=True)

# If user not logged in, show login page and stop.
if not st.session_state.jwt_token:
    login_page()
    st.stop()

# Validate token
ok, sub = verify_jwt(st.session_state.jwt_token)
if not ok:
    st.warning("Session expired or invalid. Please sign in again.")
    st.session_state.clear()
    st.rerun()
else:
    # refresh user data
    users = read_json(USERS_FILE)
    st.session_state.user_data = users.get(sub, st.session_state.user_data)

# ---------------------------
# Sidebar (cleaned)
# ---------------------------
with st.sidebar:
    # Use uploaded screenshot as logo / preview (local path)
    logo_path = ROOT / "assets" / "logo.png"
    # If you don't have a logo in assets, we'll display the placeholder via URL
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.image("https://via.placeholder.com/200x80/5b2ee1/ffffff?text=AI-KnowMap", use_container_width=True)

    st.markdown(f"### {st.session_state.user_data.get('name', 'User')}")
    st.caption(f"@{st.session_state.username} • {st.session_state.user_data.get('role','user')}")

    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("#### Project Status")
    st.write("• Dataset:", "✅" if df_ok(st.session_state.data) else "⏳")
    st.write("• Entities:", "✅" if df_ok(st.session_state.entities_data) else "⏳")
    st.write("• Triples:", "✅" if df_ok(st.session_state.triples_data) else "⏳")
    st.write("• Graph:", "✅" if st.session_state.graph is not None else "⏳")


# ---------------------------
# Main tabs (A+B mix styling)
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Home", "📂 Dataset", "🧠 NLP Pipeline",
    "🌐 Knowledge Graph", "🔍 Semantic Search",
    "🛠 Admin"
])
# ---------------------------  
# Tab 1: PROFESSIONAL HOME + SMART PROGRESSIVE UI + CHAT ON RIGHT
# ---------------------------
with tab1:
    st.markdown('<div class="main-header">AI-KnowMap — Your Knowledge, Connected</div>', unsafe_allow_html=True)

    # === HERO WELCOME ===
    user_name = st.session_state.user_data.get('name', st.session_state.username).split()[0]
    current_hour = datetime.now().hour
    greeting = "Good morning" if current_hour < 12 else "Good afternoon" if current_hour < 18 else "Good evening"
    
    st.markdown(f"""
    <div style="text-align:center; padding: 2rem; background: rgba(91,46,225,0.12); border-radius: 18px; 
                backdrop-filter: blur(12px); border: 1px solid rgba(91,46,225,0.2); margin: 1.5rem 0;">
        <h2 style="margin:0; background: linear-gradient(90deg, #5b2ee1, #3b82f6); -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent;">
            {greeting}, {user_name}! 👋
        </h2>
        <p style="font-size:1.1rem; color:#555; margin:0.5rem 0;">
            Turn unstructured text into powerful, explorable knowledge networks.
        </p>
        <div style="display:flex; justify-content:center; gap:1.5rem; margin-top:1rem; flex-wrap:wrap;">
            <div style="text-align:center;">
                <div style="font-size:1.8rem; font-weight:bold; color:#5b2ee1;">{len(st.session_state.data) if df_ok(st.session_state.data) else 0}</div>
                <div style="font-size:0.85rem; color:#666;">Records</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.8rem; font-weight:bold; color:#3b82f6;">{len(st.session_state.entities_data) if df_ok(st.session_state.entities_data) else 0}</div>
                <div style="font-size:0.85rem; color:#666;">Entities</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.8rem; font-weight:bold; color:#8b5cf6;">{st.session_state.graph.number_of_nodes() if st.session_state.graph else 0}</div>
                <div style="font-size:0.85rem; color:#666;">Graph Nodes</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Import random once (fixes NameError)
    import random

    # Define tips
    tips = [
        "💡 Drag nodes in the graph to explore connections!",
        "🔍 Ask questions like: 'What companies work on quantum computing?'",
        "✏️ Use Admin → Graph Editor to merge duplicate entities",
        "📊 Export your graph as interactive HTML or high-res PNG",
        "💾 Your progress is saved automatically"
    ]

    # Check progress
    has_data = df_ok(st.session_state.data)
    has_entities = df_ok(st.session_state.entities_data)
    has_triples = df_ok(st.session_state.triples_data)
    has_graph = st.session_state.graph is not None and st.session_state.graph.number_of_nodes() > 0

    # === TWO COLUMNS: Dashboard + Chat ===
    col_main, col_chat = st.columns([2.8, 1.2], gap="large")

    with col_main:
        # If nothing done yet → clean welcome
        if not (has_data or has_entities or has_triples or st.session_state.graph):
            st.markdown("#### 🚀 Get Started in Seconds")
            st.info("You're all set! Start by loading a dataset or trying the demo below.")
            
            # Show some helpful cards
            st.markdown("#### 🎯 What can AI-KnowMap do?")
            feat1, feat2, feat3 = st.columns(3)
            with feat1:
                st.markdown("""
                <div class="small-card" style="padding:1.5rem;">
                    <div style="font-size:2.5rem; margin-bottom:0.5rem;">📚</div>
                    <strong>Extract Knowledge</strong><br>
                    <small style="color:#666;">Automatically identify entities and relationships from your documents</small>
                </div>
                """, unsafe_allow_html=True)
            with feat2:
                st.markdown("""
                <div class="small-card" style="padding:1.5rem;">
                    <div style="font-size:2.5rem; margin-bottom:0.5rem;">🕸️</div>
                    <strong>Visualize Connections</strong><br>
                    <small style="color:#666;">See how concepts relate in interactive network graphs</small>
                </div>
                """, unsafe_allow_html=True)
            with feat3:
                st.markdown("""
                <div class="small-card" style="padding:1.5rem;">
                    <div style="font-size:2.5rem; margin-bottom:0.5rem;">🔎</div>
                    <strong>Search Semantically</strong><br>
                    <small style="color:#666;">Find relevant information based on meaning, not just keywords</small>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            # Show progress metrics
            st.markdown("#### 📊 Current Project Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Records", len(st.session_state.data) if has_data else 0)
            c2.metric("Entities", len(st.session_state.entities_data) if has_entities else 0)
            c3.metric("Triples", len(st.session_state.triples_data) if has_triples else 0)
            c4.metric("Graph Nodes", st.session_state.graph.number_of_nodes() if st.session_state.graph else 0)

            if st.session_state.graph and st.session_state.graph.number_of_nodes() > 0:
                st.markdown("#### 🌐 Live Knowledge Graph Preview")
                preview_html = visualize_graph(st.session_state.graph, height="420px", physics=True)
                st.components.v1.html(preview_html, height=450, scrolling=False)

        # === PROGRESSIVE NEXT STEPS ===
        st.markdown("#### 🎯 Next Steps for You")

        if not has_data:
            st.markdown("""
            <div class="small-card" style="text-align:center; padding:1.8rem;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">📤</div>
                <strong>Upload or load a dataset</strong><br>
                <small style="color:#666;">Go to Dataset tab to get started</small>
            </div>
            """, unsafe_allow_html=True)

        elif not has_entities:
            st.markdown("""
            <div class="small-card" style="text-align:center; padding:1.8rem;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">🏷️</div>
                <strong>Run Entity Extraction</strong><br>
                <small style="color:#666;">Go to NLP Pipeline to extract entities</small>
            </div>
            """, unsafe_allow_html=True)

        elif not has_triples:
            st.markdown("""
            <div class="small-card" style="text-align:center; padding:1.8rem;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">🔗</div>
                <strong>Run Relation Extraction</strong><br>
                <small style="color:#666;">Go to NLP Pipeline to find relationships</small>
            </div>
            """, unsafe_allow_html=True)

        elif not st.session_state.graph or st.session_state.graph.number_of_nodes() == 0:
            st.markdown("""
            <div class="small-card" style="text-align:center; padding:1.8rem;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">🌍</div>
                <strong>Build your Knowledge Graph</strong><br>
                <small style="color:#666;">Go to Knowledge Graph tab to visualize</small>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.success("✅ **Completed!** Your knowledge graph is ready to explore")

                # === QUICK ACTIONS — ALWAYS VISIBLE FROM START & 100% WORKING ===
        st.markdown("#### Quick Actions")
        qa1, qa2, qa3 = st.columns(3)

        with qa1:
            if st.button("Load Demo Dataset", use_container_width=True, type="secondary"):
                demo_df = pd.DataFrame({
                    "title": ["Climate Change Impact", "AI in Healthcare", "Quantum Breakthrough", "Renewable Energy", "Space Exploration"],
                    "body": [
                        "Climate change is accelerating due to human activity and requires immediate policy action...",
                        "Artificial intelligence is transforming diagnostics and personalized medicine at unprecedented speed...",
                        "MIT researchers achieved stable quantum entanglement at room temperature...",
                        "Solar panel efficiency reached 47% in new perovskite-silicon tandem cells...",
                        "NASA's Artemis program aims to return humans to the Moon by 2026..."
                    ]
                })
                st.session_state.data = demo_df
                st.success("Demo dataset loaded successfully!")
                st.rerun()

        with qa2:
            if st.button("Run Full Demo Pipeline", use_container_width=True, type="primary"):
                with st.spinner("Running full demo pipeline... This takes ~15–25 seconds"):
                    try:
                        # 1. Load demo data if missing
                        if not df_ok(st.session_state.data):
                            demo_df = pd.DataFrame({
                                "title": [
                                    "Climate Change Impact 2025", "AI Revolution in Healthcare", 
                                    "Quantum Computing Breakthrough", "Renewable Energy Surge", 
                                    "NASA Artemis Program", "CRISPR Gene Editing Advances"
                                ],
                                "body": [
                                    "Global temperatures rise 1.5°C by 2030. Carbon markets expand rapidly. China leads in solar...",
                                    "AI diagnostics now outperform radiologists in 94% of cases. Google DeepMind launches AlphaDiagnose...",
                                    "IBM achieves 1000-qubit processor with 99.9% fidelity. Commercial quantum advantage expected 2027...",
                                    "Perovskite-silicon solar cells reach 48% efficiency. India becomes world leader in solar exports...",
                                    "Artemis III will land first woman and person of color on Moon in 2026. Lunar gateway station operational...",
                                    "CRISPR-Cas13 used to cure genetic blindness in clinical trials. FDA fast-tracks approval..."
                                ]
                            })
                            st.session_state.data = demo_df

                        # 2. Extract entities
                        if not df_ok(st.session_state.entities_data):
                            ents = extract_entities_from_data(st.session_state.data)
                            st.session_state.entities_data = ents
                            save_entities(ents)

                        # 3. Extract relations
                        if not df_ok(st.session_state.triples_data):
                            trips = extract_relations_from_data(st.session_state.data)
                            st.session_state.triples_data = trips
                            save_triples(trips)

                        # 4. Build graph
                        if st.session_state.graph is None or st.session_state.graph.number_of_nodes() == 0:
                            g = build_knowledge_graph(st.session_state.triples_data, max_nodes=200)
                            st.session_state.graph = g

                        st.success("Full demo pipeline completed! Your 3D graph is ready")
                        add_log("INFO", "Full demo pipeline executed from Home")
                        st.rerun()

                    except Exception as e:
                        st.error("Something went wrong during pipeline")
                        st.code(str(e))

        with qa3:
            if st.button("Start Fresh Project", use_container_width=True):
                keys_to_clear = [
                    "data", "entities_data", "triples_data", "graph",
                    "semantic_engine", "has_done_semantic_search"
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("All cleared! New project started.")
                add_log("INFO", "User started fresh project")
                st.rerun()

                    # === RECENT ACTIVITY — BACK TO ORIGINAL CLEAN POSITION (below Quick Actions) ===
        st.markdown("#### Recent Activity")
        
        logs = read_json(LOGS_FILE) or []
        recent = [l for l in logs[-6:] if "logged in" not in l.get("message", "").lower()]
        
        if recent:
            for log in reversed(recent):
                ts = log["timestamp"].split("T")[0].split("-")[1:]  # MM-DD
                ts = "/".join(ts)
                msg = log["message"]
                st.markdown(f"**{ts}** • {msg}")
        else:
            st.caption("No recent activity yet — start building your graph!")

        # === PRO TIP (back to bottom) ===
        import random
        tips = [
            "Drag nodes in the graph to explore connections!",
            "Use Admin → Graph Editor to merge duplicate entities",
            "Try the 3D view — hold Shift + drag to pan!",
            "Export your graph as interactive HTML anytime",
            "Your progress is saved automatically"
        ]

        # Pro Tip at bottom
        st.markdown(f"**💡 Pro Tip:** {random.choice(tips)}")

    # ========================================
    # RIGHT SIDE: BEAUTIFUL CHAT PANEL
    # ========================================
    with col_chat:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #5b2ee1, #3b82f6); color:white; padding:1.3rem; 
                    border-radius:16px; text-align:center; margin-bottom:1rem; box-shadow:0 10px 30px rgba(91,46,225,0.4);">
            <h3 style="margin:0; font-size:1.4rem;">💬 Knowledge Assistant</h3>
            <p style="margin:4px 0 0; font-size:0.9rem; opacity:0.9;">Ask questions about your data</p>
        </div>
        """, unsafe_allow_html=True)

        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Reduced height from 400 to 300
        chat_area = st.container(height=300)
        with chat_area:
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align:center; padding:2rem 1rem; color:#888;">
                    <div style="font-size:3rem; margin-bottom:0.5rem;">💭</div>
                    <p style="font-size:0.9rem; margin:0;">Start a conversation!</p>
                    <p style="font-size:0.8rem; color:#aaa; margin-top:0.5rem;">Try asking:<br>"What entities are in my data?"</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div style="background:#e0f2fe; padding:0.7rem; border-radius:10px; margin:0.5rem 0;">
                            <small style='color:#0c4a6e'><b>You:</b> {msg['content']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:#f3f4f6; padding:0.7rem; border-radius:10px; margin:0.5rem 0;">
                            <small style='color:#1f2937'><b>🤖 AI:</b> {msg['content']}</small>
                        </div>
                        """, unsafe_allow_html=True)

        # Input
        if prompt := st.chat_input("Ask anything about your data...", key="home_chat"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("🤔 Thinking..."):
                answer = get_document_chat_answer(prompt)

                if st.session_state.graph:
                    nodes = [w.strip(".,!?\"'") for w in prompt.split() if w.strip(".,!?\"'") in st.session_state.graph.nodes()]
                    if nodes:
                        answer += f"\n\n🔗 Related nodes: **{', '.join(nodes[:5])}**"

                st.session_state.messages.append({"role": "assistant", "content": answer})

            st.rerun()

        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()


# ---------------------------
# Tab 2: Dataset
# ---------------------------
with tab2:
    st.markdown('<div class="card"><h3>📂 Dataset Selection & Upload</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        ds_choice = st.radio("Source", ["Upload File", "Demo Dataset"], horizontal=True)
        if ds_choice == "Upload File":
            uploaded = st.file_uploader("Upload CSV or TXT", type=["csv","txt"])
            if uploaded is not None:
                try:
                    df = load_dataset(uploaded)
                    st.session_state.data = df
                    st.success(f"Loaded {len(df)} records")
                    st.dataframe(df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        else:
            if st.button("Load Demo Dataset"):
                demo_df = pd.DataFrame({
                    "title": [
                        "Climate Change Impact",
                        "AI in Healthcare",
                        "Quantum Computing Breakthrough",
                        "Renewable Energy Solutions",
                        "Space Exploration Updates"
                    ],
                    "body": [
                        "Climate change is affecting global weather patterns...",
                        "Artificial intelligence is revolutionizing healthcare...",
                        "Researchers at MIT announced a major breakthrough...",
                        "Solar and wind energy are becoming more efficient...",
                        "NASA launched a mission to outer planets..."
                    ]
                })
                st.session_state.data = demo_df
                st.success("Demo dataset loaded")
                st.dataframe(demo_df.head(), use_container_width=True)

    with col2:
        st.markdown('<div class="small-card">', unsafe_allow_html=True)
        st.markdown("#### Format Notes")
        st.write("- CSV: needs 'body' or 'text' column")
        st.write("- TXT: one document per line")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: NLP Pipeline
with tab3:
    st.markdown('<div class="card"><h3>🧠 NLP Processing Pipeline</h3>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first in the Upload Data tab!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏷️ Entity Extraction")
            st.markdown("Extract named entities (persons, organizations, locations, etc.) from your text.")
            
            if st.button("🚀 Run Entity Extraction", type="primary", use_container_width=True):
                with st.spinner("Extracting entities... This may take a while."):
                    entities_data = extract_entities_from_data(st.session_state.data)
                    st.session_state.entities_data = entities_data
                    st.success(f"✅ Extracted entities from {len(entities_data)} records!")
            
            if st.session_state.entities_data is not None:
                st.markdown("##### 📊 Entity Statistics")
                # Flatten entities for analysis
                all_entities = []
                for entities_list in st.session_state.entities_data['entities']:
                    all_entities.extend(entities_list)
                
                if all_entities:
                    entity_df = pd.DataFrame(all_entities, columns=['Entity', 'Type'])
                    entity_counts = entity_df['Type'].value_counts()
                    
                    st.bar_chart(entity_counts)
                    
                    st.markdown("##### 🔝 Top Entities by Type")
                    for entity_type in entity_counts.head(5).index:
                        top_entities = entity_df[entity_df['Type'] == entity_type]['Entity'].value_counts().head(5)
                        with st.expander(f"{entity_type} ({entity_counts[entity_type]} total)"):
                            st.write(top_entities)
        
        with col2:
            st.markdown("#### 🔗 Relation Extraction")
            st.markdown("Extract subject-relation-object triples from your text.")
            
            if st.button("🚀 Run Relation Extraction", type="primary", use_container_width=True):
                with st.spinner("Ext    racting relations... This may take a while."):
                    triples_data = extract_relations_from_data(st.session_state.data)
                    st.session_state.triples_data = triples_data
                    st.success(f"✅ Extracted {len(triples_data)} triples!")
            
            if st.session_state.triples_data is not None:
                st.markdown("##### 📊 Relation Statistics")
                
                relation_counts = st.session_state.triples_data['Relation'].value_counts()
                st.bar_chart(relation_counts.head(10))
                
                st.markdown("##### 🔝 Top Relations")
                st.write(relation_counts.head(10))
        
        # Display extracted data
        if st.session_state.entities_data is not None or st.session_state.triples_data is not None:
            st.markdown("---")
            st.markdown("#### 📋 Extracted Data Preview")
            
            tab_ent, tab_rel = st.tabs(["Entities", "Relations"])
            
            with tab_ent:
                if st.session_state.entities_data is not None:
                    st.dataframe(st.session_state.entities_data.head(20), use_container_width=True)
                    st.download_button(
                        "⬇️ Download Entities CSV",
                        st.session_state.entities_data.to_csv(index=False),
                        "entities_extracted.csv",
                        "text/csv"
                    )
            
            with tab_rel:
                if st.session_state.triples_data is not None:
                    st.dataframe(st.session_state.triples_data.head(20), use_container_width=True)
                    st.download_button(
                        "⬇️ Download Triples CSV",
                        st.session_state.triples_data.to_csv(index=False),
                        "extracted_triples.csv",
                        "text/csv"
                    )

# ==================== TAB 4: KNOWLEDGE GRAPH (BEST OF BOTH WORLDS) ====================
with tab4:
    st.markdown("#### Interactive Knowledge Graph")

    # === Prerequisites Check ===
    if not df_ok(st.session_state.data):
        st.warning("Please upload a dataset first in the **Dataset** tab.")
        st.info("→ Go to **Dataset** tab → Upload or click **Load Demo Dataset**")
        st.stop()

    if not df_ok(st.session_state.entities_data):
        st.warning("Entities not extracted yet!")
        st.info("→ Go to **NLP Pipeline** tab → Click **Extract Entities** first")
        st.stop()

    if not df_ok(st.session_state.triples_data):
        st.warning("Relations not extracted yet!")
        st.info("→ Go to **NLP Pipeline** tab → Click **Extract Relations**")
        st.stop()

    # =================================================================
    # ONE CLEAN COLUMN: Everything centered and beautiful
    # =================================================================
    if st.session_state.get("graph") is None:
        # ——— NO GRAPH YET: Show big build button ———
        st.info("Your triples are ready! Click below to build your knowledge universe")
        col_btn, col_slider = st.columns([2, 1])
        with col_btn:
            if st.button("Build & Visualize Knowledge Graph", type="primary", use_container_width=True):
                with col_slider:
                    max_nodes = st.slider("Max Nodes", 10, 1000, 300, key="temp_max_nodes")
                with st.spinner("Building your knowledge graph..."):
                    G = build_knowledge_graph(st.session_state.triples_data, max_nodes=max_nodes)
                    st.session_state.graph = G
                    st.success(f"Graph built! {G.number_of_nodes()} nodes • {G.number_of_edges()} edges")
                    add_log("INFO", "Knowledge graph built from tab")
                    st.rerun()
    else:
        # ——— GRAPH EXISTS: Show everything beautifully ———
        view_mode = st.radio(
            "View Mode", 
            ["2D", "3D"], 
            horizontal=True, 
            index=0 if st.session_state.get("last_view_mode", "2D") == "2D" else 1,
            key="kg_view_mode"
        )
        st.session_state.last_view_mode = view_mode  # remember choice

        with st.spinner(f"Rendering in {view_mode} mode..."):
            html = visualize_graph(
                graph=st.session_state.graph,
                height="720px",
                physics=(view_mode == "2D"),
                three_d=(view_mode == "3D"),
                auto_cluster=True,
                node_size_multiplier=9,
                font_size=14
            )
            st.components.v1.html(html, height=760, scrolling=True)

        # Caption
        if view_mode == "2D":
            st.caption("Drag nodes • Double-click to fix position • Auto-clustered")
        else:
            st.success("3D Mode Active → Rotate • Zoom • Hold Shift + Drag to pan")

        # Stats
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Nodes", st.session_state.graph.number_of_nodes())
        with col_stat2:
            st.metric("Edges", st.session_state.graph.number_of_edges())
        with col_stat3:
            st.metric("Density", f"{nx.density(st.session_state.graph):.4f}")

        # =============================================
        # EXPORT SECTION — Clean & Beautiful
        # =============================================
        st.divider()
        st.markdown("### Export & Download")

        save_name = st.text_input(
            "Export name",
            value=f"KnowledgeGraph_{datetime.now().strftime('%b%d_%H%M')}",
            key="export_name"
        )

        btn2, btn3 = st.columns(2)

        with btn2:
            if st.button("Interactive HTML", use_container_width=True):
                path = save_pyvis_html(st.session_state.graph, save_name)
                with open(path, "rb") as f:
                    st.download_button(
                        "Download HTML",
                        data=f,
                        file_name=path.name,
                        mime="text/html",
                        use_container_width=True
                    )

        with btn3:
            if st.button("Static PNG", use_container_width=True):
                path = save_graph_png(st.session_state.graph, save_name)
                with open(path, "rb") as f:
                    st.download_button(
                        "Download PNG",
                        data=f,
                        file_name=path.name,
                        mime="image/png",
                        use_container_width=True
                    )
# ---------------------------
# Tab 5: Semantic Search
# Semantic Search Tab
with tab5:
    st.markdown('<p class="sub-header">🔍 Semantic Search & Subgraph Extraction</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    
    if not any([st.session_state.data is not None, st.session_state.triples_data is not None, st.session_state.entities_data is not None]):
        st.warning("⚠️ Please upload data or run the NLP pipeline first")
    else:
        # Search Section
        st.markdown("#### 🔎 Search the Knowledge")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input("Search query (e.g., 'quantum computing', 'India', 'technology')", key="search_query")
        
        with col2:
            top_k = st.slider("Top Results", 1, 10, 5, key="top_k_slider")
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.search_results = None
                st.rerun()
        
        # Perform search
        if search_clicked and query:
            try:
                with st.spinner("Searching..."):
                    ss = st.session_state.get("semantic_engine", None)
                    if ss is None:
                        from modules.semantic_search import SemanticSearch
                        ss = SemanticSearch()
                        
                        df_texts = st.session_state.data if st.session_state.data is not None else None
                        df_triples = st.session_state.triples_data if st.session_state.triples_data is not None else None
                        entities_list = None
                        if st.session_state.entities_data is not None:
                            entities_list = SemanticSearch.extract_entities_from_session(st.session_state.entities_data)
                        
                        ss.build_index(df_texts=df_texts, df_triples=df_triples, entities_list=entities_list)
                        st.session_state.semantic_engine = ss
                    
                    results = ss.search(query, top_k=top_k)
                    st.session_state.has_done_semantic_search = True
                    st.session_state.search_results = results
                    
            except Exception as e:
                st.error(f"❌ Search error: {str(e)}")
        
        # Display search results
        if st.session_state.search_results:
            results = st.session_state.search_results
            
            if not results:
                st.warning("No matches found")
            else:
                st.success(f"✅ Found {len(results)} results")
                st.markdown("---")
                
                # Create tabs for each result with subgraph
                result_tabs = st.tabs([f"Result {i+1}" for i in range(len(results))])
                
                for idx, (tab, res) in enumerate(zip(result_tabs, results)):
                    with tab:
                        # Display result info
                        st.markdown(f"### Result {res['rank']}")
                        st.metric("Similarity Score", f"{res['score']:.3f}")
                        
                        meta = res.get('meta', {})
                        
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.write("**Type:**", meta.get('type', 'unknown'))
                        with col_info2:
                            st.write("**Source:**", meta.get('source', 'unknown'))
                        
                        st.markdown("#### 📄 Content")
                        text_display = res['text']
                        if len(text_display) > 500:
                            text_display = text_display[:500] + "..."
                        st.info(text_display)
                        
                        # Determine entity to search
                        search_entity = None
                        if meta.get('type') == 'entity':
                            search_entity = meta.get('entity')
                        elif meta.get('type') == 'triple':
                            search_entity = meta.get('subject')
                        
                        # Show subgraph section
                        if search_entity:
                            st.markdown("---")
                            st.markdown("#### 🌐 Knowledge Graph Context")
                            
                            # Use checkbox instead of button
                            show_graph = st.checkbox(
                                f"Show subgraph for: **{search_entity}**", 
                                key=f"show_graph_{idx}_{hash(search_entity)}"
                            )
                            
                            if show_graph:
                                if st.session_state.graph is None:
                                    st.error("⚠️ Please build the knowledge graph first (go to Tab 4: Knowledge Graph)")
                                else:
                                    try:
                                        with st.spinner(f"Loading subgraph for {search_entity}..."):
                                            from modules.graph_builder import search_subgraph, visualize_graph
                                            
                                            # Generate subgraph
                                            sub = search_subgraph(st.session_state.graph, search_entity, depth=2)
                                            
                                            if sub.number_of_nodes() > 0:
                                                # Show metrics
                                                col_m1, col_m2, col_m3 = st.columns(3)
                                                with col_m1:
                                                    st.metric("Nodes", sub.number_of_nodes())
                                                with col_m2:
                                                    st.metric("Edges", sub.number_of_edges())
                                                with col_m3:
                                                    st.metric("Search Depth", 2)
                                                
                                                # Show nodes list
                                                with st.expander("📋 View all connected nodes"):
                                                    nodes_list = list(sub.nodes())
                                                    for i, node in enumerate(nodes_list, 1):
                                                        st.write(f"{i}. {node}")
                                                
                                                # Visualize
                                                st.markdown("##### 🎨 Interactive Visualization")
                                                try:
                                                    html = visualize_graph(sub, height="500px")
                                                    st.components.v1.html(html, height=520, scrolling=True)
                                                except Exception as viz_err:
                                                    st.error(f"Visualization failed: {str(viz_err)}")
                                            else:
                                                st.warning(f"No graph connections found for '{search_entity}'")
                                                st.info("This entity may not be connected to other entities in your knowledge graph.")
                                    
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
        
        # Index controls in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 🔧 Semantic Search Controls")
        
        if st.session_state.get("semantic_engine") is not None:
            st.sidebar.success("✅ Index Ready")
            if st.sidebar.button("🔄 Rebuild Index"):
                st.session_state.semantic_engine = None
                st.session_state.search_results = None
                st.sidebar.success("Index cleared!")
                st.rerun()
        else:
            st.sidebar.info("⏳ Index will be built on first search")
# ---------------------------
# Tab 6: Admin
# ---------------------------
with tab6:
    st.markdown('<div class="card"><h3>🛠 Admin Dashboard</h3>', unsafe_allow_html=True)
    tab_stats, tab_logs, tab_edit, tab_feedback = st.tabs(["📊 Stats", "📜 Logs", "🧭 Graph Editor", "💬 Feedback"])

    with tab_stats:
        st.subheader("System Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset Size", len(st.session_state.data) if df_ok(st.session_state.data) else 0)
        c2.metric("Entities Extracted", len(st.session_state.entities_data) if df_ok(st.session_state.entities_data) else 0)
        c3.metric("Relations Extracted", len(st.session_state.triples_data) if df_ok(st.session_state.triples_data) else 0)
        c4.metric("Graph Nodes", st.session_state.graph.number_of_nodes() if st.session_state.graph is not None else 0)

        st.markdown("#### Pipeline Status")
        pipeline = [
            ("Upload Data", df_ok(st.session_state.data)),
            ("Extract Entities", df_ok(st.session_state.entities_data)),
            ("Extract Relations", df_ok(st.session_state.triples_data)),
            ("Build Graph", st.session_state.graph is not None)
        ]
        cols = st.columns(len(pipeline))
        for i, (name, done) in enumerate(pipeline):
            with cols[i]:
                if done:
                    st.success(name)
                else:
                    st.warning(name)

    with tab_logs:
        st.subheader("Logs")
        logs = read_json(LOGS_FILE) or []
        if not logs:
            st.info("No logs yet.")
        else:
            for log in reversed(logs[-40:]):
                lvl = log.get("level", "INFO")
                ts = log.get("timestamp")
                msg = log.get("message")
                if lvl == "ERROR":
                    st.error(f"[{ts}] {msg}")
                elif lvl == "WARNING":
                    st.warning(f"[{ts}] {msg}")
                else:
                    st.info(f"[{ts}] {msg}")

    with tab_edit:
        st.subheader("Graph Refinement")
        if st.session_state.graph is None:
            st.warning("No graph available. Build a graph first.")
        else:
            st.markdown("**Rename Node**")
            old = st.text_input("Existing node label (exact)", key="rename_old")
            new = st.text_input("New label", key="rename_new")
            if st.button("Rename Node"):
                g = st.session_state.graph
                if old in g:
                    nx.relabel_nodes(g, {old: new}, copy=False)
                    st.success("Node renamed")
                    add_log("INFO", f"Renamed node {old} -> {new}")
                else:
                    st.error("Node not found")

            st.markdown("**Merge Nodes (A -> B)**")
            node_a = st.text_input("Node A (will be merged into B)", key="merge_a")
            node_b = st.text_input("Node B (target)", key="merge_b")
            if st.button("Merge Nodes"):
                g = st.session_state.graph
                if node_a in g and node_b in g:
                    for nbr in list(g.neighbors(node_a)):
                        if nbr != node_b:
                            g.add_edge(node_b, nbr)
                    if g.has_node(node_a):
                        g.remove_node(node_a)
                    st.success("Nodes merged")
                    add_log("INFO", f"Merged {node_a} into {node_b}")
                else:
                    st.error("Nodes not found")

            st.markdown("**Delete Edge**")
            du = st.text_input("Subject", key="del_u")
            dv = st.text_input("Object", key="del_v")
            if st.button("Delete Edge"):
                g = st.session_state.graph
                if g.has_edge(du, dv):
                    g.remove_edge(du, dv)
                    st.success("Edge removed")
                    add_log("INFO", f"Deleted edge {du} -> {dv}")
                else:
                    st.error("Edge not found")

    with tab_feedback:
        st.subheader("User Feedback")
        with st.form("feedback_form"):
            ftype = st.selectbox("Type", ["Bug Report", "Feature Request", "General Feedback"])
            ftext = st.text_area("Your Feedback", height=150)
            rating = st.slider("Rate your experience", 1, 5, 4)
            submit_fb = st.form_submit_button("Submit")

        if submit_fb:
            if ftext:
                cache.add_feedback({"type": ftype, "text": ftext, "rating": rating, "timestamp": datetime.now(timezone.utc).isoformat()})
                st.success("Thank you for your feedback")
            else:
                st.error("Please enter feedback text")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("<div class='footer-note'>🧠 AI-KnowMap — Built with Streamlit • Internship Project 2025</div>", unsafe_allow_html=True)
