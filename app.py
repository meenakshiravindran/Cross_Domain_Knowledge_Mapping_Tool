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
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

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

def load_graph_from_file(name: str):
    graphs = read_json(GRAPHS_FILE) or {}
    if name not in graphs:
        return None
    return nx.readwrite.json_graph.node_link_graph(graphs[name])

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
  padding: 16px;
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
    st.markdown('<div class="main-header">🧠 AI-KnowMap — Interdisciplinary Knowledge Mapper</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        tabs = st.tabs(["🔐 Sign In", "📝 Register"])

        # SIGN IN
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

        # REGISTER
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

        st.markdown('</div>', unsafe_allow_html=True)

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

    st.markdown("---")
    st.markdown("#### Saved Graphs")

    saved_graphs = list((read_json(GRAPHS_FILE) or {}).keys())
    sel = st.selectbox("Open saved graph", ["-- none --"] + saved_graphs)
    if sel and sel != "-- none --":
        g = load_graph_from_file(sel)
        if g is not None:
            st.session_state.graph = g
            st.success(f"Loaded graph: {sel}")

    st.markdown("---")
    st.markdown("Developer Controls")
    if st.button("Rebuild Semantic Index"):
        st.session_state.semantic_engine = None
        st.success("Index cleared; will rebuild on next search")

# ---------------------------
# Main tabs (A+B mix styling)
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Home", "📂 Dataset", "🧠 NLP Pipeline",
    "🌐 Knowledge Graph", "🔍 Semantic Search",
    "🛠 Admin"
])

# ---------------------------
# Tab 1: Home (mix of screenshot + dashboard cards)
# ---------------------------
with tab1:
    st.markdown('<div class="main-header">AI-KnowMap — Interdisciplinary Knowledge Mapping</div>', unsafe_allow_html=True)
    left, right = st.columns([3,1])

    # Left: big centered preview (Option A)
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Project Preview")
        # Use the uploaded screenshot path (one of the screenshots you uploaded)
        preview_paths = [
            ROOT / "Screenshot 2025-11-24 120049.png",
        ]
        shown = False
        for p in preview_paths:
            if p.exists():
                st.image(str(p), caption="UI Preview", use_container_width=True)
                shown = True
                break
        if not shown:
            st.info("Preview image not found. Put your screenshot in the project root or assets folder.")

        st.markdown("### Quick Actions")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Load Demo Dataset", key="home_demo_btn"):
                demo = pd.DataFrame({
                    "title": ["Climate Change", "AI Healthcare", "Quantum Breakthrough"],
                    "body": ["Climate text", "AI text", "Quantum text"]
                })
                st.session_state.data = demo
                st.success("Demo dataset loaded")
        with c2:
            if st.button("Run Entity Extraction (demo)"):
                if df_ok(st.session_state.data):
                    ents = extract_entities_from_data(st.session_state.data)
                    st.session_state.entities_data = ents
                    save_entities(ents)
                    st.success("Entities extracted (demo)")
                else:
                    st.warning("No dataset loaded")
        with c3:
            if st.button("Build Graph (demo)"):
                if df_ok(st.session_state.triples_data):
                    g = build_knowledge_graph(st.session_state.triples_data, max_nodes=100)
                    st.session_state.graph = g
                    st.success("Graph built (demo)")
                else:
                    st.warning("No triples available")

        st.markdown('</div>', unsafe_allow_html=True)

    # Right: feature cards and stats (Option B)
    with right:
        st.markdown('<div class="small-card">', unsafe_allow_html=True)
        st.markdown("#### Highlights")
        st.write("• JWT-based secure login")
        st.write("• Multi-source dataset support")
        st.write("• NER & Relation Extraction")
        st.write("• Interactive KG + Semantic Search")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="small-card">', unsafe_allow_html=True)
        st.markdown("#### Stats")
        st.metric("Graphs saved", len(read_json(GRAPHS_FILE) or {}))
        st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown('<p class="sub-header">🧠 NLP Processing Pipeline</p>', unsafe_allow_html=True)
    
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
                with st.spinner("Extracting relations... This may take a while."):
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

# ---------------------------
# Tab 4: Knowledge Graph
# ---------------------------
with tab4:
    st.markdown('<div class="card"><h3>🌐 Knowledge Graph</h3>', unsafe_allow_html=True)
    if not df_ok(st.session_state.triples_data):
        st.warning("Run relation extraction in the NLP Pipeline tab first.")
    else:
        colL, colR = st.columns([3,1])
        with colR:
            max_nodes = st.slider("Max Nodes", 10, 500, 100)
            if st.button("Build Knowledge Graph"):
                with st.spinner("Building graph..."):
                    g = build_knowledge_graph(st.session_state.triples_data, max_nodes=max_nodes)
                    st.session_state.graph = g
                    add_log("INFO", f"Graph built with {g.number_of_nodes()} nodes")
                    st.success("Graph built successfully")
            if st.session_state.graph is not None:
                stats = get_graph_stats(st.session_state.graph)
                st.metric("Nodes", stats['nodes'])
                st.metric("Edges", stats['edges'])
                st.metric("Density", f"{stats['density']:.4f}")

        with colL:
            if st.session_state.graph is not None:
                st.subheader("Interactive Graph")
                try:
                    html_graph = visualize_graph(st.session_state.graph, height="650px")
                    st.components.v1.html(html_graph, height=670, scrolling=True)
                except Exception:
                    # fallback to pyvis render to file
                    net = Network(height="650px", width="100%", bgcolor="#ffffff")
                    for n, d in st.session_state.graph.nodes(data=True):
                        net.add_node(n, label=str(n))
                    for u, v, d in st.session_state.graph.edges(data=True):
                        net.add_edge(u, v)
                    tmpfile = DATA_DIR / "temp_graph.html"
                    net.save_graph(str(tmpfile))
                    st.components.v1.html(tmpfile.read_text(), height=670, scrolling=True)
    st.markdown('</div>', unsafe_allow_html=True)

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