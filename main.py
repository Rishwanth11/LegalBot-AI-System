# main.py — Professional UI polish + color theme
# Functional behavior unchanged: no IPC, carry-over permanently enabled,
# New Chat & History in sidebar, BNS/BNSS/BSA quick actions, bookmarks, 5-star rating.

import streamlit as st
import json
import os
import glob
import google.generativeai as genai
import re
import numpy as np
import time
import datetime
from typing import List

from nlp_processor import preprocess_text, load_dataset, train_classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Basic Streamlit config
# -----------------------
st.set_page_config(page_title="LegalBot", layout="wide", initial_sidebar_state="expanded")

# -----------------------
# Embedded Gemini API Key (your choice)
# -----------------------
# NOTE: This is insecure for production. Keep it only for testing as you requested.
GEMINI_API_KEY = "AIzaSyDodGuUbdV5khgHzL94Yg21_ru8ZLBuz3w"

# -----------------------
# Compatibility helpers
# -----------------------
def safe_rerun():
    if hasattr(st, "rerun"):
        return st.rerun()
    if hasattr(st, "experimental_rerun"):
        return st.experimental_rerun()
    raise RuntimeError("This Streamlit version does not support rerun. Upgrade Streamlit.")

def safe_modal(title: str):
    if hasattr(st, "modal"):
        return st.modal(title)
    else:
        class _ExpanderCtx:
            def _init_(self, title):
                self.title = title
                self.container = None
            def _enter_(self):
                self.container = st.expander(self.title, expanded=True)
                return self.container
            def _exit_(self, exc_type, exc, tb):
                return False
        return _ExpanderCtx(title)

# -----------------------
# Paths & auth constants
# -----------------------
AUTH_STATUS_KEY = "authenticated"
DATA_FOLDER = "data"
USER_DB_FILE = os.path.join(DATA_FOLDER, "users.json")
FEEDBACK_FILE = os.path.join(DATA_FOLDER, "feedback.json")

# -----------------------
# Configure Gemini (genai)
# -----------------------
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"AI Key Error: {e}")
    st.stop()

# -----------------------
# Category mapping
# -----------------------
CATEGORY_TO_FILE = {
    "criminal": "bns.json",
    "procedural": "bnss.json",
    "evidence": "bsa.json",
}

# -----------------------
# Optional PDF support (fpdf)
# -----------------------
_have_fpdf = False
try:
    from fpdf import FPDF
    _have_fpdf = True
except Exception:
    _have_fpdf = False

# -----------------------
# User DB utilities
# -----------------------
def load_user_database():
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_user_database(users):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    tmp = USER_DB_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, USER_DB_FILE)

def register_user(username, password):
    u = username.strip().lower()
    p = password.strip()
    if not u or not p:
        return False, "Username & password required."
    users = load_user_database()
    if u in users:
        return False, "Username already exists."
    users[u] = p
    save_user_database(users)
    return True, "Registration successful!"

def verify_login(username, password):
    u = username.strip().lower()
    p = password.strip()
    users = load_user_database()
    return users.get(u) == p

# -----------------------
# Feedback & chat utilities
# -----------------------
def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_feedback(data):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    tmp = FEEDBACK_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, FEEDBACK_FILE)

def append_feedback(entry):
    fb = load_feedback()
    fb.append(entry)
    save_feedback(fb)

def log_feedback(index, ftype, extra=None):
    entry = {
        "message_index": index,
        "type": ftype,
        "timestamp": time.time(),
        "content": st.session_state.messages[index].get("content", ""),
        "user": st.session_state.get("user", "unknown")
    }
    if extra:
        entry.update(extra)
    if ftype in ["useful", "useless"]:
        st.session_state.messages[index]["feedback"] = ftype
    if ftype == "rating":
        st.session_state.messages[index]["rating"] = extra.get("rating") if extra else st.session_state.messages[index].get("rating")
    append_feedback(entry)
    safe_rerun()

# -----------------------
# Bookmarks handling
# -----------------------
def init_bookmarks():
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = []

def add_bookmark(title: str, content: str, source_sections: List[dict]):
    init_bookmarks()
    bm = {
        "id": int(time.time() * 1000),
        "title": title,
        "content": content,
        "ts": time.time(),
        "sources": source_sections
    }
    st.session_state.bookmarks.insert(0, bm)
    save_path = os.path.join(DATA_FOLDER, "bookmarks.json")
    os.makedirs(DATA_FOLDER, exist_ok=True)
    try:
        existing = []
        if os.path.exists(save_path):
            existing = json.load(open(save_path, "r", encoding="utf-8"))
        existing.insert(0, bm)
        tmp = save_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, save_path)
    except Exception:
        pass

def remove_bookmark(bmid):
    init_bookmarks()
    st.session_state.bookmarks = [b for b in st.session_state.bookmarks if b["id"] != bmid]
    save_path = os.path.join(DATA_FOLDER, "bookmarks.json")
    if os.path.exists(save_path):
        try:
            existing = json.load(open(save_path, "r", encoding="utf-8"))
            existing = [b for b in existing if b["id"] != bmid]
            tmp = save_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=4)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(tmp, save_path)
        except Exception:
            pass

# -----------------------
# Chat utilities
# -----------------------
def new_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am LegalBot. How can I help?", "ts": time.time()}
    ]
    st.session_state.last_successful_query = ""
    safe_rerun()

def logout_user():
    for k in ['user', AUTH_STATUS_KEY, 'pending_prompt', 'pending_timestamp', 'show_user_menu', 'show_confirm_logout', 'enable_carry']:
        if k in st.session_state:
            del st.session_state[k]
    for k in ['messages', 'history_log', 'last_successful_query']:
        if k in st.session_state:
            del st.session_state[k]
    safe_rerun()

# -----------------------
# Load legal data & classifier
# -----------------------
@st.cache_resource
def load_all_models_and_data():
    all_data = []
    json_files = glob.glob(os.path.join(DATA_FOLDER, "*.json"))
    if not json_files:
        st.error(f"No JSON data files found in '{DATA_FOLDER}'.")
        return None, None, None, None, None, None

    section_map = {}
    categorized_data = {"criminal": [], "procedural": [], "evidence": []}

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                file_name = os.path.basename(file_path)
                for section in data:
                    section['source_document'] = file_name
                all_data.extend(data)
                for cat, f_name in CATEGORY_TO_FILE.items():
                    if f_name == file_name:
                        categorized_data[cat].extend(data)
                for section in data:
                    key = (file_name, section.get('section_number'))
                    section_map[key] = section
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    dataset_path = os.path.join(DATA_FOLDER, "query_dataset.json")
    dataset = load_dataset(dataset_path) if os.path.exists(dataset_path) else None
    classifier = train_classifier(dataset) if dataset else None

    corpus = [s['text'] for s in all_data if 'text' in s and s['text']]
    if not corpus:
        st.error("Legal corpus empty or malformed.")
        return categorized_data, section_map, classifier, None, None, all_data

    corpus_preprocessed = [preprocess_text(t) for t in corpus]
    tfidf_vectorizer = TfidfVectorizer()
    corpus_vectors = tfidf_vectorizer.fit_transform(corpus_preprocessed)

    return categorized_data, section_map, classifier, tfidf_vectorizer, corpus_vectors, all_data

categorized_legal_data, legal_section_map, query_classifier, tfidf_vectorizer, corpus_vectors, all_legal_data = load_all_models_and_data()

# -----------------------
# RAG helpers
# -----------------------
def check_for_correlated_sections(ranked_sections, section_map):
    definition_key = ("bns.json", "101")
    punishment_key = ("bns.json", "103")

    is_definition_retrieved = any(
        s.get('section_number') == '101' and s.get('source_document') == 'bns.json'
        for s in ranked_sections
    )
    is_punishment_retrieved = any(
        s.get('section_number') == '103' and s.get('source_document') == 'bns.json'
        for s in ranked_sections
    )

    if is_definition_retrieved and not is_punishment_retrieved and punishment_key in section_map:
        ranked_sections.append(section_map[punishment_key])
    if is_punishment_retrieved and not is_definition_retrieved and definition_key in section_map:
        ranked_sections.append(section_map[definition_key])

    return ranked_sections

# -----------------------
# find_relevant_sections
# -----------------------
def find_relevant_sections(query, processed_query, category, categorized_data, section_map, vectorizer, corpus_vectors, all_legal_data):
    query_lower = query.lower().strip()
    found_sections = []

    chat_words = ["hi", "hello", "hey", "how are you", "what your name", "what is your name",
                  "love you", "ok", "thanks", "thank you", "no thanks", "no", "can say name",
                  "what are you doing", "what mean", "bye", "goodbye"]
    if processed_query in chat_words or processed_query.replace(" ", "") in chat_words:
        return [{"chat": True}]

    m_def = re.search(r'\bwhat (is|mean|does) (bns|bnss|bsa)\b', query_lower)
    if m_def:
        doc_short = m_def.group(2)
        doc_map_short = {"bns": "bns.json", "bnss": "bnss.json", "bsa": "bsa.json"}
        file_name = doc_map_short.get(doc_short)
        if file_name:
            has_any = any(s.get('source_document') == file_name for s in all_legal_data)
            if has_any:
                return [{"doc_info": file_name}]
            else:
                return []

    if query_lower.startswith("show common"):
        if query_lower == "show common bns sections":
            st.info("Query classified as: *Quick Action (BNS)*")
            criminal_list = categorized_data.get("criminal", [])
            return criminal_list[:min(len(criminal_list), 10)]
        if query_lower == "show common bnss sections":
            st.info("Query classified as: *Quick Action (BNSS)*")
            procedural_list = categorized_data.get("procedural", [])
            return procedural_list[:min(len(procedural_list), 10)]
        if query_lower == "show common bsa sections":
            st.info("Query classified as: *Quick Action (BSA)*")
            evidence_list = categorized_data.get("evidence", [])
            return evidence_list[:min(len(evidence_list), 10)]
        return []

    section_match = re.search(r'(bns|bnss|bsa)?\s*(?:section\s*)?(\d+)', query_lower)
    doc_map = {"bns": "bns.json", "bnss": "bnss.json", "bsa": "bsa.json"}

    if section_match:
        doc_key = section_match.group(1)
        section_num = section_match.group(2)

        if doc_key:
            key = (doc_map[doc_key], section_num)
            if key in section_map:
                found_sections.append(section_map[key])
        else:
            for fname in set(doc_map.values()):
                key = (fname, section_num)
                if key in section_map:
                    found_sections.append(section_map[key])

    if found_sections:
        st.info("Query classified as: *Section Number Lookup*")
        return found_sections

    if category in categorized_data:
        data_to_search = categorized_data[category]
    else:
        data_to_search = all_legal_data

    valid_sections_to_search = [s for s in data_to_search if 'text' in s and s['text']]
    if not valid_sections_to_search or not vectorizer or corpus_vectors is None:
        return []

    data_indices = [all_legal_data.index(section) for section in valid_sections_to_search if section in all_legal_data]

    try:
        empty_corpus = (corpus_vectors.size == 0)
    except Exception:
        empty_corpus = False

    if not data_indices or not vectorizer or empty_corpus:
        return []

    query_vector = vectorizer.transform([preprocess_text(query)])
    corpus_subset_vectors = corpus_vectors[data_indices]
    similarity_scores = cosine_similarity(query_vector, corpus_subset_vectors)[0]
    top_n_indices_in_subset = np.argsort(similarity_scores)[::-1][:5]

    ranked_sections = []
    for rank_index in top_n_indices_in_subset:
        full_corpus_index = data_indices[rank_index]
        section = all_legal_data[full_corpus_index]
        score = similarity_scores[rank_index]
        if score > 0.1:
            ranked_sections.append(section)

    ranked_sections = check_for_correlated_sections(ranked_sections, section_map)
    return ranked_sections

# -----------------------
# generate_answer_with_llm
# -----------------------
def generate_answer_with_llm(query, relevant_sections):
    def deterministic_from_sections(q, sections):
        if not sections:
            return "I couldn't find any matching legal sections in the corpus for that query."

        if q.lower().startswith("show common"):
            lines = []
            for sec in sections:
                sec_no = sec.get("section_number", "(no number)")
                title = sec.get("title") or (sec.get("text","")[:120].strip() + ("..." if len(sec.get("text",""))>120 else ""))
                lines.append(f"- Section {sec_no}: {title}")
            return "Here are the common sections:\n\n" + "\n".join(lines)

        out = []
        out.append("Answer (based only on matched sections):\n")
        for s in sections:
            src = s.get("source_document", "UNKNOWN")
            sec_no = s.get("section_number", "UNKNOWN")
            title = s.get("title", "")
            text = s.get("text", "")
            preview = title if title else (text[:300] + ("..." if len(text) > 300 else ""))
            out.append(f"- {src} Section {sec_no}: {preview}\n")
        out.append("\nIf you need a natural-language summary, check server logs or ensure Gemini credentials are valid.")
        return "\n".join(out)

    if relevant_sections and isinstance(relevant_sections, list):
        first = relevant_sections[0]
        if isinstance(first, dict) and first.get("chat"):
            low = query.lower().strip()
            if any(w in low for w in ["hi", "hello", "hey"]):
                return "Hello! How can I assist you today?"
            if "thank" in low:
                return "You're welcome! Anything else I can help with?"
            if any(w in low for w in ["bye", "goodbye"]):
                return "Goodbye — feel free to come back with more questions!"
            return "Hello — how can I help?"

        if isinstance(first, dict) and first.get("doc_info"):
            file_name = first.get("doc_info")
            pretty = file_name.replace('.json','').upper()
            return (
                f"{pretty} refers to the legal document stored in the app as *{file_name}*. "
                "The corpus contains sections from that source. "
                f"Try: show common {pretty.lower()} sections or request a section number."
            )

    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')

        if relevant_sections:
            context = (
                "You are LegalBot, a helpful AI legal assistant. Your task is to answer the user's question.\n"
                "Base your answer only on the relevant legal sections provided below.\n"
                f"*User's Question:* {query}\n\n"
                "--- Relevant Legal Sections ---\n\n"
            )
            for i, section in enumerate(relevant_sections):
                src = section.get('source_document', '')
                src_label = src.replace('.json','').upper() if src else src
                title_preview = section.get('title','')
                context += f"*Source {i+1} ({src_label} Section {section.get('section_number')} : {title_preview}):*\n"
                context += f"{section.get('text','')}\n\n"
            context += "--- End of Sections ---\n\n"
            context += "Please provide a clear and direct answer to the user's question based only on these sections."
            try:
                response = model.generate_content(context)
                text = getattr(response, "text", None) or str(response)
                if not text or not text.strip():
                    raise RuntimeError("Empty response from LLM")
                return text
            except Exception:
                response = model.generate(context)
                text = getattr(response, "text", None) or str(response)
                if not text or not text.strip():
                    raise RuntimeError("Empty response from LLM (generate fallback)")
                return text
        else:
            try:
                response = model.generate_content(query)
                text = getattr(response, "text", None) or str(response)
                if not text or not text.strip():
                    raise RuntimeError("Empty response from LLM")
                return text
            except Exception:
                response = model.generate(query)
                text = getattr(response, "text", None) or str(response)
                if not text or not text.strip():
                    raise RuntimeError("Empty response from LLM (generate fallback)")
                return text

    except Exception as e:
        print(f"[generate_answer_with_llm] Gemini/LLM error: {e}")
        try:
            st.error("AI service error — returning deterministic fallback.")
        except Exception:
            pass
        return deterministic_from_sections(query, relevant_sections)

# -----------------------
# Styling: professional color palette and polished UI
# -----------------------
PRIMARY = "#0b5fff"        # strong blue
ACCENT = "#00b894"         # green accent
MUTED = "#6b7280"          # gray
CARD_BG = "#ffffff"        # light card background for light theme
CARD_BG_DARK = "#0f1724"
APP_BG = "#f6f9fc"
SIDEBAR_BG = "#ffffff"
HEADER_BG = "#ffffff"

LIGHT_CSS = f"""
:root {{
  --primary: {PRIMARY};
  --accent: {ACCENT};
  --muted: {MUTED};
  --app-bg: {APP_BG};
  --card-bg: {CARD_BG};
  --sidebar-bg: {SIDEBAR_BG};
  --header-bg: {HEADER_BG};
  --text-color: #0f1724;
}}

body {{
  background-color: var(--app-bg);
  color: var(--text-color);
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}}

.stApp .css-18e3th9 {{ /* main container */
  background-color: var(--app-bg) !important;
}}

header .block-container {{
  background: linear-gradient(90deg, rgba(11,95,255,0.06), rgba(0,184,148,0.04));
}}

[data-testid="stSidebar"] {{
  background-color: var(--sidebar-bg) !important;
  padding-top: 18px;
  border-right: 1px solid rgba(15,23,36,0.04);
}}

.stButton>button, .stDownloadButton>button {{
  background-color: var(--primary) !important;
  color: white !important;
  border: none !important;
  padding: 8px 12px !important;
  border-radius: 10px !important;
  box-shadow: 0 6px 14px rgba(11,95,255,0.12) !important;
}}

.stButton>button[disabled], .stDownloadButton>button[disabled] {{
  opacity: 0.6;
}}

.css-1d391kg {{ /* chat input container mimic - best-effort */
  border-radius: 10px;
}}

.message-card {{
  border-radius: 12px;
  padding: 12px;
  background: var(--card-bg);
  box-shadow: 0 6px 18px rgba(6,15,40,0.06);
  margin-bottom: 12px;
  color: var(--text-color);
}}

.message-assistant {{
  background: linear-gradient(180deg, rgba(11,95,255,0.05), rgba(11,95,255,0.02));
  border-left: 4px solid var(--primary);
}}

.message-user {{
  background: linear-gradient(180deg, rgba(0,184,148,0.04), rgba(0,184,148,0.02));
  border-left: 4px solid var(--accent);
}}

.header-bar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  background: var(--header-bg);
  border-bottom: 1px solid rgba(15,23,36,0.04);
  box-shadow: 0 2px 6px rgba(15,23,36,0.02);
  border-radius: 0 0 8px 8px;
  margin-bottom: 12px;
}}

.header-title {{
  font-size:18px;
  font-weight:700;
  color: var(--text-color);
}}

.top-controls {{
  display:flex;
  gap:8px;
}}

.small-muted {{
  color: var(--muted);
  font-size:12px;
}}

.quick-btn {{
  background-color: transparent !important;
  border: 1px solid rgba(11,95,255,0.12) !important;
  color: var(--primary) !important;
  padding: 8px 10px !important;
  border-radius: 8px !important;
}}

.rating-star {{
  font-size:16px;
  padding:6px 9px;
  border-radius:8px;
  border: 1px solid rgba(0,0,0,0.06);
  background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(250,250,250,0.95));
}}
"""

DARK_CSS = f"""
:root {{
  --primary: {PRIMARY};
  --accent: {ACCENT};
  --muted: #9aa4b2;
  --app-bg: #071121;
  --card-bg: {CARD_BG_DARK};
  --sidebar-bg: #071528;
  --header-bg: #071122;
  --text-color: #e6eef8;
}}

body {{
  background-color: var(--app-bg);
  color: var(--text-color);
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}}

.stApp .css-18e3th9 {{
  background-color: var(--app-bg) !important;
}}

[data-testid="stSidebar"] {{
  background-color: var(--sidebar-bg) !important;
  padding-top: 18px;
  border-right: 1px solid rgba(255,255,255,0.03);
}}

.stButton>button, .stDownloadButton>button {{
  background-color: var(--primary) !important;
  color: white !important;
  border: none !important;
  padding: 8px 12px !important;
  border-radius: 10px !important;
  box-shadow: 0 6px 14px rgba(11,95,255,0.06) !important;
}}

.message-card {{
  border-radius: 12px;
  padding: 12px;
  background: var(--card-bg);
  box-shadow: 0 6px 18px rgba(0,0,0,0.4);
  margin-bottom: 12px;
  color: var(--text-color);
}}

.message-assistant {{
  background: linear-gradient(180deg, rgba(11,95,255,0.06), rgba(11,95,255,0.02));
  border-left: 4px solid var(--primary);
}}

.message-user {{
  background: linear-gradient(180deg, rgba(0,184,148,0.04), rgba(0,184,148,0.02));
  border-left: 4px solid var(--accent);
}}

.header-bar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  background: var(--header-bg);
  border-bottom: 1px solid rgba(255,255,255,0.02);
  box-shadow: 0 2px 6px rgba(0,0,0,0.25);
  border-radius: 0 0 8px 8px;
  margin-bottom: 12px;
}}

.header-title {{
  font-size:18px;
  font-weight:700;
  color: var(--text-color);
}}

.small-muted {{
  color: var(--muted);
  font-size:12px;
}}

.rating-star {{
  font-size:16px;
  padding:6px 9px;
  border-radius:8px;
  border: 1px solid rgba(255,255,255,0.03);
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
}}
"""

def apply_theme_css():
    if st.session_state.get('theme_dark', False):
        st.markdown(f"<style>{DARK_CSS}</style>", unsafe_allow_html=True)
    else:
        st.markdown(f"<style>{LIGHT_CSS}</style>", unsafe_allow_html=True)

# -----------------------
# Header / top area
# -----------------------
def render_header_with_user_menu():
    cols = st.columns([8, 1])
    left = cols[0]
    right = cols[1]
    with left:
        st.markdown(
            "<div class='header-bar'><div class='header-title'>⚖ LegalBot — AI-Powered Judiciary Reference System</div>"
            "<div class='small-muted'>Search BNS, BNSS, BSA sections • Professional interface</div></div>",
            unsafe_allow_html=True
        )
    with right:
        username_display = st.session_state.get("user", None)
        if username_display:
            if st.button(f"👤 {username_display}", key="top_user_button"):
                st.session_state.show_user_menu = True
            if st.session_state.get("show_user_menu", False):
                with safe_modal("User Profile"):
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.markdown(f"<div style='font-size:48px'>🧑‍⚖</div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"{username_display}")
                        st.markdown("Logged in")
                        st.markdown("---")
                        if st.button("Log out", key="profile_logout_btn"):
                            st.session_state.show_confirm_logout = True
                        if st.button("Close", key="profile_close_btn"):
                            st.session_state.show_user_menu = False
                            safe_rerun()
        else:
            st.write("")

# -----------------------
# Init state
# -----------------------
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am LegalBot. I can answer questions about the BNS (Criminal), BNSS (Procedural), and BSA (Evidence). How can I help?", "ts": time.time()}
    ]
if 'history_log' not in st.session_state:
    st.session_state.history_log = []
if 'last_successful_query' not in st.session_state:
    st.session_state.last_successful_query = ""
if 'theme_dark' not in st.session_state:
    st.session_state.theme_dark = False
if 'show_user_menu' not in st.session_state:
    st.session_state.show_user_menu = False
if 'show_confirm_logout' not in st.session_state:
    st.session_state.show_confirm_logout = False
if 'show_history_modal' not in st.session_state:
    st.session_state.show_history_modal = False
if 'pending_prompt' not in st.session_state:
    st.session_state.pending_prompt = None
if 'pending_timestamp' not in st.session_state:
    st.session_state.pending_timestamp = None
# carry-over permanently enabled
if 'enable_carry' not in st.session_state:
    st.session_state.enable_carry = True
st.session_state.use_llm = True

init_bookmarks()

# -----------------------
# Render header + apply theme
# -----------------------
render_header_with_user_menu()
apply_theme_css()

# -----------------------
# Sidebar: polished layout, New Chat + History moved here
# -----------------------
with st.sidebar:
    st.markdown("### Account & Settings")
    theme_toggle = st.checkbox("Dark theme", value=st.session_state.theme_dark, key="theme_toggle")
    if theme_toggle != st.session_state.theme_dark:
        st.session_state.theme_dark = theme_toggle
        apply_theme_css()
    username_display = st.session_state.get("user", None)
    if username_display:
        with st.expander(f"👤 {username_display}", expanded=False):
            st.markdown(f"*Logged in as:* {username_display}")
            if st.button("Log out", key="sidebar_logout"):
                st.session_state.show_confirm_logout = True

    st.markdown("---")

    # New Chat + History (polished)
    if st.button("🔁 New Chat", key="sidebar_new_chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am LegalBot. How can I help?", "ts": time.time()}
        ]
        st.session_state.last_successful_query = ""
        st.session_state.show_user_menu = False
        st.session_state.show_confirm_logout = False
        st.session_state.show_history_modal = False
        safe_rerun()

    if st.button("📚 History", key="sidebar_history", use_container_width=True):
        st.session_state.show_history_modal = True

    st.markdown("---")

    with st.expander("🔖 Bookmarks", expanded=False):
        init_bookmarks()
        if not st.session_state.bookmarks:
            st.info("No bookmarks yet. Click the ⭐ next to any answer to save it.")
        else:
            for b in st.session_state.bookmarks:
                st.markdown(f"{b['title']}**  \n_added {datetime.datetime.fromtimestamp(b['ts']).strftime('%Y-%m-%d %H:%M:%S')}_")
                st.caption(b['content'][:220] + ("..." if len(b['content'])>220 else ""))
                cols = st.columns([1,1,1])
                if cols[0].button("Download TXT", key=f"bm_txt_{b['id']}"):
                    txt = f"{b['title']}\n\n{b['content']}\n\nSources:\n"
                    for s in b.get("sources", []):
                        src_doc = s.get("source_document", "")
                        sec_no = s.get("section_number", "")
                        title = s.get("title", "")
                        txt += f"- {src_doc} Section {sec_no}: {title}\n"
                    st.download_button("Click to download", txt, file_name=f"bookmark_{b['id']}.txt")
                if cols[1].button("Download JSON", key=f"bm_json_{b['id']}"):
                    st.download_button("Click to download", json.dumps(b, indent=2), file_name=f"bookmark_{b['id']}.json")
                if cols[2].button("Remove", key=f"bm_rm_{b['id']}"):
                    remove_bookmark(b['id'])
                    safe_rerun()
        if st.session_state.bookmarks:
            if st.button("Download all bookmarks (.json)"):
                st.download_button("Download bookmarks", json.dumps(st.session_state.bookmarks, indent=2), file_name="bookmarks_all.json")

    st.markdown("---")
    st.markdown("### Quick Actions")
    if st.button("BNS Sections (Criminal)", use_container_width=True):
        st.session_state.pending_prompt = "show common bns sections"
        st.session_state.pending_timestamp = time.time()
        safe_rerun()

    if st.button("BNSS Sections (Procedural)", use_container_width=True):
        st.session_state.pending_prompt = "show common bnss sections"
        st.session_state.pending_timestamp = time.time()
        safe_rerun()

    if st.button("BSA Sections (Evidence)", use_container_width=True):
        st.session_state.pending_prompt = "show common bsa sections"
        st.session_state.pending_timestamp = time.time()
        safe_rerun()

    st.markdown("---")
    st.markdown("### Export Chat")
    chat_text = "\n\n".join([f"{m['role'].upper()} ({datetime.datetime.fromtimestamp(m.get('ts', time.time())).strftime('%Y-%m-%d %H:%M:%S')}):\n{m['content']}" for m in st.session_state.messages])
    chat_json = json.dumps(st.session_state.messages, indent=2)

    st.download_button("Download chat (.txt)", chat_text, file_name="legalbot_chat.txt")
    st.download_button("Download chat (.json)", chat_json, file_name="legalbot_chat.json")

    if _have_fpdf:
        if st.button("Download chat (.pdf)"):
            try:
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for m in st.session_state.messages:
                    role = m.get("role", "").upper()
                    ts = datetime.datetime.fromtimestamp(m.get("ts", time.time())).strftime('%Y-%m-%d %H:%M:%S')
                    content = m.get("content", "")
                    pdf.multi_cell(0, 8, f"{role} ({ts}):")
                    pdf.multi_cell(0, 8, content)
                    pdf.ln(2)
                out_path = os.path.join("/tmp" if os.name != "nt" else ".", f"legalbot_chat_{int(time.time())}.pdf")
                pdf.output(out_path)
                with open(out_path, "rb") as f:
                    st.download_button("Click to download PDF", f.read(), file_name=os.path.basename(out_path))
            except Exception as e:
                st.error(f"PDF export failed: {e}")
    else:
        st.info("PDF export unavailable (install fpdf to enable).")

# History modal (keeps same behavior)
if st.session_state.get("show_history_modal", False):
    with safe_modal("Conversation History"):
        if not st.session_state.get("history_log"):
            st.info("No past conversations logged.")
        else:
            for idx in range(len(st.session_state.history_log)-1, -1, -1):
                history = st.session_state.history_log[idx]
                title = next((m['content'] for m in history if m['role'] == 'user'), "New Chat")
                with st.expander(f"Chat {idx+1}: {title[:60]}"):
                    for msg in history:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        ts = datetime.datetime.fromtimestamp(msg.get("ts", time.time())).strftime("%Y-%m-%d %H:%M:%S")
                        st.markdown(f"{role.title()}** ({ts}):")
                        st.write(content)
                    st.write("---")
                    if st.button("Load this chat", key=f"load_hist_modal_{idx}"):
                        st.session_state.messages = list(history)
                        st.session_state.show_history_modal = False
                        safe_rerun()
        if st.button("Close", key="close_hist_modal"):
            st.session_state.show_history_modal = False
            safe_rerun()

# Confirm logout modal
if st.session_state.get("show_confirm_logout", False) and not st.session_state.get("show_user_menu", False):
    with safe_modal("Confirm Logout"):
        st.markdown("Are you sure you want to log out?")
        a, b = st.columns([1,1])
        if a.button("Yes, log out"):
            st.session_state.show_confirm_logout = False
            logout_user()
        if b.button("Cancel"):
            st.session_state.show_confirm_logout = False
            safe_rerun()

# -----------------------
# Helper: carry-over heuristic
# -----------------------
def should_carry_over(prompt_text: str) -> bool:
    if not st.session_state.get('last_successful_query'):
        return False
    if not st.session_state.get('enable_carry', True):
        return False

    p = prompt_text.strip()
    if not p:
        return False

    words = p.split()
    if len(words) > 3:
        return False

    low = p.lower()

    if low.startswith("show common") or re.match(r'^(bns|bnss|bsa)\b', low) or re.match(r'^(section\s*\d+)$', low) or re.match(r'^\d+$', low):
        return False

    if re.fullmatch(r'\d+|section\s*\d+', low.strip()):
        return False

    continuation_starters = {'also', 'and', 'plus', 'more', 'again', 'further', 'instead', 'then'}
    first = words[0].lower().strip(" ,.?")
    if first in continuation_starters:
        return True

    question_words = {'why', 'how', 'when', 'where', 'who', 'what'}
    if first in question_words or low.endswith('?'):
        return False

    return False

# -----------------------
# Main UI when authenticated
# -----------------------
if st.session_state.get(AUTH_STATUS_KEY, False):

    # render chat messages (polished cards)
    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        feedback_status = message.get('feedback', None)
        ts_str = datetime.datetime.fromtimestamp(message.get('ts', time.time())).strftime("%Y-%m-%d %H:%M:%S")

        avatar = "🤖" if role == "assistant" else "🧑"
        role_label = "Assistant" if role == "assistant" else "You"

        card_class = "message-card message-assistant" if role == "assistant" else "message-card message-user"

        with st.container():
            cols = st.columns([0.06, 0.94])
            with cols[0]:
                st.markdown(f"<div style='font-size:26px;line-height:1.0'>{avatar}</div>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(
                    f"<div class='{card_class}'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    f"<div style='font-weight:600'>{role_label}</div>"
                    f"<div class='small-muted' style='font-size:12px'>{ts_str}</div>"
                    f"</div>"
                    f"<div style='margin-top:8px'>{content}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                if role == "assistant" and i > 0:
                    st.markdown("---")
                    controls_cols = st.columns([1,1,3,3])
                    if controls_cols[0].button("⭐ Bookmark", key=f"bookmark_{i}"):
                        title = content.strip().split("\n")[0][:60] or "Bookmark"
                        sources = message.get("sources", []) if isinstance(message.get("sources"), list) else []
                        add_bookmark(title=title, content=content, source_sections=sources)
                        try:
                            st.toast("Bookmarked")
                        except Exception:
                            pass
                        safe_rerun()

                    useful_label = "✅ Useful" if feedback_status == 'useful' else "👍 Useful"
                    if controls_cols[1].button(useful_label, key=f"useful_{i}"):
                        log_feedback(i, 'useful')

                    useless_label = "❌ Useless" if feedback_status == 'useless' else "👎 Useless"
                    if controls_cols[2].button(useless_label, key=f"useless_{i}"):
                        log_feedback(i, 'useless')

                    rating_cols = st.columns([1,1,1,1,1])
                    for star_val, col in enumerate(rating_cols, start=1):
                        # show styled button with star label
                        if col.button("★" * star_val, key=f"rate_{i}_{star_val}"):
                            log_feedback(i, 'rating', {"rating": star_val})
                            try:
                                st.toast(f"Rated {star_val} ★")
                            except Exception:
                                pass
                            safe_rerun()

                    if message.get("rating"):
                        st.caption(f"Rating: {'★' * message.get('rating')} ({message.get('rating')}/5)")
                    if feedback_status:
                        st.caption(f"Feedback: *{feedback_status.upper()}*")

    # Input handling
    prompt = None

    if st.session_state.get("pending_prompt"):
        st.info(f"Quick Action queued: {st.session_state.pending_prompt} — it will run when you press Enter or the prompt is consumed.")

    if st.session_state.get("pending_prompt"):
        prompt = st.session_state.pending_prompt.strip()
    else:
        prompt = st.chat_input("Ask about BNS, BNSS, or BSA sections...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "ts": time.time()})

        with st.spinner("Analyzing and generating answer..."):
            if should_carry_over(prompt):
                full_query = st.session_state.last_successful_query + " and " + prompt
                st.info(f"Context Carry-Over: Processing combined query: '{full_query}'")
            else:
                full_query = prompt

            processed_prompt = preprocess_text(full_query)
            predicted_category = "general"
            sections = []

            if processed_prompt.strip() and query_classifier:
                if not processed_prompt.lower().startswith("show common") and not re.search(r'(bns|bnss|bsa)?\s*(?:section\s*)?(\d+)', full_query.lower()):
                    try:
                        predicted_category = query_classifier.predict([processed_prompt])[0]
                        st.info(f"Query classified as: *{predicted_category}*")
                    except Exception:
                        predicted_category = "general"

            sections = find_relevant_sections(full_query, processed_prompt, predicted_category, categorized_legal_data, legal_section_map, tfidf_vectorizer, corpus_vectors, all_legal_data)

            answer = generate_answer_with_llm(full_query, sections)

            if sections and len(prompt.split()) > 1 and not prompt.lower().startswith("show common"):
                st.session_state.last_successful_query = full_query

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer, "ts": time.time()})
        st.session_state.history_log.append(list(st.session_state.messages))

        if st.session_state.get("pending_prompt") and st.session_state.get("pending_timestamp"):
            if time.time() - st.session_state.pending_timestamp < 120:
                st.session_state.pending_prompt = None
                st.session_state.pending_timestamp = None

else:
    # Not authenticated: login/register view (unchanged)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("LegalBot Access 🔒")
        st.info("Please Log In or Register to use the system.")
        tab_register, tab_login = st.tabs(["Register", "Log In"])

        with tab_login:
            st.subheader("Existing User Login")
            with st.form("login_form"):
                login_username = st.text_input("Username", key="login_user")
                login_password = st.text_input("Password", type="password", key="login_pass")
                login_button = st.form_submit_button("Log In")
                if login_button:
                    if verify_login(login_username, login_password):
                        st.session_state[AUTH_STATUS_KEY] = True
                        st.session_state['user'] = login_username.strip().lower()
                        st.success("Login successful!")
                        st.session_state.show_user_menu = False
                        st.session_state.show_confirm_logout = False
                        safe_rerun()
                    else:
                        st.error("Invalid username or password.")

        with tab_register:
            st.subheader("New User Registration")
            with st.form("register_form"):
                reg_username = st.text_input("New Username", key="reg_user")
                reg_password = st.text_input("New Password", type="password", key="reg_pass")
                reg_button = st.form_submit_button("Register")
                if reg_button:
                    if len(reg_username.strip()) < 4 or len(reg_password.strip()) < 6:
                        st.error("Username must be at least 4 characters and password 6 characters.")
                    else:
                        success, message = register_user(reg_username, reg_password)
                        if success:
                            st.session_state[AUTH_STATUS_KEY] = True
                            st.session_state['user'] = reg_username.strip().lower()
                            st.success(message)
                            st.session_state.show_user_menu = False
                            st.session_state.show_confirm_logout = False
                            safe_rerun()
                        else:
                            st.error(message)