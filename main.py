# main.py - FINAL PROJECT CODE (Integrated Fixes and Professional UI)
import streamlit as st
import json
import os
import glob
import google.generativeai as genai
import re
import numpy as np

# Import our NLP functions and sklearn components (Ensure nlp_processor.py is present)
from nlp_processor import preprocess_text, load_dataset, train_classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- Configuration ---
DATA_FOLDER = "data"

# A map to link categories to their specific JSON files
CATEGORY_TO_FILE = {
    "criminal": "bns.json",
    "procedural": "bnss.json",
    "evidence": "bsa.json"
}

# Configure the Gemini API key
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    st.error("GOOGLE_API_KEY environment variable not set. Please set it in your terminal.")
    st.stop()
except Exception as e:
    st.error(f"Error configuring AI: {e}")
    st.stop()

# --- Utility Functions for State Management ---

def new_chat():
    """Resets the current chat session."""
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am LegalBot. I can answer questions about the BNS (Criminal), BNSS (Procedural), and BSA (Evidence). How can I help?"}
    ]
    st.session_state.last_successful_query = ""
    st.rerun()

def log_feedback(index, feedback_type):
    """Logs the user feedback (useful/useless) for a specific bot message."""
    st.session_state.messages[index]['feedback'] = feedback_type
    
    if 'history_log' not in st.session_state:
         st.session_state.history_log = []
    
    if not st.session_state.history_log or st.session_state.history_log[-1] != st.session_state.messages:
        st.session_state.history_log.append(list(st.session_state.messages))

# --- Data Loading and Model Setup ---
@st.cache_resource
def load_all_models_and_data():
    """
    Loads all JSON data, trains the classifier, and initializes the TF-IDF 
    vectorizer and corpus for RAG search.
    """
    all_data = []
    json_files = glob.glob(os.path.join(DATA_FOLDER, "*.json"))
    
    if not json_files:
        st.error(f"No JSON data files found in '{DATA_FOLDER}'. Run pdf_extractor.py.")
        return None, None, None, None, None, None

    section_map = {} 
    categorized_data = {"criminal": [], "procedural": [], "evidence": []}

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data) 
                
                file_name = os.path.basename(file_path)
                category = None
                for cat, f_name in CATEGORY_TO_FILE.items():
                    if f_name == file_name:
                        category = cat
                        break
                
                if category:
                    categorized_data[category].extend(data)
                
                for section in data:
                    key = (file_name, section['section_number'])
                    section_map[key] = section
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    # 2. Load and Train the classifier
    dataset_path = os.path.join(DATA_FOLDER, "query_dataset.json")
    dataset = load_dataset(dataset_path)
    if not dataset:
        print("Classifier dataset not loaded.")
        classifier = None
    else:
        classifier = train_classifier(dataset)

    # 3. Setup TF-IDF for RAG Ranking 
    corpus = [
        section['text'] 
        for section in all_data 
        if 'text' in section and section['text']
    ]
    
    if not corpus:
        st.error("Fatal Error: Legal data corpus is empty or malformed.")
        return categorized_data, section_map, classifier, None, None, all_data

    corpus_preprocessed = [preprocess_text(text) for text in corpus] 
    
    tfidf_vectorizer = TfidfVectorizer()
    corpus_vectors = tfidf_vectorizer.fit_transform(corpus_preprocessed) 
    
    return categorized_data, section_map, classifier, tfidf_vectorizer, corpus_vectors, all_data


# --- Correlated Retrieval Helper Function ---
def check_for_correlated_sections(ranked_sections, section_map):
    """
    Checks if a murder definition (BNS 101) was found, and adds BNS 103 (Punishment) 
    if it's missing, or vice versa, to ensure complete context.
    """
    definition_key = ("bns.json", "101")
    punishment_key = ("bns.json", "103")
    
    is_definition_retrieved = any(s.get('section_number') == '101' and s.get('source_document') == 'bns.pdf' for s in ranked_sections)
    is_punishment_retrieved = any(s.get('section_number') == '103' and s.get('source_document') == 'bns.pdf' for s in ranked_sections)
    
    if is_definition_retrieved and not is_punishment_retrieved and punishment_key in section_map:
        print("Action: Auto-adding BNS Section 103 (Punishment for murder).")
        ranked_sections.append(section_map[punishment_key])
        
    if is_punishment_retrieved and not is_definition_retrieved and definition_key in section_map:
        print("Action: Auto-adding BNS Section 101 (Definition of murder).")
        ranked_sections.append(section_map[definition_key])
            
    return ranked_sections


# --- Search and Generation (The "Smart" Part) ---
def find_relevant_sections(query, processed_query, category, categorized_data, section_map, vectorizer, corpus_vectors, all_legal_data):
    """
    Performs a smarter search using a combination of direct section lookup and 
    TF-IDF Cosine Similarity ranking within the predicted category.
    """
    query_lower = query.lower().strip()
    found_sections = []
    
    # 1. Handle Chatter/Greetings
    chat_words = ["hi", "hello", "hey", "how are you", "what your name", "what is your name", "love you", "ok", "thanks", "thank you", "no thanks", "no", "can say name", "what are you doing", "what mean"]
    if processed_query in chat_words or processed_query.replace(" ", "") in chat_words:
        return [] 

    # 2. Handle Quick Action Buttons
    if query_lower.startswith("show common"):
        if query_lower == "show common bns sections":
            st.info("Query classified as: **Quick Action (BNS)**")
            return categorized_data.get("criminal", [])[98:108]
        if query_lower == "show common bnss sections":
            st.info("Query classified as: **Quick Action (BNSS)**")
            return categorized_data.get("procedural", [])[:10]
        if query_lower == "show common bsa sections":
            st.info("Query classified as: **Quick Action (BSA)**")
            return categorized_data.get("evidence", [])[:10]
        return [] 

    # 3. Handle Direct Section Number Match
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
            for doc_name in doc_map.values():
                key = (doc_name, section_num)
                if key in section_map:
                    found_sections.append(section_map[key])
    
    if found_sections:
        st.info("Query classified as: **Section Number Lookup**")
        return found_sections
    
    # 4. Ranked Search using TF-IDF
    if category in categorized_data:
        data_to_search = categorized_data[category]
    else:
        data_to_search = all_legal_data 

    valid_sections_to_search = [s for s in data_to_search if 'text' in s and s['text']]
    data_indices = [all_legal_data.index(section) for section in valid_sections_to_search if section in all_legal_data]
    
    # CRITICAL FIX: Use .size to check if the sparse matrix is empty, avoiding ValueError
    if not data_indices or not vectorizer or corpus_vectors.size == 0:
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
        if score > 0.1: # Threshold for relevance
             ranked_sections.append(section)
             
    # 5. POST-RETRIEVAL CORRELATION (NEW STEP)
    # Automatically add BNS 103/101 if the other is retrieved
    ranked_sections = check_for_correlated_sections(ranked_sections, legal_section_map)

    return ranked_sections


def generate_answer_with_llm(query, relevant_sections):
    """
    Uses a Generative LLM (Gemini) to create an answer based on RAG context.
    """
    if relevant_sections:
        if query.lower().startswith("show common"):
            context = (
                "You are LegalBot. The user has asked for a list of common sections. "
                "List the following sections, giving *only* the Section Number and its Title (the 'title' field, which is a preview of the text). "
                "Format it as a simple bulleted list.\n"
                "Example: \n* **Section 101:** Punishment for murder...\n\n"
                "--- Relevant Legal Sections ---\n\n"
            )
        else:
            context = (
                "You are LegalBot, a helpful AI legal assistant. Your task is to answer the user's question.\n"
                "Base your answer *only* on the relevant legal sections provided below.\n"
                "Quote the section number and source (e.g., BNS Section 101) for your information.\n"
                "Do not use any outside knowledge. If the answer is not in the provided sections, say so.\n\n"
                f"**User's Question:** {query}\n\n"
                "--- Relevant Legal Sections ---\n\n"
            )
        
        for i, section in enumerate(relevant_sections):
            source_doc = section['source_document'].split('.')[0].upper()
            title_preview = section['title']
            context += f"**Source {i+1} ({source_doc} Section {section['section_number']}: {title_preview}):**\n"
            context += f"{section['text']}\n\n"
        
        context += "--- End of Sections ---\n\n"
        
        if query.lower().startswith("show common"):
             context += "Your Answer (as a bulleted list):"
        else:
            context += "Please provide a clear and direct answer to the user's question based *only* on these sections."

    else:
        context = (
            "You are LegalBot, a helpful AI legal assistant. You are speaking to a user.\n"
            "If the user's question is a greeting or general chat (like 'hi', 'i love you', 'ok', 'thanks', 'no', 'what are you doing'), respond politely and conversationally.\n"
            "If the user is asking a legal question that you don't have specific sections for, "
            "politely explain that you can only answer questions about the BNS, BNSS, and BSA "
            "and suggest they try searching for specific keywords or section numbers.\n\n"
            f"**User's Question:** {query}\n\n"
            "**Your Answer:**"
        )

    try:
        model = genai.GenerativeModel('models/gemini-pro-latest') 
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        print(f"Error during AI generation: {e}")
        return f"An error occurred while generating the answer: {e}"

# --- Load Data and Models ONCE ---
categorized_legal_data, legal_section_map, query_classifier, tfidf_vectorizer, corpus_vectors, all_legal_data = load_all_models_and_data()

if not query_classifier:
    st.error("NLP Classification model failed to load. The app cannot proceed.")
    st.stop()

# --- CRITICAL FIX: Initialize Session State Attributes AT THE TOP ---
if 'history_log' not in st.session_state:
    st.session_state.history_log = []

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am LegalBot. I can answer questions about the BNS (Criminal), BNSS (Procedural), and BSA (Evidence). How can I help?"}
    ]
    
if 'last_successful_query' not in st.session_state:
    st.session_state.last_successful_query = ""
# --- END CRITICAL FIX ---


# --- Streamlit App UI ---
st.title("⚖️ LegalBot")
st.subheader("AI-Powered Judiciary Reference System")

# --- Sidebar ---
with st.sidebar:
    st.title("LegalBot Menu")
    
    # 1. New Chat Button
    if st.button("New Chat", use_container_width=True):
        new_chat()
    
    st.markdown("---")
    st.title("Quick Actions")
    st.write("Click a button to see some common sections:")
    
    # 2. Quick Action Buttons
    if st.button("Common BNS Sections (Criminal)", use_container_width=True):
        st.session_state.quick_action = "show common bns sections"
    if st.button("Common BNSS Sections (Procedural)", use_container_width=True):
        st.session_state.quick_action = "show common bnss sections"
    if st.button("Common BSA Sections (Evidence)", use_container_width=True):
        st.session_state.quick_action = "show common bsa sections"
        
    st.markdown("---")
    st.title("History Tab")

    if not st.session_state.history_log:
        st.info("No past conversations logged.")
    else:
        for idx in range(len(st.session_state.history_log) -1, -1, -1):
            history = st.session_state.history_log[idx]
            title = next((msg['content'] for msg in history if msg['role'] == 'user'), "New Chat")
            
            with st.expander(f"Chat {idx + 1}: {title[:30]}..."):
                for msg in history:
                    st.caption(f"**{msg['role'].title()}**: {msg['content'][:50]}...")
                
                if st.button("Load Chat", key=f"load_hist_{idx}", use_container_width=True):
                    st.session_state.messages = history
                    st.rerun()

# --- Main Chat Interface & Feedback Implementation ---
for i, message in enumerate(st.session_state.messages):
    role = message["role"]
    content = message["content"]
    feedback_status = message.get('feedback', None)

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant" and i > 0: 
            
            st.markdown("---")
            
            cols = st.columns([1, 1, 1, 3]) 
            
            useful_label = "👍 Useful" if feedback_status != 'useful' else "✅ Useful"
            if cols[0].button(useful_label, key=f"useful_{i}"):
                log_feedback(i, 'useful')
                st.rerun()

            useless_label = "👎 Useless" if feedback_status != 'useless' else "❌ Useless"
            if cols[1].button(useless_label, key=f"useless_{i}"):
                log_feedback(i, 'useless')
                st.rerun()

            # --- VISUAL FIX: Re-insert st.code for GUARANTEED Copy Functionality ---
            # NOTE: This will visually display the answer in a small box, 
            # but it is the only way to get a native copy button in standard Streamlit.
            cols[2].code(content, language='text') 
            
            if feedback_status:
                 cols[3].caption(f"Feedback recorded: **{feedback_status.upper()}**")

# Check and execute Quick Action button click
if "quick_action" in st.session_state and st.session_state.quick_action:
    prompt = st.session_state.quick_action
    st.session_state.quick_action = None
else:
    prompt = st.chat_input("Ask about BNS, BNSS, or BSA sections...")

# --- Handle User Prompt Submission ---
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Analyzing and generating answer..."):
        
        # --- CONTEXT-AWARE QUERY PREPARATION ---
        if len(prompt.split()) <= 3 and st.session_state.last_successful_query:
            full_query = st.session_state.last_successful_query + " and " + prompt
            st.info(f"Context Carry-Over: Processing combined query: '{full_query}'")
        else:
            full_query = prompt
        
        processed_prompt = preprocess_text(full_query)
        predicted_category = "general"
        sections = []
        
        # 1. Prediction/Routing
        if processed_prompt.strip() and query_classifier:
            if not processed_prompt.lower().startswith("show common") and not re.search(r'(bns|bnss|bsa)?\s*(?:section\s*)?(\d+)', full_query.lower()):
                 predicted_category = query_classifier.predict([processed_prompt])[0]
                 st.info(f"Query classified as: **{predicted_category}**")
        
        # Call the search function
        sections = find_relevant_sections(full_query, processed_prompt, predicted_category, categorized_legal_data, legal_section_map, tfidf_vectorizer, corpus_vectors, all_legal_data)
        
        # 2. Generation
        answer = generate_answer_with_llm(full_query, sections)
        
        # --- SAVE THE SUCCESSFUL QUERY FOR NEXT TURN ---
        if sections and len(prompt.split()) > 1 and not prompt.lower().startswith("show common"): 
             st.session_state.last_successful_query = full_query
    
    # 3. Display and Log
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    st.session_state.history_log.append(list(st.session_state.messages))