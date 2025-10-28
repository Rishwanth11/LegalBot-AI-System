# main.py
import streamlit as st
import json
import os
import glob
import google.generativeai as genai
import re

# --- Configuration ---
DATA_FOLDER = "data"
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    st.error("GOOGLE_API_KEY environment variable not set. Please set it in your terminal.")
    st.stop()
except Exception as e:
    st.error(f"Error configuring AI: {e}")
    st.stop()

# --- Data Loading ---
@st.cache_data
def load_all_legal_data():
    """Loads and combines data from all .json files."""
    all_data = []
    json_files = glob.glob(os.path.join(DATA_FOLDER, "*.json"))
    
    if not json_files:
        st.error(f"No JSON data files found in '{DATA_FOLDER}'. Run pdf_extractor.py.")
        return [], {}

    print(f"Loading data from files: {json_files}")
    section_map = {} # For fast lookup by section number
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data) 
                for section in data:
                    key = (os.path.basename(file_path), section['section_number'])
                    section_map[key] = section
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    print(f"Successfully loaded {len(all_data)} total sections.")
    return all_data, section_map

# --- Search and Generation (The "Smart" Part) ---
def find_relevant_sections(query, legal_data, section_map):
    """
    Performs a smarter search to find relevant sections.
    """
    query_lower = query.lower().strip()
    
    greetings = ["hi", "hello", "hey", "how are you", "whats is your name", "what is your name", "i love you"]
    if query_lower in greetings:
        return [] # Return an empty list for greetings

    # Try to find a section number (e.g., "bns 101", "section 326", "99")
    section_match = re.search(r'(bns|bnss|bsa)?\s*(?:section\s*)?(\d+)', query_lower)
    
    found_sections = []
    
    doc_map = {
        "bns": "bns.json",
        "bnss": "bnss.json",
        "bsa": "bsa.json"
    }

    if section_match:
        doc_key = section_match.group(1) # e.g., "bns" or None
        section_num = section_match.group(2) # e.g., "326"
        
        if doc_key: # e.g., "bns 326"
            key = (doc_map[doc_key], section_num)
            if key in section_map:
                found_sections.append(section_map[key])
        else: # e.g., "section 326" or just "326"
            for doc_name in doc_map.values():
                key = (doc_name, section_num)
                if key in section_map:
                    found_sections.append(section_map[key])
    
    # If no section number matched, do a text search
    if not found_sections:
         for section in legal_data:
            # Check if query is in the first 200 chars of text (like a title search)
            if query_lower in section["text"][:200].lower():
                found_sections.append(section)
            # If not found in title, check the full text
            elif query_lower in section["text"].lower():
                found_sections.append(section)
                
    # Return top 5 matches
    return found_sections[:5] 

def generate_answer_with_llm(query, relevant_sections):
    """
    Uses a Generative LLM (Gemini) to create an answer.
    """
    
    # --- PROMPT 1: RAG (Retrieval-Augmented Generation) ---
    if relevant_sections:
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
            # Get the first part of the text as a "title"
            title_preview = ' '.join(section['text'].split()[:15]) + "..."
            
            context += f"**Source {i+1} ({source_doc} Section {section['section_number']}: {title_preview}):**\n"
            context += f"{section['text']}\n\n"
        
        context += "--- End of Sections ---\n\n"
        context += "Please provide a clear and direct answer to the user's question based *only* on these sections."

    # --- PROMPT 2: General Conversation ---
    else:
        context = (
            "You are LegalBot, a helpful AI legal assistant. You are speaking to a user.\n"
            "If the user's question is a greeting or general chat (like 'hi', 'i love you'), respond politely and conversationally.\n"
            "If the user is asking a legal question that you don't have specific sections for, "
            "politely explain that you can only answer questions about the BNS, BNSS, and BSA "
            "and suggest they try searching for specific keywords or section numbers.\n\n"
            f"**User's Question:** {query}\n\n"
            "**Your Answer:**"
        )

    # Call the Gemini model with the chosen prompt
    try:
        model = genai.GenerativeModel('models/gemini-pro-latest') 
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        print(f"Error during AI generation: {e}")
        return f"An error occurred while generating the answer: {e}"

# --- Load Data ---
legal_data_all, legal_section_map = load_all_legal_data()
if not legal_data_all:
    st.error("Legal data could not be loaded. Stopping app.")
    st.stop()

# --- Streamlit App UI ---
st.title("⚖️ LegalBot")
st.subheader("AI-Powered Judiciary Reference System")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am LegalBot. You can ask me about the Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), and Bharatiya Sakshya Adhiniyam (BSA). How can I help?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about BNS, BNSS, or BSA sections..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Finding relevant sections and generating answer..."):
        sections = find_relevant_sections(prompt, legal_data_all, legal_section_map)
        answer = generate_answer_with_llm(prompt, sections)
    
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})