# main.py
import streamlit as st
import json
import os
import glob
import google.generativeai as genai
import re
# Import our new NLP functions
from nlp_processor import preprocess_text, load_dataset, train_classifier

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

# --- Data Loading ---
@st.cache_resource  # Use cache_resource for models/data that shouldn't be re-created
def load_all_models_and_data():
    """
    Loads all JSON data, the sample dataset, and trains the classifier.
    This function runs only once.
    """
    # 1. Load all legal sections from all JSONs
    all_data = []
    json_files = glob.glob(os.path.join(DATA_FOLDER, "*.json"))
    
    if not json_files:
        st.error(f"No JSON data files found in '{DATA_FOLDER}'. Run pdf_extractor.py.")
        return None, None, None

    print(f"Loading data from files: {json_files}")
    section_map = {} # For fast lookup by section number
    
    # Create a dictionary to hold data per category
    categorized_data = {
        "criminal": [],
        "procedural": [],
        "evidence": []
    }

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data) # We still need a list of all data for fallback
                
                # Find the category for this file
                file_name = os.path.basename(file_path)
                category = None
                for cat, f_name in CATEGORY_TO_FILE.items():
                    if f_name == file_name:
                        category = cat
                        break
                
                if category:
                    categorized_data[category].extend(data)
                
                # Create the section map for fast lookups
                for section in data:
                    key = (file_name, section['section_number'])
                    section_map[key] = section
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    print(f"Successfully loaded {len(all_data)} total sections.")

    # 2. Load the training dataset
    dataset_path = os.path.join(DATA_FOLDER, "query_dataset.json")
    dataset = load_dataset(dataset_path)
    if not dataset:
        st.error(f"Could not load {dataset_path}. Classifier cannot be trained.")
        return all_data, section_map, None

    # 3. Train the classifier
    classifier = train_classifier(dataset)
    if not classifier:
        st.error("Model training failed. Check nlp_processor.py.")
        return all_data, section_map, None
        
    print("NLP Classifier trained and ready.")
    return categorized_data, section_map, classifier

# --- Search and Generation (The "Smart" Part) ---
def find_relevant_sections(query, processed_query, category, categorized_data, section_map):
    """
    Performs a smarter search to find relevant sections.
    """
    query_lower = query.lower().strip()
    
    # --- UPDATED CHAT WORDS ---
    chat_words = ["hi", "hello", "hey", "how are you", "what your name", 
                  "what is your name", "love you", "ok", "thanks", "thank you", 
                  "no thanks", "no", "can say name", "what are you doing", "what mean"]
    if processed_query in chat_words or processed_query.replace(" ", "") in chat_words:
        return [] # Return an empty list for chatter

    # --- UPDATED LOGIC ---
    # Try to find a section number (e.g., "bns 101", "section 326", "99")
    # We check for this FIRST, before classification.
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
    
    # If no section number matched, do a text search only in the predicted category
    if not found_sections:
        data_to_search = categorized_data.get(category, [])
        if not data_to_search:
             # Fallback to all data if category is weird
             data_to_search = categorized_data["criminal"] + categorized_data["procedural"] + categorized_data["evidence"]
             
        # --- NEW SEARCH LOGIC (INTERSECTION / "AND" search) ---
        # Search for the lemmatized (root) words from the query
        # We will require ALL tokens to be in the text
        
        search_tokens = processed_query.split()
        if not search_tokens:
            return [] # No search terms

        for section in data_to_search:
            text_lower = section["text"].lower()
            all_tokens_found = True # Start by assuming all tokens are found
            for token in search_tokens:
                # Use regex to find the token as a whole word
                query_regex = r'\b' + re.escape(token) + r'\b'
                if not re.search(query_regex, text_lower):
                    all_tokens_found = False # If one token is missing, fail this section
                    break 
            
            if all_tokens_found: # Only add if ALL tokens were found
                if section not in found_sections:
                    found_sections.append(section)
                
    return found_sections[:5] # Return top 5 matches

def generate_answer_with_llm(query, relevant_sections):
    """
    Uses a Generative LLM (Gemini) to create an answer.
    """
    
    # --- PROMPT 1: RAG (Retrieval-Augmented Generation) ---
    if relevant_sections:
        context = (
            "You are LegalBot, a helpful AI legal assistant. Your task is to answer the user's question.\n"
            "Base your answer only on the relevant legal sections provided below.\n"
            "Quote the section number and source (e.g., BNS Section 101) for your information.\n"
            "Do not use any outside knowledge. If the answer is not in the provided sections, say so.\n\n"
            f"*User's Question:* {query}\n\n"
            "--- Relevant Legal Sections ---\n\n"
        )
        
        for i, section in enumerate(relevant_sections):
            source_doc = section['source_document'].split('.')[0].upper()
            title_preview = section['title'] # Use the title we created
            
            context += f"*Source {i+1} ({source_doc} Section {section['section_number']}: {title_preview}):*\n"
            context += f"{section['text']}\n\n"
        
        context += "--- End of Sections ---\n\n"
        context += "Please provide a clear and direct answer to the user's question based only on these sections."

    # --- PROMPT 2: General Conversation ---
    else:
        context = (
            "You are LegalBot, a helpful AI legal assistant. You are speaking to a user.\n"
            "If the user's question is a greeting or general chat (like 'hi', 'i love you', 'ok', 'thanks', 'no', 'what are you doing'), respond politely and conversationally.\n"
            "If the user is asking a legal question that you don't have specific sections for, "
            "politely explain that you can only answer questions about the BNS, BNSS, and BSA "
            "and suggest they try searching for specific keywords or section numbers.\n\n"
            f"*User's Question:* {query}\n\n"
            "*Your Answer:*"
        )

    # Call the Gemini model with the chosen prompt
    try:
        model = genai.GenerativeModel('models/gemini-pro-latest') 
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        print(f"Error during AI generation: {e}")
        return f"An error occurred while generating the answer: {e}"

# --- Load Data and Train Models ONCE ---
categorized_legal_data, legal_section_map, query_classifier = load_all_models_and_data()
if not query_classifier:
    st.error("NLP Classification model failed to load. The app cannot proceed.")
    st.stop()

# --- Streamlit App UI ---
st.title("⚖ LegalBot")
st.subheader("AI-Powered Judiciary Reference System")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am LegalBot. I can answer questions about the BNS (Criminal), BNSS (Procedural), and BSA (Evidence). How can I help?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about BNS, BNSS, or BSA sections..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Analyzing and generating answer..."):
        # 1. Preprocess the query
        processed_prompt = preprocess_text(prompt)
        
        # --- UPDATED LOGIC ---
        predicted_category = ""
        sections = []
        
        # Check for section number match first (fixes "section 202" bug)
        section_match = re.search(r'(bns|bnss|bsa)?\s*(?:section\s*)?(\d+)', prompt.lower())
        
        chat_words = ["hi", "hello", "hey", "how are you", "what your name", 
                      "what is your name", "love you", "ok", "thanks", "thank you", 
                      "no thanks", "no", "can say name", "what are you doing", "what mean"]
        
        if processed_prompt.replace(" ", "") in chat_words:
            # It's a greeting
            sections = []
        elif section_match:
            # It's a section number lookup, skip classification
            st.info("Query classified as: *Section Number Lookup*")
            sections = find_relevant_sections(prompt, processed_prompt, "general", categorized_legal_data, legal_section_map)
        elif processed_prompt.strip(): # Check if it's not empty
            # It's a topic query, so classify it
            predicted_category = query_classifier.predict([processed_prompt])[0]
            st.info(f"Query classified as: *{predicted_category}*") # Show the category
            sections = find_relevant_sections(prompt, processed_prompt, predicted_category, categorized_legal_data, legal_section_map)
        
        # 4. Generate: Ask the LLM to create an answer
        answer = generate_answer_with_llm(prompt, sections)
    
    # Bot's message
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})