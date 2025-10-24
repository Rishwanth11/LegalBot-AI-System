# main.py
import streamlit as st
import json
import os
import glob # To find our JSON files

DATA_FOLDER = "data"

@st.cache_data  # This decorator caches the data so it doesn't reload every time
def load_all_legal_data():
    """
    Loads and combines data from all .json files in the 'data' folder.
    """
    all_data = []
    json_files = glob.glob(os.path.join(DATA_FOLDER, "*.json"))
    
    if not json_files:
        st.error(f"No JSON data files found in '{DATA_FOLDER}' folder. Please run pdf_extractor.py first.")
        return []

    print(f"Loading data from files: {json_files}")
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data) # Add all sections from this file to our main list
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    print(f"Successfully loaded {len(all_data)} total pages/sections.")
    return all_data

def find_relevant_section(query, legal_data):
    """
    Performs a simple keyword search through the loaded legal data.
    """
    query = query.lower() # Convert query to lowercase for easier matching
    
    # --- Simple Search Logic ---
    # We will look for the query in the 'text' of each section.
    # This is a basic search and can be improved later with NLP/regex.
    
    found_sections = []
    for section in legal_data:
        if query in section["text"].lower():
            found_sections.append(section)
            
    if not found_sections:
        return "I'm sorry, I couldn't find a relevant section for your query. Please try different keywords."

    # --- Format the Response ---
    # For now, let's just return the text from the first match.
    first_match = found_sections[0]
    
    response = f"**From {first_match['source_document']} (Page {first_match['page_number']}):**\n\n"
    response += f"{first_match['text']}"
    
    return response

# --- Load the data once when the app starts ---
legal_data_all = load_all_legal_data()

# --- Main App Logic (from before) ---
st.title("⚖️ LegalBot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Explain your legal problem here..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the chatbot's response by searching the data
    response = find_relevant_section(prompt, legal_data_all)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})