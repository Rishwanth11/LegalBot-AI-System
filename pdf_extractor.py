# pdf_extractor.py
import PyPDF2
import os
import json
import glob
import re  # Import the Regular Expressions library
import nltk # Import NLTK

DATA_FOLDER = "data"
OUTPUT_FOLDER = "data"

def extract_all_text_from_pdf(pdf_path):
    """Opens a PDF and extracts all text into a single string."""
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' was not found.")
        return None
        
    full_text = ""
    try:
        with open(pdf_path, 'rb') as pdf_file: 
            reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(reader.pages)
            print(f"Reading {num_pages} pages from {os.path.basename(pdf_path)}...")
            
            for page_num in range(num_pages):
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        # --- NEW CLEANING ---
                        # Remove common PDF headers/footers that confuse the parser
                        # This removes lines that are just page numbers
                        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE) 
                        text = re.sub(r"THE BHARATIYA NYAYA SANHITA, 2023", "", text, flags=re.IGNORECASE)
                        text = re.sub(r"THE BHARATIYA NAGARIK SURAKSHA SANHITA, 2023", "", text, flags=re.IGNORECASE)
                        text = re.sub(r"THE BHARATIYA SAKSHYA ADHINIYAM, 2023", "", text, flags=re.IGNORECASE)
                        
                        full_text += text + "\n" # Add text and a newline
                except Exception as page_e:
                    print(f"Warning: Could not read page {page_num} of {os.path.basename(pdf_path)}. Error: {page_e}")
            
            # Clean up common PDF extraction issues
            full_text = re.sub(r'(\w)-\n(\w)', r'\1\2', full_text) # Re-join hyphenated words
            full_text = re.sub(r'\n+', '\n', full_text) # Remove extra blank lines
            return full_text

    except Exception as e:
        print(f"An unexpected error occurred while reading '{os.path.basename(pdf_path)}': {e}")
        return None

def structure_text_with_regex(full_text, source_doc_name):
    """
    Uses Regex to find and structure legal sections.
    This new pattern is more robust.
    """
    print(f"Structuring data for {source_doc_name} using Regex...")
    structured_data = []
    
    # --- UPDATED ROBUST REGEX ---
    # This pattern looks for a section number (e.g., "99.") at the beginning of a line.
    # It then captures ALL text (including newlines) until it finds the *next*
    # section number at the start of a line, OR the end of the document.
    
    pattern = re.compile(
        r"^(\d+)\.\s+((?:.|\n)*?)(?=^\d+\.\s|\Z)",
        re.MULTILINE # This makes ^ match the start of each line
    )

    matches = pattern.finditer(full_text)
    
    for match in matches:
        section_number = match.group(1).strip()
        
        # Clean up the text content
        text_content = ' '.join(match.group(2).strip().split())
        
        # Extract the first sentence or first 15 words as a "title"
        title = " ".join(text_content.split()[:15]) + "..."
        
        if text_content: # Only add if we found text
            structured_data.append({
                "section_number": section_number,
                "title": title, # Use the first part of the text as a title/preview
                "text": text_content, # The full text of the section
                "source_document": source_doc_name
            })
            
    print(f"Found and structured {len(structured_data)} sections.")
    return structured_data

def save_to_json(data, output_path):
    """Saves the structured data list to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved structured data to {output_path}")
    except Exception as e:
        print(f"Error saving data to JSON '{output_path}': {e}")

# --- Main execution block (for testing) ---
if __name__ == "__main__":
    
    # --- Setup NLTK (Downloads if not found) ---
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
        
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt...")
        nltk.download('punkt')
    # --- End Setup ---
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True) 
    pdf_files = glob.glob(os.path.join(DATA_FOLDER, "*.pdf"))
    
    if not pdf_files:
        print(f"Error: No PDF files found in the '{DATA_FOLDER}' directory.")
    else:
        print(f"Found PDF files: {[os.path.basename(f) for f in pdf_files]}")

    for pdf_path in pdf_files:
        pdf_base_name = os.path.basename(pdf_path)
        print(f"\n--- Processing {pdf_base_name} ---")
        
        full_text = extract_all_text_from_pdf(pdf_path)
        
        if not full_text:
            continue 
            
        structured_legal_data = structure_text_with_regex(full_text, pdf_base_name)
        
        output_json_filename = os.path.splitext(pdf_base_name)[0] + ".json"
        output_json_path = os.path.join(OUTPUT_FOLDER, output_json_filename)
        
        save_to_json(structured_legal_data, output_json_path)

        if structured_legal_data:
             print(f"\n--- Sample from {output_json_filename} (First section found) ---")
             print(json.dumps(structured_legal_data[0], indent=4))
        
    print("\n--- All PDF processing complete ---")
    