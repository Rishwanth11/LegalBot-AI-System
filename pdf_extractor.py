# pdf_extractor.py
import PyPDF2
import os
import json
import glob
import re  # Import the Regular Expressions library

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
    This new pattern is simpler and more robust.
    """
    print(f"Structuring data for {source_doc_name} using Regex...")
    structured_data = []
    
    # --- UPDATED REGEX ---
    # This pattern simply looks for a number followed by a period,
    # and then captures all text until the next number/period or the end of the file.
    
    pattern = re.compile(
        r"\n(\d+)\.\s+((?:.|\n)*?)(?=\n\d+\.\s+|\Z)",
        re.IGNORECASE
    )

    matches = pattern.finditer(full_text)
    
    for match in matches:
        section_number = match.group(1).strip()
        # The first ~100 chars of the text will serve as a 'title'/'preview'
        text_content = ' '.join(match.group(2).strip().split()) # Clean up text
        
        if text_content: # Only add if we found text
            structured_data.append({
                "section_number": section_number,
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

# --- Main execution block ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True) 
    
    pdf_files = glob.glob(os.path.join(DATA_FOLDER, "*.pdf"))
    
    if not pdf_files:
        print(f"Error: No PDF files found in the '{DATA_FOLDER}' directory.")
    else:
        print(f"Found PDF files: {[os.path.basename(f) for f in pdf_files]}")

    for pdf_path in pdf_files:
        pdf_base_name = os.path.basename(pdf_path)
        print(f"\n--- Processing {pdf_base_name} ---")
        
        # 1. Extract all text into one big string
        full_text = extract_all_text_from_pdf(pdf_path)
        
        if not full_text:
            continue 
            
        # 2. Structure the text using Regex
        structured_legal_data = structure_text_with_regex(full_text, pdf_base_name)
        
        # 3. Save the new structured data (overwrites the old page-based JSON)
        output_json_filename = os.path.splitext(pdf_base_name)[0] + ".json"
        output_json_path = os.path.join(OUTPUT_FOLDER, output_json_filename)
        
        save_to_json(structured_legal_data, output_json_path)

        # Print a sample from the generated JSON
        if structured_legal_data:
             print(f"\n--- Sample from {output_json_filename} (First section found) ---")
             print(json.dumps(structured_legal_data[0], indent=4))
        
    print("\n--- All PDF processing complete ---")