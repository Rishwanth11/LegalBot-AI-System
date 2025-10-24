# pdf_extractor.py
import PyPDF2
import os
import json
import glob # To find all PDF files

DATA_FOLDER = "data" # Define the folder where PDFs are stored
OUTPUT_FOLDER = "data" # Define where to save the JSON files

def extract_all_text_from_pdf(pdf_path):
    """Opens a PDF file and extracts text from all pages."""
    if not os.path.exists(pdf_path):
        return f"Error: The file '{pdf_path}' was not found."
        
    all_pages_text = []
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
                        all_pages_text.append(text.strip()) 
                    else:
                        # Append minimal info if no text extracted, helps maintain page count
                        all_pages_text.append(f"[[Page {page_num}: No text extracted]]") 
                except Exception as page_e:
                    print(f"Warning: Could not extract text from page {page_num} of {os.path.basename(pdf_path)}. Error: {page_e}")
                    all_pages_text.append(f"[[Page {page_num}: Error extracting text]]")
            
            return all_pages_text

    except Exception as e:
        # Include filename in error message
        return f"An unexpected error occurred while reading '{os.path.basename(pdf_path)}': {e}"

def structure_text_data(pages_text, source_doc_name):
    """
    (Basic Implementation) Takes list of page texts and structures it.
    Creates one dictionary per page.
    """
    print(f"Structuring data for {source_doc_name}...")
    structured_data = []
    for i, page_text in enumerate(pages_text):
         # Basic cleaning: replace multiple newlines/spaces if needed (optional)
         cleaned_text = ' '.join(page_text.split()) 
         structured_data.append({
             "page_number": i,
             "text": cleaned_text, # Store cleaned text
             "source_document": source_doc_name
         })
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
    # Ensure the output directory exists (it's the same as data folder here)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True) 
    
    # Find all PDF files in the data folder
    pdf_files = glob.glob(os.path.join(DATA_FOLDER, "*.pdf"))
    
    if not pdf_files:
        print(f"Error: No PDF files found in the '{DATA_FOLDER}' directory.")
    else:
        print(f"Found PDF files: {[os.path.basename(f) for f in pdf_files]}")

    # Process each PDF file
    for pdf_path in pdf_files:
        print(f"\n--- Processing {os.path.basename(pdf_path)} ---")
        
        # 1. Extract text from all pages
        raw_pages_text = extract_all_text_from_pdf(pdf_path)
        
        if isinstance(raw_pages_text, str): # Check if extraction returned an error
            print(raw_pages_text)
            continue # Skip to the next file if there was an error
            
        # 2. Structure the extracted text
        pdf_base_name = os.path.basename(pdf_path)
        structured_legal_data = structure_text_data(raw_pages_text, pdf_base_name)
        
        # 3. Save the structured data to a JSON file (named after the PDF)
        output_json_filename = os.path.splitext(pdf_base_name)[0] + ".json"
        output_json_path = os.path.join(OUTPUT_FOLDER, output_json_filename)
        save_to_json(structured_legal_data, output_json_path)

        # Print a small sample from the generated JSON
        if structured_legal_data:
             print(f"\n--- Sample from {output_json_filename} (First entry) ---")
             print(json.dumps(structured_legal_data[0], indent=4))
        
    print("\n--- Processing Complete ---")