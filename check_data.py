# check_data.py - Utility script to verify data integrity

import os
import json
import glob

DATA_FOLDER = "data"
REQUIRED_LEGAL_FILES = ["bns.json", "bnss.json", "bsa.json"]
REQUIRED_NLP_FILE = "query_dataset.json"
USER_DB_FILE = "users.json"

def check_file_integrity(file_path, required_keys=None):
    """Checks if a JSON file exists, is valid JSON, and contains data."""
    if not os.path.exists(file_path):
        return False, f"File not found."

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return False, "File is corrupted (Invalid JSON format)."
    except Exception as e:
        return False, f"Read error: {e}"

    if not data:
        return False, "File is empty or contains no data."

    if required_keys and isinstance(data, list):
        # Check if the first item contains the required keys (e.g., 'text' for legal data)
        if data and isinstance(data[0], dict) and not all(key in data[0] for key in required_keys):
             return False, f"Data structure invalid. Missing one of these keys: {required_keys}"

    return True, f"Contains {len(data)} records."

def run_data_checks():
    print("--- LegalBot Data Integrity Check ---")
    
    # 1. Check Legal Data Corpus
    print("\n[1] Checking Legal Corpus Integrity:")
    all_ok = True
    
    for filename in REQUIRED_LEGAL_FILES:
        path = os.path.join(DATA_FOLDER, filename)
        status, message = check_file_integrity(path, required_keys=['text', 'section_number'])
        
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {filename}: {message}")
        if not status:
            all_ok = False

    # 2. Check NLP Training Data
    print("\n[2] Checking NLP Training Data:")
    path = os.path.join(DATA_FOLDER, REQUIRED_NLP_FILE)
    status, message = check_file_integrity(path, required_keys=['query', 'category'])
    
    status_icon = "✅" if status else "❌"
    print(f"  {status_icon} {REQUIRED_NLP_FILE}: {message}")
    if not status:
        all_ok = False

    # 3. Check User Database Status
    print("\n[3] Checking User Database Status:")
    path = os.path.join(DATA_FOLDER, USER_DB_FILE)
    if os.path.exists(path):
        status, message = check_file_integrity(path)
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {USER_DB_FILE}: {message} (Authentication is active)")
    else:
        print(f"  ⚠️ {USER_DB_FILE}: File not found. Registration is required upon launch.")
        
    print("\n--- Check Complete ---")
    if all_ok:
        print("STATUS: All essential legal and NLP data files are valid and ready to use.")
    else:
        print("STATUS: Errors found. Please run pdf_extractor.py or verify your data files.")

if __name__ == "__main__":
    run_data_checks()