# check_models.py
import google.generativeai as genai
import os

try:
    # Read the API key from the environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        print("Please set the key in your terminal first, then run this script again:")
        print("Example: $env:GOOGLE_API_KEY = \"YOUR_API_KEY\"")
        exit()
        
    genai.configure(api_key=api_key)

except Exception as e:
    print(f"An error occurred during configuration: {e}")
    exit()

print("Successfully configured API key. Fetching available models...")
print("="*40)

try:
    # List all models that support the 'generateContent' method
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(model.name)
except Exception as e:
    print(f"An error occurred while trying to list models: {e}")
    print("This might be an issue with your API key permissions or network.")

print("="*40)
print("Finished. Please copy one of the model names from the list above (e.g., 'models/gemini-pro').")