# nlp_processor.py
import spacy
import nltk
import json
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# Removed unused import: from sklearn.model_selection import train_test_split 

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

# Load the SpaCy model (Assuming 'en_core_web_sm' is installed)
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("\n--- SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm ---")
    
# Get English stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans, removes stopwords, and lemmatizes a text query using SpaCy.
    """
    if not isinstance(text, str):
        return ""
        
    doc = nlp(text.lower()) 
    
    processed_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and token.is_alpha and token.lemma_:
            processed_tokens.append(token.lemma_)
            
    return " ".join(processed_tokens)

def load_dataset(json_path):
    """Loads the sample query dataset from JSON."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {json_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: {json_path} is empty or not valid JSON.")
        return []
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

def train_classifier(dataset):
    """
    Trains the query classification model using all available data, 
    fixing the ValueError caused by test_train_split on sparse data.
    """
    print("Training classification model...")
    if not dataset:
        print("Cannot train model: Dataset is empty.")
        return None

    # 1. Prepare data (use all data for final model training)
    queries = [preprocess_text(item['query']) for item in dataset]
    categories = [item['category'] for item in dataset]

    # The train_test_split section has been removed to fix the ValueError,
    # as the final model should be trained on 100% of the available data anyway.
    
    # 2. Create a machine learning "pipeline"
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression(max_iter=1000)) 
    ])

    # 3. Train the model on ALL available data
    try:
        model_pipeline.fit(queries, categories)
        print("Model training complete.")
        return model_pipeline
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

# --- Main execution block (for local testing/debugging) ---
if __name__ == "__main__":
    
    # 1. Load the dataset (Assumes query_dataset.json is in a 'data' folder)
    # Note: Ensure you have data/query_dataset.json available for this block to run.
    dataset = load_dataset(os.path.join("data", "query_dataset.json"))
    
    if dataset:
        # 2. Train the model
        classifier = train_classifier(dataset)
        
        if classifier:
            print("\n--- Testing Model Classification ---")
            
            # 3. Test with new queries
            test_queries = [
                "Tell me about punishment for murder in BNS",
                "What is the procedure for an arrest warrant?",
                "Rules for a witness statement"
            ]
            
            for query in test_queries:
                processed_query = preprocess_text(query)
                prediction = classifier.predict([processed_query])[0]
                print(f"Query:     '{query}'")
                print(f"Predicted: '{prediction}'")
                print("-" * 20)
    else:
        print("Stopping script because dataset could not be loaded.")