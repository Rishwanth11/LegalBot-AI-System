# nlp_processor.py
import spacy
import nltk
import json
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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

# Load the SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("\n--- SpaCy model 'en_core_web_sm' not found. ---")
    print("Please run this command in your terminal:")
    print("python -m spacy download en_core_web_sm")
    exit()
    
# Get English stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and lemmatizes a text query.
    """
    doc = nlp(text.lower()) # Use SpaCy for efficient processing
    
    processed_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and token.is_alpha:
            processed_tokens.append(token.lemma_)
            
    return " ".join(processed_tokens) # Return a single string

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

# --- NEW: Model Training Function ---
def train_classifier(dataset):
    """
    Trains a classification model based on the sample dataset.
    """
    print("Training classification model...")
    if not dataset:
        print("Cannot train model: Dataset is empty.")
        return None

    # 1. Prepare data
    queries = [preprocess_text(item['query']) for item in dataset]
    categories = [item['category'] for item in dataset]

    # 2. Create a machine learning "pipeline"
    # This pipeline first converts text to numbers (TfidfVectorizer)
    # and then trains a classifier (LogisticRegression)
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression())
    ])

    # 3. Train the model
    try:
        model_pipeline.fit(queries, categories)
        print("Model training complete.")
        return model_pipeline
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

# --- Main execution block (for testing) ---
if __name__ == "__main__":
    
    # 1. Load the dataset
    dataset = load_dataset(os.path.join("data", "query_dataset.json"))
    
    if dataset:
        # 2. Train the model
        classifier = train_classifier(dataset)
        
        if classifier:
            print("\n--- Testing Model Classification ---")
            
            # 3. Test with new queries
            test_queries = [
                "Tell me about murder punishment",
                "What is the rule for a police arrest?",
                "What is a witness statement?"
            ]
            
            for query in test_queries:
                processed_query = preprocess_text(query)
                prediction = classifier.predict([processed_query])[0]
                print(f"Query:     '{query}'")
                print(f"Predicted: '{prediction}'")
                print("-" * 20)
    else:
        print("Stopping script because dataset could not be loaded.")