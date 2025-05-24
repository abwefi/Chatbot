import csv
import random
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, render_template, request
import re
import pickle
import os

app = Flask(__name__)

# Download required NLTK data (updated for newer NLTK versions)
required_downloads = [
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
    ('corpora/omw-1.4', 'omw-1.4')
]

for path, name in required_downloads:
    try:
        nltk.data.find(path)
    except LookupError:
        print(f"Downloading {name}...")
        nltk.download(name, quiet=True)

class AdvancedChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000, lowercase=True, stop_words='english')
        self.intents = []
        self.responses = []
        self.intent_vectors = None
        self.fallback_responses = [
            "I'm still learning. Could you rephrase that?",
            "That's an interesting question. Can you provide more context?",
            "I'm not quite sure about that. Try asking differently.",
            "My neural networks are processing... Could you be more specific?",
            "I don't have sufficient data on that topic. What else can I help with?",
            "Let me think... Can you elaborate on your question?",
            "That's beyond my current knowledge base. Ask me something else!",
            "I'm constantly evolving. That query needs more training data.",
        ]
        self.greeting_patterns = [
            r'\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b',
            r'\b(how are you|how\'s it going|what\'s up)\b',
            r'\b(nice to meet you|pleasure to meet you)\b'
        ]
        self.goodbye_patterns = [
            r'\b(bye|goodbye|see you|farewell|exit|quit)\b',
            r'\b(thank you|thanks|thx)\b',
            r'\b(good night|good day)\b'
        ]
        
        # Create default training data if CSV doesn't exist
        self.create_default_training_data()
        self.load_intents()
        self.train_model()

    def create_default_training_data(self):
        """Create a default CSV file if it doesn't exist"""
        if not os.path.exists('chat_log.csv'):
            default_data = [
                {"input": "What is Python?", "response": "Python is a high-level programming language known for its simplicity and readability."},
                {"input": "How do I create a function in Python?", "response": "You can create a function using the 'def' keyword followed by the function name and parameters."},
                {"input": "What is machine learning?", "response": "Machine learning is a subset of AI that enables computers to learn and make decisions from data."},
                {"input": "Tell me about variables", "response": "Variables are containers that store data values. In Python, you don't need to declare variable types."},
                {"input": "What is a loop?", "response": "A loop is a programming construct that repeats a block of code until a condition is met."},
                {"input": "How do I install packages?", "response": "You can install Python packages using pip, for example: pip install package_name"},
                {"input": "What is debugging?", "response": "Debugging is the process of finding and fixing errors or bugs in your code."},
                {"input": "Explain data types", "response": "Python has several data types including integers, floats, strings, lists, dictionaries, and booleans."},
                {"input": "What is an API?", "response": "An API (Application Programming Interface) is a set of protocols for building software applications."},
                {"input": "How do I handle errors?", "response": "You can handle errors in Python using try-except blocks to catch and manage exceptions."}
            ]
            
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as file:
                fieldnames = ['input', 'response']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(default_data)
            print("Created default training data (chat_log.csv)")

    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        if not text or not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            print(f"Tokenization error: {e}")
            # Fallback to simple split if tokenization fails
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                try:
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
                except Exception as e:
                    print(f"Lemmatization error for token '{token}': {e}")
                    processed_tokens.append(token)
        
        return ' '.join(processed_tokens)

    def load_intents(self):
        """Load and preprocess training data"""
        try:
            with open('chat_log.csv', mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if 'input' in row and 'response' in row:
                        processed_input = self.preprocess_text(row["input"])
                        if processed_input:  # Only add non-empty processed inputs
                            self.intents.append(processed_input)
                            self.responses.append(row["response"])
            print(f"Loaded {len(self.intents)} training examples")
        except Exception as e:
            print(f"Error loading training data: {e}")
            # Add some basic fallback data
            self.intents = ["hello", "python programming", "help"]
            self.responses = ["Hello! How can I help?", "I can help with Python!", "I'm here to assist you!"]

    def train_model(self):
        """Train the TF-IDF vectorizer"""
        if self.intents:
            try:
                self.intent_vectors = self.vectorizer.fit_transform(self.intents)
                self.save_model()
                print("Model trained successfully")
            except Exception as e:
                print(f"Training error: {e}")

    def save_model(self):
        """Save trained model to avoid retraining"""
        try:
            with open('chatbot_model.pkl', 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'intent_vectors': self.intent_vectors,
                    'intents': self.intents,
                    'responses': self.responses
                }, f)
        except Exception as e:
            print(f"Could not save model: {e}")

    def load_model(self):
        """Load pre-trained model if available"""
        try:
            if os.path.exists('chatbot_model.pkl'):
                with open('chatbot_model.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.vectorizer = data['vectorizer']
                    self.intent_vectors = data['intent_vectors']
                    self.intents = data['intents']
                    self.responses = data['responses']
                return True
        except Exception as e:
            print(f"Could not load model: {e}")
        return False

    def calculate_similarity(self, user_input, threshold=0.3):
        """Calculate cosine similarity between user input and training data"""
        processed_input = self.preprocess_text(user_input)
        
        if not processed_input or self.intent_vectors is None:
            return None, 0
        
        try:
            # Transform user input using the fitted vectorizer
            user_vector = self.vectorizer.transform([processed_input])
            
            # Calculate similarities
            similarities = cosine_similarity(user_vector, self.intent_vectors)[0]
            
            # Find the best match
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            if best_similarity >= threshold:
                return best_match_idx, best_similarity
        except Exception as e:
            print(f"Similarity calculation error: {e}")
        
        return None, 0

    def handle_patterns(self, user_input):
        """Handle specific patterns like greetings and goodbyes"""
        user_input_lower = user_input.lower()
        
        # Check for greetings
        for pattern in self.greeting_patterns:
            if re.search(pattern, user_input_lower):
                greetings = [
                    "Hello! I'm NEXUS AI. How can I assist you today?",
                    "Greetings, human! What information do you seek?",
                    "Hi there! I'm ready to help with your queries.",
                    "Welcome! My neural networks are at your service.",
                    "Hello! What can I process for you today?"
                ]
                return random.choice(greetings)
        
        # Check for goodbyes
        for pattern in self.goodbye_patterns:
            if re.search(pattern, user_input_lower):
                goodbyes = [
                    "Goodbye! Feel free to return when you need assistance.",
                    "Farewell! It was a pleasure helping you today.",
                    "See you later! I'll be here when you need me.",
                    "Until next time! Stay curious!",
                    "Goodbye! Remember, I'm always learning and improving."
                ]
                return random.choice(goodbyes)
        
        return None

    def get_response(self, user_input):
        """Generate response using NLP techniques"""
        if not user_input or not isinstance(user_input, str):
            return "I didn't receive any input. Please try again."
            
        # First check for specific patterns
        pattern_response = self.handle_patterns(user_input)
        if pattern_response:
            return pattern_response
        
        # Use similarity matching
        best_match_idx, similarity = self.calculate_similarity(user_input)
        
        if best_match_idx is not None and best_match_idx < len(self.responses):
            response = self.responses[best_match_idx]
            
            # Add confidence indicator for high similarity matches
            if similarity > 0.7:
                confidence_phrases = [
                    "I'm confident that: ",
                    "Based on my analysis: ",
                    "With high certainty: ",
                    "My neural networks indicate: "
                ]
                return random.choice(confidence_phrases) + response
            elif similarity > 0.5:
                moderate_phrases = [
                    "I believe: ",
                    "It seems that: ",
                    "Based on available data: ",
                    "My analysis suggests: "
                ]
                return random.choice(moderate_phrases) + response
            else:
                return response
        else:
            # Smart fallback responses based on input analysis
            return self.generate_smart_fallback(user_input)

    def generate_smart_fallback(self, user_input):
        """Generate contextual fallback responses"""
        user_input_lower = user_input.lower()
        
        # Analyze the input for keywords to provide better fallback
        if any(word in user_input_lower for word in ['python', 'code', 'programming', 'function']):
            return "I notice you're asking about programming. While I don't have specific information about that, I can help with general Python concepts. Try asking about basic programming terms!"
        
        elif any(word in user_input_lower for word in ['what', 'how', 'why', 'when', 'where']):
            return "That's a great question! I'm still expanding my knowledge base. Could you try rephrasing or asking about something more specific?"
        
        elif any(word in user_input_lower for word in ['help', 'assist', 'support']):
            return "I'm here to help! You can ask me about Python programming, general technology concepts, or try rephrasing your question differently."
        
        else:
            return random.choice(self.fallback_responses)

# Initialize the chatbot
print("Initializing NEXUS AI Chatbot...")
chatbot = AdvancedChatbot()
print("Chatbot initialized successfully!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chatbot_response():
    user_message = request.args.get('msg')
    if not user_message:
        return "I didn't receive any input. Please try again."
    
    try:
        bot_response = chatbot.get_response(user_message)
        return bot_response
    except Exception as e:
        print(f"Response generation error: {e}")
        return f"Neural network error: {str(e)}. Please try a different query."

@app.route("/train", methods=["POST"])
def retrain_model():
    """Endpoint to retrain the model if needed"""
    try:
        chatbot.load_intents()
        chatbot.train_model()
        return {"status": "success", "message": "Model retrained successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Training failed: {str(e)}"}

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)