import string
import spacy

# Emotion mapping matching notebook
EMOTION_ID_TO_LABEL = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

EMOTION_LABEL_TO_ID = {v: k for k, v in EMOTION_ID_TO_LABEL.items()}

EMOTION_CONFIG = {
    "joy": {
        "id": 5,
        "emoji": "😊",
        "color": "#2ED573",
        "gradient": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
        "description": "Feeling of happiness, pleasure, or contentment."
    },
    "sadness": {
        "id": 0,
        "emoji": "😢",
        "color": "#70A1FF",
        "gradient": "linear-gradient(135deg, #3a7bd5 0%, #3a6073 100%)",
        "description": "Feeling of sorrow, unhappiness, or despair."
    },
    "anger": {
        "id": 1,
        "emoji": "😠",
        "color": "#FF4757",
        "gradient": "linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%)",
        "description": "Feeling of strong displeasure, annoyance, or hostility."
    },
    "fear": {
        "id": 4,
        "emoji": "😨",
        "color": "#FFA502",
        "gradient": "linear-gradient(135deg, #ff9966 0%, #ff5e62 100%)",
        "description": "Feeling of anxiety, threat, or apprehension."
    },
    "love": {
        "id": 2,
        "emoji": "❤️",
        "color": "#FF6B81",
        "gradient": "linear-gradient(135deg, #ff758c 0%, #ff7eb3 100%)",
        "description": "Feeling of deep affection, fondness, or tenderness."
    },
    "surprise": {
        "id": 3,
        "emoji": "😲",
        "color": "#1E90FF",
        "gradient": "linear-gradient(135deg, #4ea5d9 0%, #4460a0 100%)",
        "description": "Feeling of astonishment, wonder, or unexpected amazement."
    }
}

# Initialize SpaCy blank english model as per main.ipynb
try:
    nlp = spacy.blank("en")
    for word in ["not", "no", "never"]:
        nlp.vocab[word].is_stop = False
except Exception:
    nlp = None

def remove_punctuation(text: str) -> str:
    """Removes standard punctuation marks from text."""
    if not isinstance(text, str):
        return ""
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text: str) -> str:
    """Removes numeric digits from text."""
    if not isinstance(text, str):
        return ""
    return "".join([ch for ch in text if not ch.isdigit()])

def remove_emojis(text: str) -> str:
    """Removes non-ASCII characters / emojis."""
    if not isinstance(text, str):
        return ""
    return "".join([ch for ch in text if ch.isascii()])

def remove_stopwords(text: str) -> str:
    """Removes stopwords using spaCy or fallback basic stopword list, keeping negations."""
    if not text:
        return ""
    
    if nlp is not None:
        doc = nlp(text)
        cleaned = [token.text for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(cleaned)
    else:
        # Basic fallback list if spacy is unavailable
        basic_stops = {"i", "me", "my", "myself", "we", "our", "ours", "you", "your", "yours", 
                       "he", "him", "his", "she", "her", "it", "its", "they", "them", "what", 
                       "which", "who", "whom", "this", "that", "these", "those", "am", "is", 
                       "are", "was", "were", "be", "been", "being", "have", "has", "had", "do"}
        tokens = text.split()
        cleaned = [tok for tok in tokens if tok.lower() not in basic_stops]
        return " ".join(cleaned)

def preprocess_text(text: str) -> str:
    """
    Applies the full NLP preprocessing pipeline matching main.ipynb:
    1. Lowercasing
    2. Punctuation removal
    3. Number removal
    4. Emoji/Non-ASCII removal
    5. Stopwords removal (keeping negations: not, no, never)
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    txt = text.lower()
    txt = remove_punctuation(txt)
    txt = remove_numbers(txt)
    txt = remove_emojis(txt)
    txt = remove_stopwords(txt)
    return txt
