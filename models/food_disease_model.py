# models/food_disease_model.py
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFBertModel
import requests
import pickle
import logging

# Set up basic logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pre-trained models and tokenizers
logger.info("Loading pre-trained LSTM and Transformer models...")
lstm_food_model = load_model("lstm_food_model.h5")
lstm_sentence_model = load_model("lstm_sentence_model.h5")
transformer_model = TFBertModel.from_pretrained("transformer_food_disease_model")

# Load tokenizers from saved file
logger.info("Loading tokenizers from pickle file...")
with open("tokenizers.pkl", "rb") as f:
    tokenizers = pickle.load(f)
    disease_tokenizer = tokenizers["disease"]
    food_tokenizer = tokenizers["food"]
    sentence_tokenizer = tokenizers["sentence"]
    maximum_length = tokenizers["max_len"]

# Load BERT tokenizer
logger.info("Loading BERT tokenizer...")
bert_tokenizer = BertTokenizer.from_pretrained("bert_tokenizer")

# Prediction function with detailed steps
def get_food_to_avoid(disease_input):
    """Predict food to avoid based on disease input using LSTM and Transformer models."""
    logger.info(f"Processing disease input: {disease_input}")
    
    # LSTM Prediction
    disease_sequence = disease_tokenizer.texts_to_sequences([disease_input])
    padded_disease = pad_sequences(disease_sequence, maxlen=maximum_length, padding="post")
    
    food_prediction = lstm_food_model.predict(padded_disease, verbose=0)
    sentence_prediction = lstm_sentence_model.predict(padded_disease, verbose=0)
    
    food_index = np.argmax(food_prediction, axis=-1)[0]
    sentence_index = np.argmax(sentence_prediction, axis=-1)[0]
    
    predicted_food = food_tokenizer.sequences_to_texts([food_index])[0] if food_index in food_tokenizer.index_word else "None"
    predicted_sentence = sentence_tokenizer.sequences_to_texts([sentence_index])[0] if sentence_index in sentence_tokenizer.index_word else "No specific foods to avoid"

    # Transformer Prediction
    transformer_input = bert_tokenizer(disease_input, padding="max_length", truncation=True, max_length=20, return_tensors="tf")
    transformer_prediction = transformer_model.predict({
        "input_ids": transformer_input["input_ids"],
        "attention_mask": transformer_input["attention_mask"]
    }, verbose=0)
    
    transformer_food_pred = np.argmax(transformer_prediction[0], axis=-1)[0]
    transformer_sentence_pred = np.argmax(transformer_prediction[1], axis=-1)[0]
    
    transformer_food = bert_tokenizer.decode(transformer_food_pred, skip_special_tokens=True) or "None"
    transformer_sentence = bert_tokenizer.decode(transformer_sentence_pred, skip_special_tokens=True) or "No specific foods to avoid"

    # Combine results, prioritizing Transformer output
    final_result = [{
        "food_entity": transformer_food if transformer_food != "None" else predicted_food,
        "sentence": transformer_sentence if transformer_sentence != "No specific foods to avoid" else predicted_sentence
    }]
    logger.info(f"Prediction result: {final_result}")
    return final_result

# Function to send result to http://localhost:5000/food
def send_to_food_endpoint(disease):
    """Send the prediction result to the specified endpoint."""
    try:
        prediction_result = get_food_to_avoid(disease)
        logger.info(f"Sending result to http://localhost:5000/food: {prediction_result}")
        
        # Make POST request to the endpoint
        response = requests.post(
            "http://localhost:5000/food",
            json=prediction_result,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            logger.info("Successfully sent result to endpoint")
        else:
            logger.error(f"Failed to send result. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending result to endpoint: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

# Example execution
if __name__ == "__main__":
    test_disease = "Diabetes"
    logger.info("Starting prediction process...")
    send_to_food_endpoint(test_disease)