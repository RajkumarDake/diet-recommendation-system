# models/food_disease_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# Load dataset
data = pd.read_csv("../datasets/genomic_dataset.csv")  # Columns: food_entity, disease_entity, sentence

# Preprocessing for LSTM
def preprocess_lstm_data(data):
    # Tokenize disease_entity (input) and food_entity (output)
    disease_tokenizer = Tokenizer()
    food_tokenizer = Tokenizer()
    sentence_tokenizer = Tokenizer()

    disease_tokenizer.fit_on_texts(data["disease_entity"])
    food_tokenizer.fit_on_texts(data["food_entity"])
    sentence_tokenizer.fit_on_texts(data["sentence"])

    # Convert text to sequences
    X = disease_tokenizer.texts_to_sequences(data["disease_entity"])
    y_food = food_tokenizer.texts_to_sequences(data["food_entity"])
    y_sentence = sentence_tokenizer.texts_to_sequences(data["sentence"])

    # Pad sequences
    max_len = max(max(len(seq) for seq in X), max(len(seq) for seq in y_food), max(len(seq) for seq in y_sentence))
    X = pad_sequences(X, maxlen=max_len, padding="post")
    y_food = pad_sequences(y_food, maxlen=max_len, padding="post")
    y_sentence = pad_sequences(y_sentence, maxlen=max_len, padding="post")

    return X, y_food, y_sentence, disease_tokenizer, food_tokenizer, sentence_tokenizer, max_len

# Build and train LSTM model
def build_lstm_model(input_dim, output_dim, max_len):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=64, input_length=max_len),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(output_dim, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Preprocessing for Transformer
def preprocess_transformer_data(data):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(
        data["disease_entity"].tolist(),
        padding=True,
        truncation=True,
        max_length=20,
        return_tensors="tf"
    )
    food_labels = tokenizer(
        data["food_entity"].tolist(),
        padding=True,
        truncation=True,
        max_length=20,
        return_tensors="tf"
    )
    sentence_labels = tokenizer(
        data["sentence"].tolist(),
        padding=True,
        truncation=True,
        max_length=50,
        return_tensors="tf"
    )
    return inputs, food_labels, sentence_labels, tokenizer

# Build Transformer model
def build_transformer_model():
    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    input_ids = tf.keras.layers.Input(shape=(20,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(20,), dtype=tf.int32, name="attention_mask")
    
    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]  # Take the last hidden state
    food_output = tf.keras.layers.Dense(30522, activation="softmax", name="food_output")(bert_output)  # BERT vocab size
    sentence_output = tf.keras.layers.Dense(30522, activation="softmax", name="sentence_output")(bert_output)
    
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=[food_output, sentence_output])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Training function
def train_models(data):
    # LSTM
    X, y_food, y_sentence, disease_tok, food_tok, sentence_tok, max_len = preprocess_lstm_data(data)
    X_train, X_test, y_food_train, y_food_test, y_sentence_train, y_sentence_test = train_test_split(
        X, y_food, y_sentence, test_size=0.2, random_state=42
    )
    
    lstm_food_model = build_lstm_model(len(disease_tok.word_index) + 1, len(food_tok.word_index) + 1, max_len)
    lstm_sentence_model = build_lstm_model(len(disease_tok.word_index) + 1, len(sentence_tok.word_index) + 1, max_len)
    
    lstm_food_model.fit(X_train, y_food_train, epochs=10, batch_size=32, validation_data=(X_test, y_food_test))
    lstm_sentence_model.fit(X_train, y_sentence_train, epochs=10, batch_size=32, validation_data=(X_test, y_sentence_test))

    # Transformer
    inputs, food_labels, sentence_labels, tokenizer = preprocess_transformer_data(data)
    train_inputs, test_inputs = train_test_split(inputs, test_size=0.2, random_state=42)
    train_food_labels, test_food_labels = train_test_split(food_labels["input_ids"], test_size=0.2, random_state=42)
    train_sentence_labels, test_sentence_labels = train_test_split(sentence_labels["input_ids"], test_size=0.2, random_state=42)
    
    transformer_model = build_transformer_model()
    transformer_model.fit(
        {"input_ids": train_inputs["input_ids"], "attention_mask": train_inputs["attention_mask"]},
        {"food_output": train_food_labels, "sentence_output": train_sentence_labels},
        epochs=3,
        batch_size=8,
        validation_data=(
            {"input_ids": test_inputs["input_ids"], "attention_mask": test_inputs["attention_mask"]},
            {"food_output": test_food_labels, "sentence_output": test_sentence_labels}
        )
    )
    
    return lstm_food_model, lstm_sentence_model, transformer_model, disease_tok, food_tok, sentence_tok, tokenizer, max_len

# Prediction function
def get_food_to_avoid(disease, lstm_food_model, lstm_sentence_model, transformer_model, disease_tok, food_tok, sentence_tok, tokenizer, max_len):
    # LSTM Prediction
    disease_seq = disease_tok.texts_to_sequences([disease])
    disease_padded = pad_sequences(disease_seq, maxlen=max_len, padding="post")
    
    food_pred = lstm_food_model.predict(disease_padded)
    sentence_pred = lstm_sentence_model.predict(disease_padded)
    
    food_idx = np.argmax(food_pred, axis=-1)[0]
    sentence_idx = np.argmax(sentence_pred, axis=-1)[0]
    
    food = food_tok.sequences_to_texts([food_idx])[0] if food_idx in food_tok.index_word else "None"
    sentence = sentence_tok.sequences_to_texts([sentence_idx])[0] if sentence_idx in sentence_tok.index_word else "No specific foods to avoid for this disease"

    # Transformer Prediction
    transformer_input = tokenizer(disease, padding="max_length", truncation=True, max_length=20, return_tensors="tf")
    transformer_pred = transformer_model.predict({
        "input_ids": transformer_input["input_ids"],
        "attention_mask": transformer_input["attention_mask"]
    })
    
    food_pred_tf = np.argmax(transformer_pred[0], axis=-1)[0]
    sentence_pred_tf = np.argmax(transformer_pred[1], axis=-1)[0]
    
    food_tf = tokenizer.decode(food_pred_tf, skip_special_tokens=True) or "None"
    sentence_tf = tokenizer.decode(sentence_pred_tf, skip_special_tokens=True) or "No specific foods to avoid for this disease"

    # Combine results (using Transformer as primary, fallback to LSTM)
    result = [{
        "food_entity": food_tf if food_tf != "None" else food,
        "sentence": sentence_tf if sentence_tf != "No specific foods to avoid for this disease" else sentence
    }]
    return result

# Train and save models (run once)
if __name__ == "__main__":
    lstm_food_model, lstm_sentence_model, transformer_model, disease_tok, food_tok, sentence_tok, tokenizer, max_len = train_models(data)
    
    # Save models (optional)
    lstm_food_model.save("lstm_food_model.h5")
    lstm_sentence_model.save("lstm_sentence_model.h5")
    transformer_model.save_pretrained("transformer_food_disease_model")
    import pickle
    with open("tokenizers.pkl", "wb") as f:
        pickle.dump({"disease": disease_tok, "food": food_tok, "sentence": sentence_tok, "max_len": max_len}, f)
    tokenizer.save_pretrained("bert_tokenizer")

    # Example usage
    disease = "Diabetes"
    result = get_food_to_avoid(disease, lstm_food_model, lstm_sentence_model, transformer_model, disease_tok, food_tok, sentence_tok, tokenizer, max_len)
    print(result)