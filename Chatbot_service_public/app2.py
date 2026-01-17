import streamlit as st
import json
import string
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras import Sequential
from keras.layers import Dense, Dropout
import keras

# Configuration de la page
st.set_page_config(page_title="Assistant Service Public", page_icon="🏛️")

# --- PARTIE 1 : ENTRAINEMENT & CHARGEMENT (Mis en cache) ---
# On utilise le cache pour ne pas ré-entraîner le modèle à chaque clic
@st.cache_resource
def train_and_load_model():
    # Downloads nécessaires
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")
        nltk.download("wordnet")

    # Load intents
    with open('intents.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    lemmatizer = WordNetLemmatizer()
    words = []
    classes = []
    data_x = []
    data_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            data_x.append(pattern)
            data_y.append(intent["tag"])
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
    words = sorted(set(words))
    classes = sorted(set(classes))

    training = []
    out_empty = [0] * len(classes)

    for idx, doc in enumerate(data_x):
        bow = []
        text = nltk.word_tokenize(doc.lower())
        text = [lemmatizer.lemmatize(word) for word in text]
        for word in words:
            bow.append(1 if word in text else 0)
        
        output_row = list(out_empty)
        output_row[classes.index(data_y[idx])] = 1
        training.append([bow, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)
    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    # Build model
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation="softmax"))

    adam = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
    
    # Entraînement rapide
    model.fit(x=train_x, y=train_y, epochs=150, verbose=0)
    
    return model, words, classes, data, lemmatizer

# Chargement du modèle et des données
model, words, classes, data, lemmatizer = train_and_load_model()

# --- PARTIE 2 : FONCTIONS DE PRÉDICTION ---

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]), verbose=0)[0]
    thresh = 0.5
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return [labels[r[0]] for r in y_pred]

def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "Désolé, je n'ai pas compris votre demande. Pourriez-vous reformuler ?"
    
    tag = intents_list[0]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            # Gestion spéciale pour la date et l'heure
            if tag == "time":
                from datetime import datetime
                return random.choice(intent["responses"]) + " " + datetime.now().strftime("%H:%M")
            if tag == "date":
                from datetime import date
                return random.choice(intent["responses"]) + " " + str(date.today())
            
            return random.choice(intent["responses"])
    return "Désolé, je ne comprends pas."

# --- PARTIE 3 : INTERFACE GRAPHIQUE STREAMLIT ---

st.title("🏛️ Assistant Service Public")
st.markdown("Bienvenue. Posez vos questions sur les démarches administratives (Passeport, Mairie, Horaires...).")

# Initialiser l'historique du chat si non existant
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input("Votre question (ex: horaires mairie)..."):
    # 1. Afficher le message de l'utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)
    # Sauvegarder dans l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Calculer la réponse
    intents = pred_class(prompt, words, classes)
    response = get_response(intents, data)

    # 3. Afficher la réponse du bot
    with st.chat_message("assistant"):
        st.markdown(response)
    # Sauvegarder dans l'historique
    st.session_state.messages.append({"role": "assistant", "content": response})
