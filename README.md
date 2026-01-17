# Assistant Service Public & Reconnaissance de Chiffres (MNIST)
Ce projet a été réalisé dans le cadre de ma formation à l'*ENSA Berrechid*. Il se compose de deux parties majeures utilisant le Deep Learning :
  1.  *Un Chatbot Intelligent* : Assistant administratif pour les démarches en mairie.
  2.  *Reconnaissance de Chiffres (MNIST)* : Une interface de dessin pour tester un réseau de neurones convolutif (CNN).
## Table des matières
1. [Partie 1](#partie-1)
2. [Partie 2](#partie-1)
3. [Technologies](#technologies)
### Partie 1
***
Le chatbot utilise un modèle de réseau de neurones Dense avec traitement de langage naturel (NLP).

Fichiers : app2.py, intents.json

Fonctionnement : L'entraînement est intégré directement dans l'application. Le modèle est mis en cache pour plus de fluidité.

Lancement :
streamlit run app2.py
### Partie 2
***
Un modèle CNN (Convolutional Neural Network) capable de prédire un chiffre dessiné à la main.

Fichiers : train.py, app.py

Important : Vous devez d'abord entraîner le modèle avant de lancer l'interface.

Étape 1 (Entraînement) :
* python train.py
* (Cela générera un fichier best_model.h5)
  
Étape 2 (Interface) :
* streamlit run app.py
### Technologies :
***
* Deep Learning : Keras / TensorFlow

* NLP : NLTK (Tokenization, Lemmatization)

* Vision : OpenCV (Prétraitement d'image)

* Interface : Streamlit

