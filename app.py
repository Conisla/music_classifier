import streamlit as st
import sqlite3
import librosa
import librosa.display
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def register_new_data(data, uploaded_file_name):
    # Connexion à la base de données SQLite (créez-la si elle n'existe pas)
    conn = sqlite3.connect("audio_db.db")
    cursor = conn.cursor()

    # Créer une table pour stocker les données audio
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            audio_data BLOB,
            predicted_genre TEXT DEFAULT NULL,
            score REAL DEFAULT NULL,
            feedback TEXT DEFAULT NULL
        )
    ''')

    # Insérez le fichier audio dans la base de données
    cursor.execute(
      "INSERT INTO audio_files (filename,audio_data) VALUES (?,?)",
      (uploaded_file_name, sqlite3.Binary(data),)
    )
    conn.commit()

    # Fermez la connexion à la base de données
    conn.close() 

def get_audiodata_from_db():
    # Connexion à la base de données SQLite
    conn = sqlite3.connect("audio_db.db")
    cursor = conn.cursor()

    # Sélectionnez la dernière ligne de la table en triant par l'ID (ou une autre colonne chronologique)
    cursor.execute("SELECT * FROM audio_files ORDER BY ID DESC LIMIT 1")

    audio_data = cursor.fetchone()
    # Fermez la connexion à la base de données
    conn.close()
    
    return audio_data

def format_element(element):
  return "%.2f" % (100 * element)

def get_resized_spectrogram(audio_data, spectrogram_height):
  y, sr = librosa.load(BytesIO(bytes(audio_data)))
  # power_spectrogram = lb.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000) # amplitude squared
  power_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024) # amplitude squared
  db_spectrogram = librosa.power_to_db(power_spectrogram, ref=np.max) # decibel units
  if db_spectrogram.shape[1] != spectrogram_height:
      db_spectrogram.resize(128, spectrogram_height, refcheck=False)

  return db_spectrogram

def predict(id_audio,audio_data, model_path, genres):
  model = tf.keras.models.load_model(model_path)
  spectrogram = get_resized_spectrogram(audio_data, 660)
  np_spect_array = np.array(spectrogram)
  np_spect_array /= np.min(np_spect_array)
  np_spect_array = tf.expand_dims(np_spect_array, 0) # Create a batch
  predictions = model.predict(np_spect_array)
  score = tf.nn.softmax(predictions[0])
  #formatted_score = list(map(format_element, score))
  
  predicted_genre = genres[np.argmax(score)]
  max_score = 100 * np.max(score)

  # Update the row with predicted_genre and score
  conn = sqlite3.connect("audio_db.db")
  cursor = conn.cursor()
  cursor.execute("UPDATE audio_files SET predicted_genre=?, score=? WHERE ID=?", (predicted_genre, max_score, id_audio))
  conn.commit()
  conn.close()
  
  message = "This song is most likely a  {}  song with a {:.2f} percent confidence.".format(predicted_genre, max_score)

  return message, genres[np.argmax(score)], np.max(score)

def main():
    # Créez une interface Streamlit
    st.title("Enregistrement d'un fichier audio dans une base de données SQLite")

    # Ajoutez un bouton pour télécharger le fichier audio
    uploaded_file = st.file_uploader("Téléchargez un fichier audio au format MP3, WAV, etc.", type=["mp3", "wav"])

    # Vérifiez si un fichier a été téléchargé
    if uploaded_file is not None and st.button('Register into db'):
        # Lisez le contenu du fichier audio
        audio_data = uploaded_file.read()
        register_new_data(audio_data,uploaded_file.name)
        # Affichez un message de confirmation
        st.success("Fichier audio sauvegardé avec succès dans la base de données SQLite.")
        
    st.write('---')
    
    if  uploaded_file is not None and st.button("Predict"):
        audio_db = get_audiodata_from_db()
        st.write("filename =",audio_db[1])
        class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        st.write("Genre musicale de l'audio")
        model_path = '.\\model\\best_model_2.h5'
        resultats = predict(audio_db[0],audio_db[2],model_path,class_names)
        st.write(resultats[0])

if __name__ == "__main__":
    main()
