import streamlit as st
import librosa as librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory
from pathlib import Path
import tensorflow as tf # Pour le reseau de neurones simple et pour le CNN
import pandas as pd

def predire(chemin_image, chemin_modele):
    
    class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    modele = tf.keras.models.load_model(chemin_modele)
    image_path = Path(chemin_image)
    img = tf.keras.utils.load_img(
        image_path, target_size=(HEIGHT, WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = modele.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    st.write(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    
def main():
    st.title("Classification de Genres Musicaux")
    
    global HEIGHT 
    global WIDTH
    
    HEIGHT = 128
    WIDTH = 660

    uploaded_files = st.file_uploader(
        "Téléchargez un fichier audio au format WAV",
        type=["wav"],
        accept_multiple_files=True
    )
    
    if uploaded_files is not None:
        
        for uploaded_file in uploaded_files:
            
            st.write("Nom du fichier:", uploaded_file.name)
            
            with TemporaryDirectory() as temp_dir:
                temp_file_path = Path(temp_dir, uploaded_file.name)
                temp_file_path.write_bytes(uploaded_file.read())
                
                y, sr = librosa.load(temp_file_path, sr=None)
                
                # Generate log power spectrogram
                S = np.abs(librosa.stft(y))
                spect = librosa.power_to_db(S**2, ref=np.max)
                
                 # Sauvegarder le spectrogramme dans un fichier temporaire
                temp_img_path = Path(temp_dir, "spectrogram.png")
                plt.imsave(temp_img_path, spect, cmap='inferno')
                # Vérifier si le fichier temporaire existe
                if temp_img_path.exists():
                    img = tf.keras.utils.load_img(
                        temp_img_path, target_size=(HEIGHT, WIDTH) 
                    )
                    col1, col2, col3 = st.columns(3)
                    with col1 :
                        st.audio(uploaded_file)
                    with col2:
                        # Afficher le spectrogramme
                        fig = plt.figure(figsize=(12,2))
                        librosa.display.specshow(spect)
                        # Afficher la figure avec st.pyplot()
                        st.pyplot(fig)
                    with col3:
                        st.write("Résultats")
                        if st.button('Predict'):
                            # img_path = temp_file_path + uploaded_file.name
                            model_path = '..\model\modele-v1.h5'
                            results = predire(temp_img_path,model_path)
                            st.write(results)

# Load modèle 
# Saved_img.resize(128, 660)
# Predict() for each Saved_img
# Display predicted class name + confidence rate

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Audio Classifier")
    print('>=========Page Loaded==========<')
    main()
