import streamlit as st
import librosa as librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory
from pathlib import Path
import tensorflow as tf # Pour le reseau de neurones simple et pour le CNN

def predire(img_array, chemin_modele):
    modele = tf.keras.models.load_model(chemin_modele)
    # image_path = Path(chemin_image)
    # img = tf.keras.utils.load_img(
    #     image_path, target_size=(HEIGHT, WIDTH)
    # )
    # img_array = tf.keras.utils.img_to_array(img)
    
    st.write("Img array shape",img_array.shape)
    img_array = tf.expand_dims(img_array, 0)
    
    # Redimensionner en (128, 660) en utilisant TensorFlow/Keras
    new_shape = (128, 660)
    resized_spect = tf.image.resize(img_array, new_shape, method=tf.image.ResizeMethod.BILINEAR)
    st.write("Resized shape",resized_spect.shape)
    
    # Répéter le tableau redimensionné sur trois canaux pour créer une image RGB
    rgb_spect = np.repeat(resized_spect, 3, axis=-1)
    st.write("RGB array shape",rgb_spect.shape)

    # predictions = modele.predict(rgb_spect)
    # score = tf.nn.softmax(predictions[0])
    # print("score =",100 * np.max(score))
#   print(
#       "This image most likely belongs to {} with a {:.2f} percent confidence."
#       .format(class_names[np.argmax(score)], 100 * np.max(score))
#   )

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
                
                # # Slice to have same shapes for all
                # spect = spect[:1000,:1220]

                # # Compress for the model
                # new_shape = (100,122)
                # block_height = spect.shape[0] // new_shape[0]
                # block_width = spect.shape[1] // new_shape[1]
                # downsampled_array = spect.reshape(new_shape[0], block_height, new_shape[1], block_width).mean(3).mean(1)
                # X = np.array([downsampled_array])
                
            col1, col2, col3 = st.columns(3)
        
            with col1 :
                st.audio(uploaded_file)
                
            with col2:
                # # Afficher le spectrogramme
                fig = plt.figure(figsize=(12,2))
                librosa.display.specshow(spect)
                # Afficher la figure avec st.pyplot()
                st.pyplot(fig)
                
            with col3:
                st.write("Résultats")
                if st.button('Predict'):
                    # img_path = temp_file_path + uploaded_file.name
                    model_path = '..\model\modele-v1.h5'
                    results = predire(spect,model_path)
                    # st.write(results)

# Load modèle 
# Saved_img.resize(128, 660)
# Predict() for each Saved_img
# Display predicted class name + confidence rate

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Audio Classifier")
    print('>=========Page Loaded==========<')
    main()
