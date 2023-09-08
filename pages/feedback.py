import streamlit as st
import sqlite3

class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Établir une connexion à la base de données SQLite
conn = sqlite3.connect("audio_db.db")

# Créer un objet curseur
cursor = conn.cursor()

# Exécuter une requête SQL pour sélectionner toutes les lignes de la table "ma_table"
cursor.execute("SELECT * FROM audio_files")

# Récupérer toutes les lignes
rows = cursor.fetchall()

print("nb elem de un elem=",len(rows[0]))

# Fermer la connexion à la base de données
conn.close()

def send_feedback(id_audio,status,genre) :
    conn = sqlite3.connect("audio_db.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE audio_files SET feedback=? WHERE ID=? ", (genre, id_audio))
    conn.commit()
    conn.close()
    print('Sended')

def main() :
    st.title('Feedback sur les classifications')
    for e in rows:
        id_audio=e[0]
        filename=e[1]
        audio_data=e[2]
        model_pred=e[3]
        pred_confidence=e[4]
        st.write(f'ID = {id_audio}')
        st.write('filename =',filename)
        st.audio(audio_data)
        st.write('model_pred =',model_pred)
        st.write(f'pred_confidence = {pred_confidence:.2f} %')
        vrai_genre = st.selectbox("Selectionnez le vrai genre",class_names, key=f'select_{id_audio}_{filename}')
        if st.button('False_prediction', key=f'{id_audio}_{filename}'):
            send_feedback(id_audio, False, vrai_genre)
        st.write('---') 
if __name__ == "__main__":
    main()