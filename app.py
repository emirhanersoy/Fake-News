import streamlit as st
import pickle
import os

# Dosya yollarını kontrol etme
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')
vectorizer_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')

# Kaydedilen modeli ve vektörleştiriciyi yükleme
with open('model.pkl', 'rb') as model_file:
    pac = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Streamlit başlığı
st.title("Fake News Detection")

# Kullanıcıdan haber metni girişi
user_input = st.text_area("Haberi girin:")

if st.button("Tahmin Et"):
    if user_input:
        # Kullanıcı girdisini vektörize etme
        news_tfidf = tfidf_vectorizer.transform([user_input])
        
        # Tahmin yapma
        prediction = pac.predict(news_tfidf)
        
        # Sonucu gösterme
        st.write(f"Bu haber: **{prediction[0]}**")
    else:
        st.write("Lütfen bir haber metni girin.")
