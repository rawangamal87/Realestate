import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer



model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

st.title("تصنيف الرسائل العقارية")
text_input = st.text_area("يرجى كتابة الرسالة هنا")

if st.button("تنبؤ"):
    seq = tokenizer.texts_to_sequences([text_input])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    prediction = model.predict(padded)
    label = le.inverse_transform([np.argmax(prediction)])
    st.success(f"التصنيف المتوقع: {label[0]}")
