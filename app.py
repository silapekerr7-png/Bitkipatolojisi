import pickle
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title='Elma Yaprağı Hastalık Tahmini', page_icon='🍎')
st.title('🍎 Elma Yaprağı Hastalık Tahmini')
st.write('Bir elma yaprağı fotoğrafı yükleyin, model hastalık sınıfını tahmin etsin.')

CLASSES = ['Sağlıklı', 'Birden fazla hastalık', 'Pas (rust)', 'Kabuklanma (scab)']

@st.cache_resource
def load_models():
    clf = pickle.load(open('model.pkl', 'rb'))
    ext = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    return clf, ext

clf, extractor = load_models()

file = st.file_uploader('Yaprak fotoğrafı', type=['jpg', 'jpeg', 'png'])
if file:
    img = Image.open(file).convert('RGB')
    st.image(img, caption='Yüklenen görüntü', use_container_width=True)
    x = preprocess_input(np.array(img.resize((224, 224)), dtype=np.float32)[None])
    p = clf.predict_proba(extractor.predict(x, verbose=0))[0]
    st.success(f'Tahmin: **{CLASSES[int(p.argmax())]}**')
    st.bar_chart(pd.DataFrame({'Olasılık': p}, index=CLASSES))
