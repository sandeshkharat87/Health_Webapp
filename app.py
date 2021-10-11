import streamlit as st
from PIL import Image
import numpy as np
from Utilss import Select
from tensorflow.keras import models
import tensorflow as tf

# NM_model = tf.keras.models.load_model('models/CoronaCT_V1')
# Malaria_model = tf.keras.models.load_model('models/MLRA_V1.h5')


Pred_cls = ["Pneumonia", "Malaria", "Brain Tumor"]


p_cls = st.selectbox('Choose', Pred_cls)


@st.cache(allow_output_mutation=True)
def load_model_BRAIN_TUMOR():
    model = tf.keras.models.load_model('models/brain_Tumor_V2.h5')

    return model


@st.cache(allow_output_mutation=True)
def load_model_CORONA():
    model = tf.keras.models.load_model('models/CoronaCT_V1.h5')

    return model


@st.cache(allow_output_mutation=True)
def load_model_MLRA():
    model = tf.keras.models.load_model('models/MLRA_V1.h5')

    return model


model_BT = load_model_BRAIN_TUMOR()
model_CORONA = load_model_CORONA()
model_MLRA = load_model_MLRA()


Pred_cls = ["Pneumonia", "Malaria", "Brain Tumor"]


def ChooseModel(value):
    if value == 'Pneumonia':
        return model_CORONA

    elif value == 'Malaria':
        return model_MLRA

    else:
        return model_BT


mainModel = ChooseModel(p_cls)

img_config = Select(p_cls)


st.write(f" Prediction of {p_cls} ")

# st.set_option('deprecation.showfileUploaderEncoding', False)
IMG_FILE = st.file_uploader("Please Upload Image here....", type=['jpg', 'jpeg', 'png'])


if IMG_FILE:

    img = Image.open(IMG_FILE)

    img = np.array(img.convert("RGB").resize(img_config['size']))
    img = img / 255.0
    img = np.expand_dims(img, 0)

    pred = mainModel.predict(img)

    result = np.argmax(pred)
    score = round(np.max(pred) * 100, 2)

    result2 = img_config['cls_names'][result]

    mystr = f" {result2}  {score}% "
    st.success(mystr)
    st.image(IMG_FILE, use_column_width=True, caption=mystr)
