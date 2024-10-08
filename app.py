import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

@st.cache_resource
def carrega_modelo():
    """
        Busca modelo no google drive e carrega interpretador com tensorflow
    """

    #https://drive.google.com/file/d/1j0vKzXC6oaLyoeSmv0BiGyqrkGWNNFt5/view?usp=drive_link
    #url = 'https://drive.google.com/uc?id=1j0vKzXC6oaLyoeSmv0BiGyqrkGWNNFt5'

    url = transformar_link_drive('https://drive.google.com/file/d/1j0vKzXC6oaLyoeSmv0BiGyqrkGWNNFt5/view?usp=drive_link')

    gdown.download(url, 'modelo_quantizado16bits.tflite')
    interpreter = tf.lite.Interpreter(model_path="modelo_quantizado16bits.tflite")
    interpreter.allocate_tensors()

    return interpreter

def carrega_imagem():
    """
        Carrega imagem passada pelo o usuário e prepara imagem para o processamento

        Returns:
         image: NDArray[floating[Any]] -> Imagem processada com numpy
    """

    uploaded_file = st.file_uploader("Arraste e solte uma imagem aqui ou clique para selecionar uma", 
                                     type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success('Image foi carregada com sucesso')

        image = np.array(image, dtype=np.float32)
        image = image / 255.
        image = np.expand_dims(image, axis=0)

        return image

def transformar_link_drive(link_original):
    """
    Transforma um link do Google Drive do formato 'view' para o formato 'uc'.

    Args:
        link_original: O link do Google Drive no formato 'view'.

    Returns:
        O link do Google Drive no formato 'uc', ou None se o link não for válido.
    """
    try:
        id_arquivo = link_original.split('/d/')[1].split('/')[0]
        novo_link = f'https://drive.google.com/uc?id={id_arquivo}'
        return novo_link
    except IndexError:
        return None

def previsao(interpreter, image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['BlackMeasles', 'BlackRot', 'HealthyGrapes', 'LeafBlight']

    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100*output_data[0]

    fig = px.bar(df,y='classes',x='probabilidades (%)',  orientation='h', text='probabilidades (%)', title='Probabilidade de Classes de Doenças em Uvas')
    st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title="Classifica Folhas de Videira",
        page_icon="🍇",
    )

    st.write("# Classifica Folhas de Videira! 🍇")
    #Carrega modelo
    interpreter = carrega_modelo()
    #Carrega imagem
    image = carrega_imagem()
    #Classifica
    if image is not None:
        previsao(interpreter, image)




if __name__ == "__main__":
    main()