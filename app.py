import io
from pathlib import Path
import streamlit as st
from matplotlib.colors import to_hex
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def get(imagem_carregada, n_cores):
    # salvar a imagem do streamlit
    with open('imagem.jpg', 'wb') as file:
        file.write(imagem_carregada.getbuffer())
    # ler a imagem
    image = Image.open('imagem.jpg')
    # transformar os pixels em linhas de uma matriz
    N, M = image.size
    X = np.asarray(image).reshape((M*N, 3))
    # criar e aplicar o k-means na imagem
    model = KMeans(n_clusters=n_cores, random_state=42).fit(X)
    # capturar os centros (cores médias dos grupos)
    cores = model.cluster_centers_.astype('uint8')[np.newaxis]
    cores_hex = [to_hex(cor/255) for cor in cores[0]]

    # apagar imagem salva
    Path('imagem.jpg').unlink()
    # retornar cores
    return cores, cores_hex

def show(cores):
    fig = plt.figure()
    plt.imshow(cores)
    plt.axis('off')
    return fig

def save(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    plt.axis('off')
    return img

st.title("Gerador de paletas")
imagem = st.file_uploader("Envie sua imagem", ["jpg", "jpeg"])

col1, col2 = st.columns([.7, .3])

if imagem:
    col1.image(imagem)
    n_cores = col2.slider(
        "Quantidade de cores",
        min_value=2,
        max_value=8,
        value=5
    )
    botao_gerar_paleta = col2.button("Gerar paleta!")
    if botao_gerar_paleta:
        cores, cores_hex = get(imagem, n_cores)
        figura = show(cores)
        col2.pyplot(fig=figura)
        col2.code(f"{cores_hex}")
        
        col2.download_button(
            "Download",
            save(figura),
            "paleta.png",
            'image/png'
        )