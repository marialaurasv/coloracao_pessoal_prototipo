import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import os

from segmentacao import segmentar_pele_mtcnn
from features import extrair_features_lab, extrair_features_rosto
from mtcnn import MTCNN

# Mapa e paletas para Modo Pele (igual ao que você tinha)
mapa_clusters_pele = {
    0: "Primavera Brilhante",
    1: "Inverno Profundo",
    2: "Verão Claro",
    3: "Primavera Clara",
    4: "Inverno Brilhante",
    5: "Verão Suave",
    6: "Primavera Quente",
    7: "Outono Suave",
    8: "Verão Frio",
    9: "Outono Quente",
    10: "Inverno Frio",
    11: "Outono Profundo"
}

paletas_pele = {
    "Primavera Brilhante": ["#FF6F61", "#FFFACD", "#98FF98", "#40E0D0", "#FFFDD0"],
    "Primavera Clara": ["#FFDAB9", "#FFB6C1", "#E6E6FA", "#87CEEB", "#F5F5DC"],
    "Primavera Quente": ["#FF7F50", "#FFD700", "#FF6347", "#9ACD32", "#FFF8DC"],
    "Verão Claro": ["#B0E0E6", "#FFB6C1", "#E6E6FA", "#98FB98", "#D3D3D3"],
    "Verão Suave": ["#6699CC", "#D8BFD8", "#B2AC88", "#5F9EA0", "#A9A9A9"],
    "Verão Frio": ["#6495ED", "#DE3163", "#9370DB", "#000080", "#708090"],
    "Outono Suave": ["#D2691E", "#F5DEB3", "#8A9A5B", "#8B7355", "#FFE5B4"],
    "Outono Quente": ["#FF8C00", "#FFDB58", "#808000", "#8B4513", "#F5F5DC"],
    "Outono Profundo": ["#B22222", "#FFD700", "#228B22", "#654321", "#F5DEB3"],
    "Inverno Frio": ["#4169E1", "#FF00FF", "#8A2BE2", "#000080", "#2F4F4F"],
    "Inverno Profundo": ["#800020", "#C71585", "#50C878", "#000000", "#36454F"],
    "Inverno Brilhante": ["#FF0000", "#FF69B4", "#0000FF", "#8B00FF", "#FFFFFF"]
}

# Mapa e paletas para Modo Rosto - por enquanto usa os mesmos valores que Modo Pele
mapa_clusters_rosto = {
    0: "Outono Quente",
    1: "Verão Suave",
    2: "Inverno Profundo",
    3: "Primavera Quente",
    4: "Verão Claro",
    5: "Primavera Clara",
    6: "Primavera Brilhante",
    7: "Inverno Brilhante",
    8: "Verão Frio",
    9: "Outono Profundo",
    10: "Inverno Frio",
    11: "Outono Suave"
}
paletas_rosto = {
    "Primavera Brilhante": ["#FF6F61", "#FFFACD", "#98FF98", "#40E0D0", "#FFFDD0"],
    "Primavera Clara": ["#FFDAB9", "#FFB6C1", "#E6E6FA", "#87CEEB", "#F5F5DC"],
    "Primavera Quente": ["#FF7F50", "#FFD700", "#FF6347", "#9ACD32", "#FFF8DC"],
    "Verão Claro": ["#B0E0E6", "#FFB6C1", "#E6E6FA", "#98FB98", "#D3D3D3"],
    "Verão Suave": ["#6699CC", "#D8BFD8", "#B2AC88", "#5F9EA0", "#A9A9A9"],
    "Verão Frio": ["#6495ED", "#DE3163", "#9370DB", "#000080", "#708090"],
    "Outono Suave": ["#D2691E", "#F5DEB3", "#8A9A5B", "#8B7355", "#FFE5B4"],
    "Outono Quente": ["#FF8C00", "#FFDB58", "#808000", "#8B4513", "#F5F5DC"],
    "Outono Profundo": ["#B22222", "#FFD700", "#228B22", "#654321", "#F5DEB3"],
    "Inverno Frio": ["#4169E1", "#FF00FF", "#8A2BE2", "#000080", "#2F4F4F"],
    "Inverno Profundo": ["#800020", "#C71585", "#50C878", "#000000", "#36454F"],
    "Inverno Brilhante": ["#FF0000", "#FF69B4", "#0000FF", "#8B00FF", "#FFFFFF"]
}

# Diretório onde os modelos estão salvos
PASTA_MODELOS = "modelos_salvos"

# Carrega modelos para Modo Pele
scaler_pele = joblib.load(os.path.join(PASTA_MODELOS, "scaler_pele.pkl"))
umap_pele = joblib.load(os.path.join(PASTA_MODELOS, "umap_pele.pkl"))
kmeans_pele = joblib.load(os.path.join(PASTA_MODELOS, "kmeans_pele.pkl"))

# Carrega modelos para Modo Rosto
scaler_rosto = joblib.load(os.path.join(PASTA_MODELOS, "scaler_rosto.pkl"))
umap_rosto = joblib.load(os.path.join(PASTA_MODELOS, "umap_rosto.pkl"))
kmeans_rosto = joblib.load(os.path.join(PASTA_MODELOS, "kmeans_rosto.pkl"))

# Inicializa detector MTCNN para modo rosto
detector = MTCNN()

st.set_page_config(page_title="Classificador de Coloração Pessoal", layout="centered")
st.title("🎨 Classificador de Cartela de Coloração Pessoal")

modo = st.selectbox("Escolha o modo de análise:", ["Modo Pele", "Modo Rosto"])

uploaded_file = st.file_uploader("Envie uma imagem do rosto (frontal, iluminada)", type=["jpg", "jpeg", "png"])

def cortar_rosto_mtcnn(imagem_bgr):
    resultado = detector.detect_faces(cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB))
    if not resultado:
        return None
    face = resultado[0]
    x, y, largura, altura = face['box']
    x, y = max(0, x), max(0, y)
    recorte = imagem_bgr[y:y+altura, x:x+largura]
    return recorte

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    max_width = 400
    width_percent = max_width / float(image.size[0])
    new_height = int((float(image.size[1]) * float(width_percent)))

    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        resample_method = Image.ANTIALIAS

    resized_image = image.resize((max_width, new_height), resample_method)

    import io
    buffered = io.BytesIO()
    resized_image.save(buffered, format="PNG")
    img_str = buffered.getvalue()

    import base64
    img_b64 = base64.b64encode(img_str).decode()

    html_code = f'''
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_b64}" width="{max_width}" height="{new_height}" />
        <p style="text-align: center;">Imagem enviada</p>
    </div>
    '''

    st.markdown(html_code, unsafe_allow_html=True)

    imagem_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if modo == "Modo Pele":
        segmentada = segmentar_pele_mtcnn(imagem_np)
        if segmentada is None:
            st.error("Não foi possível detectar a pele na imagem.")
            st.stop()
        imagem_para_features = segmentada
        features = extrair_features_lab(imagem_para_features)
        features = features.reshape(1, -1)
        features_pad = scaler_pele.transform(features)
        reduzido = umap_pele.transform(features_pad)
        cluster = kmeans_pele.predict(reduzido)[0]

        cartela_nome = mapa_clusters_pele.get(cluster, "Cartela Desconhecida")
        cores = paletas_pele.get(cartela_nome, [])
    else:  # Modo Rosto
        rosto = cortar_rosto_mtcnn(imagem_np)
        if rosto is None:
            st.error("Não foi possível detectar o rosto na imagem.")
            st.stop()
        imagem_para_features = rosto
        features = extrair_features_rosto(imagem_para_features)
        features = features.reshape(1, -1)
        features_pad = scaler_rosto.transform(features)
        reduzido = umap_rosto.transform(features_pad)
        cluster = kmeans_rosto.predict(reduzido)[0]

        cartela_nome = mapa_clusters_rosto.get(cluster, "Cartela Desconhecida")
        cores = paletas_rosto.get(cartela_nome, [])

    st.success(f"✅ A imagem foi classificada na **{cartela_nome}**.")

    if cores:
        cols = st.columns(len(cores))
        for c, hexcode in zip(cols, cores):
            c.markdown(
                f"""
                <div style="
                    background-color: {hexcode};
                    width: 100%;
                    height: 80px;
                    border-radius: 4px;
                    border: 1px solid #ccc;
                "></div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("Paleta de cores não encontrada para esta cartela.")
