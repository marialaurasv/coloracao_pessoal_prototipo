import numpy as np
import cv2
from sklearn.cluster import KMeans

def extrair_features_lab(imagem_bgr):
    imagem_lab = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2LAB)
    h, w, _ = imagem_lab.shape
    imagem_flat = imagem_lab.reshape((h * w, 3))
    # Remove pixels pretos (sem pele)
    imagem_flat = imagem_flat[np.any(imagem_flat != [0, 128, 128], axis=1)]
    if imagem_flat.shape[0] == 0:
        return np.zeros(27)
    hist_l = cv2.calcHist([imagem_lab], [0], None, [9], [0, 256]).flatten()
    hist_a = cv2.calcHist([imagem_lab], [1], None, [9], [0, 256]).flatten()
    hist_b = cv2.calcHist([imagem_lab], [2], None, [9], [0, 256]).flatten()
    hist = np.concatenate([hist_l, hist_a, hist_b])
    hist = hist / np.sum(hist)  # normaliza
    return hist

def extrair_features_rosto(imagem_bgr):
    h, w, _ = imagem_bgr.shape
    imagem_flat = imagem_bgr.reshape((h * w, 3))
    # Remove fundo preto
    imagem_flat = imagem_flat[np.any(imagem_flat != [0, 0, 0], axis=1)]
    if imagem_flat.shape[0] == 0:
        return np.zeros(7)
    # Tom médio
    media_rgb = np.mean(imagem_flat, axis=0)
    # Contraste (desvio padrão da luminância)
    imagem_gray = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    contraste = np.std(imagem_gray)
    # Cor dominante com KMeans
    kmeans = KMeans(n_clusters=1, random_state=42, n_init='auto')
    cor_dominante = kmeans.fit(imagem_flat).cluster_centers_[0]
    # Junta tudo (tom médio, contraste e cor dominante)
    features = np.concatenate([
        media_rgb,                  # 3 valores
        [contraste],                # 1 valor
        cor_dominante               # 3 valores
    ])
    return features  # total: 7
