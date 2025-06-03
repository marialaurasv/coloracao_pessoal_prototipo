import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()

def segmentar_pele_mtcnn(imagem_bgr):
    """
    Recebe imagem BGR, detecta rosto com MTCNN, segmenta pele usando máscara HSV e retorna
    imagem segmentada da pele (BGR). Se não encontrar rosto, retorna None.
    """
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    resultado = detector.detect_faces(imagem_rgb)
    if not resultado:
        return None

    x, y, w, h = resultado[0]['box']
    x, y = max(0, x), max(0, y)
    rosto = imagem_rgb[y:y+h, x:x+w]

    hsv = cv2.cvtColor(rosto, cv2.COLOR_RGB2HSV)

    # Faixa para pele clara
    lower1 = np.array([0, 20, 80], dtype=np.uint8)
    upper1 = np.array([25, 150, 255], dtype=np.uint8)

    # Faixa para pele mais escura/marrom
    lower2 = np.array([25, 40, 40], dtype=np.uint8)
    upper2 = np.array([50, 150, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    resultado = cv2.bitwise_and(rosto, rosto, mask=mask)
    return cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR)
