{\rtf1\ansi\ansicpg1252\cocoartf2708
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 \
PROT\'d3TIPO DE APLICA\'c7\'c3O PARA IDENTIFICA\'c7\'c3O DE COLORA\'c7\'c3O PESSOAL COM IA\
\
Descri\'e7\'e3o:\
Este projeto classifica rostos femininos em uma das 12 cartelas da colora\'e7\'e3o pessoal\
(baseado no m\'e9todo sazonal expandido), utilizando t\'e9cnicas de vis\'e3o computacional\
e intelig\'eancia artificial.\
\
ESTRUTURA DO PROJETO:\
\
coloracao_pessoal_prototipo/\
\uc0\u9500 \u9472 \u9472  imagens_originais/        # Imagens completas dos rostos\
\uc0\u9500 \u9472 \u9472  imagens_segmentadas/      # Apenas as regi\'f5es de pele extra\'eddas\
\uc0\u9500 \u9472 \u9472  modelo/                   # Modelos treinados (opcional)\
\uc0\u9500 \u9472 \u9472  notebooks/\
\uc0\u9474    \u9492 \u9472 \u9472  processamento.ipynb   # Notebook principal com todo o pipeline\
\uc0\u9500 \u9472 \u9472  faceparsing_bisenet.pth   # Modelo pr\'e9-treinado de segmenta\'e7\'e3o facial\
\uc0\u9500 \u9472 \u9472  requirements.txt          # Lista de depend\'eancias\
\uc0\u9492 \u9472 \u9472  README.txt                # Este arquivo\
\
REQUISITOS:\
- Python 3.12.4\
- Uso de ambiente virtual (venv) recomendado\
\
INSTALA\'c7\'c3O:\
\
1. Clone o reposit\'f3rio:\
   git clone https://github.com/seuusuario/coloracao_pessoal_prototipo.git\
   cd coloracao_pessoal_prototipo\
\
2. Crie e ative um ambiente virtual:\
   python -m venv venv\
   source venv/bin/activate     (macOS/Linux)\
   venv\\Scripts\\activate        (Windows)\
\
3. Instale os pacotes:\
   pip install -r requirements.txt\
\
PIPELINE:\
\
1. Segmenta\'e7\'e3o de pele com MTCNN + BiSeNet\
2. Extra\'e7\'e3o de 27 features (histogramas LAB + cor dominante)\
3. Redu\'e7\'e3o dimensional com UMAP\
4. Clusteriza\'e7\'e3o em 12 grupos com KMeans\
5. Classifica\'e7\'e3o de novas imagens com base no cluster mais pr\'f3ximo\
\
ENTRADA:\
- imagens_originais/ (rostos completos)\
\
SA\'cdDA:\
- imagens_segmentadas/ (somente regi\'f5es de pele segmentadas)\
\
USO:\
- Insira imagens em imagens_originais/\
- Execute o Jupyter Notebook `notebooks/processamento.ipynb`\
- O pipeline ser\'e1 executado automaticamente\
\
LICEN\'c7A:\
Uso educacional e acad\'eamico. N\'e3o recomendado para produ\'e7\'e3o.\
}