# 🧠 Identificação de Coloração Pessoal com IA

Este projeto tem como objetivo identificar a cartela de coloração pessoal (sazonal expandido) a partir de imagens faciais. Utiliza técnicas de segmentação, extração de características, redução de dimensionalidade e clusterização não supervisionada para agrupar rostos em 12 categorias cromáticas.

## ✅ Requisitos

- Python 3.12.4
- Ambiente virtual recomendado

---

## 📦 Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/coloracao_pessoal_prototipo.git
cd coloracao_pessoal_prototipo
```

2. Crie e ative um ambiente virtual:

```bash
python -m venv venv
# Ative o ambiente:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## 🧪 Pipeline do Projeto

O pipeline completo está implementado no notebook `notebooks/processamento.ipynb` e segue as etapas:

### 1. Segmentação da Pele
- Segmenta apenas a região da pele do rosto, removendo elementos como olhos, sobrancelhas, boca e cabelo.
- As imagens segmentadas são salvas na pasta `imagens_segmentadas/`.

### 2. Extração de Características
- Cálculo do histograma de cor no espaço de cor LAB (`L`, `A`, `B`).
- Extração da cor dominante usando KMeans.
- Cada imagem resulta em um vetor com 27 características.

### 3. Redução de Dimensionalidade
- Utiliza UMAP para projetar os vetores de características em 2D.
- Essa projeção facilita a visualização e a clusterização das imagens.

### 4. Clusterização
- O algoritmo KMeans é aplicado sobre os dados reduzidos para formar 12 clusters.
- Cada cluster representa uma das 12 cartelas de coloração pessoal (método sazonal expandido).

### 5. Classificação de Novas Imagens
- Novas imagens passam pelas mesmas etapas de segmentação e extração de características.
- São projetadas com o UMAP já treinado e classificadas com base na distância ao centroide dos clusters do KMeans.

---

## 🖥️ Interface Gráfica com Streamlit

O arquivo `app.py` fornece uma interface gráfica para testar o modelo de forma simples e interativa.

### Como executar a aplicação:

1. Certifique-se de estar no ambiente virtual e com as dependências instaladas (`requirements.txt`).
2. No terminal, execute o seguinte comando:

```bash
streamlit run app.py
```

3. Isso abrirá automaticamente o navegador com a interface do aplicativo.

### Como usar:

- Faça upload de uma imagem com o rosto da pessoa.
- Selecione uma opção do modo: Modo Pele ou Modo Rosto.
- A aplicação segmentará automaticamente a pele da imagem.
- Em seguida, ela exibirá a cartela de coloração pessoal estimada (um dos 12 clusters).

---

## 📌 Observações

- A base de dados já deve estar presente na pasta `imagens_originais/`.
- O treinamento do modelo é feito diretamente no notebook `processamento.ipynb`.
- Os resultados finais são organizados em clusters representando as 12 cartelas.

---

