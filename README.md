# Treinamento-Rede-Yolo

# YOLOv8 + Roboflow: Treinamento e Inferência em Google Colab

Este projeto demonstra como:

1. Instalar as bibliotecas necessárias
2. Baixar um dataset customizado do Roboflow
3. Treinar o modelo YOLOv8 com esse dataset
4. Realizar a inferência em imagens enviadas pelo usuário

## Requisitos

* Conta no [Google Colab](https://colab.research.google.com/)
* API Key da [Roboflow](https://roboflow.com/)
* Dataset preparado no formato YOLOv8

## Etapas do Projeto

### 1. Instalação das Dependências

```python
!pip install -q ultralytics
!pip install -q roboflow
```

### 2. Importação dos Pacotes

```python
import os
import shutil
from roboflow import Roboflow
import cv2
import matplotlib.pyplot as plt
from google.colab import files
from ultralytics import YOLO
```

### 3. Download do Dataset da Roboflow

```python
rf = Roboflow(api_key="SUA_API_KEY_AQUI")  # Substitua pela sua API Key
project = rf.workspace("SEU_WORKSPACE").project("coco-gtbwl")
version = project.version(1)
dataset = version.download("yolov8")
dataset_path = dataset.location
```

> O dataset será baixado automaticamente no formato esperado pelo YOLOv8.

### 4. Treinamento do Modelo

```python
model = YOLO("yolov8n.pt")  # Utiliza a versão nano como base

model.train(
    data='/content/COCO-Dataset-1/data.yaml',
    epochs=30,
    imgsz=640,
)
```

> Você pode ajustar o número de épocas (`epochs`) e o tamanho da imagem (`imgsz`) conforme necessário.

### 5. Função para Mostrar Imagens com Matplotlib

```python
def mostrar(frame):
    imagem = cv2.imread(frame)
    if imagem is None:
        print(f"Erro: imagem '{frame}' não encontrada.")
        return
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.show()
```

### 6. Upload de Imagem e Inferência

```python
uploaded = files.upload()
image_path = next(iter(uploaded))

results = model(image_path)
results[0].save(filename='predictions.jpg')

mostrar('predictions.jpg')
```

> Após o upload, a imagem será processada e os resultados serão exibidos com detecções visuais.
### Exemplo de Resultado

![image](https://github.com/user-attachments/assets/c43fa02c-98fd-4c8a-a7bc-1b998ee4b865)

---

## Observações

* Substitua a API key pela sua chave pessoal da Roboflow.
* O caminho `'/content/COCO-Dataset-1/data.yaml'` pode variar dependendo do nome do dataset baixado.
* Os resultados da inferência são salvos em `predictions.jpg`.

---

Se quiser, posso gerar esse `README.md` como arquivo para download. Deseja isso?
