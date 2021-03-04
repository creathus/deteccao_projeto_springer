# Detecção de pinos e molas para o Projeto Springer - SALCOMP
O projeto tem como base principal usar o [PyTorch YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) a fim de detectar elementos presentes na fonte de alimentação do Springer. Estes elementos foram classificados como **Pino** e **Molas**

[YOLO](https://pjreddie.com/darknet/yolo/) (**You Only Look Once**) é um modelo de rede neural profunda o qual está otimizado para obter detecções em velocidades elevadas, isto é, torna-se uma boa opção para rodar em **VÍDEO**. O desempenho funciona muito melhor se o computador tem uma GPU dedicada da NVIDIA.

Por default este modelo foi treinado para dados customizados, ou seja, usando um **dataset** criado a partir de imagens obtidas do projeto em questão. O mesmo escontra-se em ```/data/custom```.

As etapas a seguir, tem como objetivo utilizar a detecção dos elementos do Springer, utilizando uma Webcam Logitech BRIO 4K. Para isso, usando o Anaconda ou Pycharm, cria-se um ambiente virtual no computador.

# Criação do ambiente no Pycharm
Cria-se un ambiente chamado ```springer_yolo``` o qual tem a versão 3.6 do interpretador **Python**.
``` 
python -n springer_yolo python=3.6
```

É necessário ativarmos o ambiente ```springer_yolo```, antes de instalarmos os requisitos principais para o funcionamento do algoritmo.
```
source activate springer_yolo
```

# Instalação dos pacotes necessários para o ambiente virtual
Dentro do arquivo ```requirements.txt``` encontram-se os pacotes necessários de acordo com a versão também. No terminal de comando do Pycharm, digite:
```
pip install -r requirements.txt
```

# Realizar o Download dos modelos treinados e pesos 
A fim de executar o modelo **Yolo** temos que realizar o download dos pesos da rede neural, os quais são os valores que possuem as conexões Para poder correr el modelo de yolo tendremos que descargar los pesos de la red neuronal, los pesos son los valores que tienen todas las conexiones entre las neuronas de la red neuronal de YOLO, este tipo de modelos son computacionalmente muy pesados de entrenar desde cero por lo cual descargar el modelo pre entrenado es una buena opción.

```
bash weights/download_weights.sh
```

Movemos los pesos descargados a la carpeta llamada weights
```
mv yolov3.weights weights/
```

# Correr el detector de objetos en video 
Por ultimo corremos este comando el cual activa la camara web para poder hacer deteccion de video sobre un video "en vivo"
```
python deteccion_video.py
```

# Modificaciones
Si en vez de correr detección de objetos sobre la webcam lo que quieres es correr el modelo sobre un video que ya fue pre grabado tienes que cambiar el comando para correr el codigo a:

```
python deteccion_video.py --webcam 0 --directorio_video <directorio_al_video.mp4>
```

# Entrenamiento 

Ahora, si lo que quieres es entrenar un modelo con las clases que tu quieras y no utilizar las 80 clases que vienen por default podemos entrenar nuestro propio modelo. Estos son los pasos que deberás seguir:

Primero deberás etiquetar las imagenes con el formato VOC, aqui tengo un video explicando como hacer este etiquetado: 

Desde la carpeta config correremos el archivo create_custom_model para generar un archivo .cfg el cual contiene información sobre la red neuronal para correr las detecciones
```
cd config
bash create_custom_model.sh <Numero_de_clases_a_detectar>
cd ..
```
Descargamos la estructura de pesos de YOLO para poder hacer transfer learning sobre esos pesos
```
cd weights
bash download_darknet.sh
cd ..
```

## Poner las imagenes y archivos de metadata en las carpetar necesarias

Las imagenes etiquetadas tienen que estar en el directorio **data/custom/images** mientras que las etiquetas/metadata de las imagenes tienen que estar en **data/custom/labels**.
Por cada imagen.jpg debe de existir un imagen.txt (metadata con el mismo nombre de la imagen)

El archivo ```data/custom/classes.names``` debe contener el nombre de las clases, como fueron etiquetadas, un renglon por clase.

Los archivos ```data/custom/valid.txt``` y ```data/custom/train.txt``` deben contener la dirección donde se encuentran cada una de las imagenes. Estos se pueden generar con el siguiente comando (estando las imagenes ya dentro de ```data/custom/images```)
```
python split_train_val.py
```

## Entrenar

 ```
 python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74 --batch_size 2
 ```

## Correr deteccion de objetos en video con nuestras clases
```
python deteccion_video.py --model_def config/yolov3-custom.cfg --checkpoint_model checkpoints/yolov3_ckpt_99.pth --class_path data/custom/classes.names  --weights_path checkpoints/yolov3_ckpt_99.pth  --conf_thres 0.85
```
