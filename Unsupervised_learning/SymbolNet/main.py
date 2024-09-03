#Biblioteca da VGG19
from tensorflow.keras.applications.vgg19 import VGG19
#Importa as bibliotecas para plotar os filtros
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg19 import decode_predictions
from tensorflow.keras import models
import my_utils

import tensorflow as tf

#Parametros do treinamento
batch_size = 32

#Cria um modelo que gera as saidas das camdas desejadas
images_path = '/home/gustavo/PROG/images/val2017/'

#Le os nomes das imagens e salva em um array
images_names = my_utils.get_images(images_path)
print("Numero de imagens: ", len(images_names))


for i, el in enumerate(maps):
    el = np.reshape(el, (1, np.shape(el)[0], np.shape(el)[1], 1))
    maps[i] = np.squeeze(tf.nn.max_pool2d(tf.convert_to_tensor((el),dtype=tf.float32), ksize=[1, 2], strides=[1, 2], padding='SAME'))
    print(i, ' -> ', np.shape(maps[i]))
maps = np.asarray(maps)