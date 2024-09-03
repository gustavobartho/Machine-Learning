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

# Import Independent Component Analysis Algorithm
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Bibliotecas dos Autoencoders
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

import os
import glob
import pickle

#**********FUNÇÕES
#------------------------------------------------------------------------------------------------
# Le os nomes das imagens e cria uma lista com os nomes
def get_images(images_path):
    images_names = []
    os.chdir(images_path)
    for file_name in glob.glob("*.jpg"):
        images_names.append(file_name)

    return images_names
#------------------------------------------------------------------------------------------------
#***********CLASSES
#------------------------------------------------------------------------------------------------
class ConvAutoencoder(object):
    def __init__(self, width, height, depth, filters, latentDim, learning_rate=1e-3):
        '''
        Inputs:
            width(int): Width of the input image in pixels.
            height(int): Height of the input image in pixels.
            depth(int): Number of channels (i.e., depth) of the input volume.
            filters(tuple(int)): A tuple that contains the set of filters for convolution operations.
            latentDim(int): The number of neurons in our fully-connected (Dense) latent vector
        '''
        self.width = width
        self.height = height
        self.depth = depth
        self.filters = filters
        self.latentDim = latentDim
        self.learning_rate = learning_rate
        self.optimizer = Adam(learning_rate)
        self.encoder, self.decoder, self.autoencoder = self.create_nets(width, height, depth, filters, latentDim)

    # ---------------------------------------------------
    def create_nets(self, width, height, depth, filters, latentDim):
        inputShape = (height, width, depth)
        chanDim = -1
        # define the input to the encoder
        inputs = Input(shape=inputShape)
        x = inputs
        # loop over the number of filters
        for f in filters:
            # apply a CONV => RELU => BN operation
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)
        # flatten the network and then construct our latent vector
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim)(x)
        # build the encoder model
        encoder = Model(inputs, latent, name="encoder")

        # start building the decoder model which will accept the
        # output of the encoder as its inputs
        latentInputs = Input(shape=(latentDim,))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
        # loop over our number of filters again, but this time in
        # reverse order
        for f in filters[::-1]:
            # apply a CONV_TRANSPOSE => RELU => BN operation
            x = Conv2DTranspose(f, (3, 3), strides=2,
                                padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        # apply a single CONV_TRANSPOSE layer used to recover the
        # original depth of the image
        x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
        outputs = Activation("sigmoid")(x)
        # build the decoder model
        decoder = Model(latentInputs, outputs, name="decoder")
        # our autoencoder is the encoder + decoder
        autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")

        autoencoder.compile(loss="mse", optimizer=self.optimizer)
        # return 3 networks
        return encoder, decoder, autoencoder

    # ---------------------------------------------------
    def train(self, epochs, batch_size, train_data, train_label, val_data, val_label):
        self.autoencoder.fit(
            train_data,
            train_label,
            validation_data=(val_data, val_label),
            epochs=epochs,
            batch_size=batch_size
        )


#------------------------------------------------------------------------------------------------
class Sensory_Net(object):
    def __init__(self):
        self.model = VGG19()
        self.extraction_layers = [6, 11, 16, 21]
        self.activation_model = models.Model(inputs=self.model.input, outputs=[layer.output for layer in np.array(self.model.layers)[self.extraction_layers]])

    #----------------------------------------------------
    # Plota as camads de uma rede enumeradas para extração
    def plot_model(self):
        j = 1

        for i in self.model.layers[1:]:
            print(j, ' ', i.name, " - ", i.output_shape)
            j += 1

        return
    # ---------------------------------------------------
    def dimRed(self, patterns, stride, num_comp=1, red_type='PCA'):
        '''
        Aplica um algoritmo de PCA em um padrão de ativações
        Inputs:
            :param patterns: (Array[image, layers][pixels, chanels]) -> vetor com a saida de uma cmada
            :param stride: (int) -> Numero de canais incluidos em uma redução
            :param num-comp: (int) -> numero de componentes que serão gerados em cada linha
            :param red_type: (String) -> Tipo de algoritmo de redução que será aplicado
                red_type = 'PCA' -> Principal Component Analysis
                red_type = 'fastICA' -> Fast Independent Component Analysis
                red_type = 'TSNE' -> t-Distributed Stochastic Neighbor Embedding
        Output:
            :return maps: (Array[image, layer][pixel, canais]
        '''

        images = []

        # Initialize the algorithm and set the number of PC's
        if (red_type == 'FastICA'):
            dim_red_func = FastICA(n_components=num_comp)

        elif (red_type == 'TSNE'):
            dim_red_func = TSNE(n_components=num_comp)

        elif (red_type == 'PCA'):
            dim_red_func = PCA(n_components=num_comp)

        # Fit and transform the model to data. It returns a list of independent components
        for image in patterns:

            maps = []
            for layer in image:

                aux = []
                k = 0

                for i in range(stride, np.shape(layer)[-1], stride):
                    aux = dim_red_func.fit_transform(layer[:, k:i]) if k == 0 else \
                        np.concatenate((aux, dim_red_func.fit_transform(layer[:, k:i])), axis=1)
                    k = i
                if(k < np.shape(layer)[-1] - 1):
                    aux = np.concatenate((aux, dim_red_func.fit_transform(layer[:, k:])), axis=1)

                f_min, f_max = np.amin(aux), np.amax(aux)
                aux = (aux - f_min) / (f_max - f_min)

                maps.append(aux)

            images.append(maps)

        return images

    # ---------------------------------------------------
    def extract_layers(self, images_path, images_names):
        '''Extrai as ativações das camadas para imagens de entrada
        Inputs:
            :param images_path: (String) -> Caminho das imagens
            :param images_names: (Array[batch_size]) -> Array com os nomes das imagens
        Outputs:
            :return: (Array[batch_size, num_layers][pixel, canais])
        '''
        patterns = []
        patterns_aux = []

        # Itera pelas imagens selecionadas
        for img_name in images_names:

            # Carrega a imagem no formato para a VGG19 e transforma em um vetor
            image = img_to_array(load_img(images_path + img_name, target_size=(224, 224)))
            image = image[np.newaxis, :]
            # Passa a imagem pelo modelo e extrai as camadas desejadas
            activations = self.activation_model.predict(image)

            for activation in activations:
                # Normaliza os valores das ativações
                act_min, act_max = np.amin(activation), np.amax(activation)
                activation = (activation - act_min) / (act_max - act_min)

                shape_1 = int(np.shape(activation)[-1])
                shape_2 = int(np.prod(np.shape(activation)) / shape_1)
                patterns_aux.append(activation.reshape((shape_2, shape_1)))

            patterns.append(patterns_aux)
            patterns_aux = []

        return patterns

    #----------------------------------------------------
    def extract_batch(self, batch_size, images_path, images_names, stride, num_comp, pooling_ratio):
        '''
        Extrai uma batch de ativações prontas para os autoencoders
        Inputs:
            :param batch_size: (int) -> Tamanho da batch
            :param images_path: (String) -> Caminho das imagens
            :param images_names:  (Array) -> Nomes das imagens
             :param stride: (int) -> Numero de canais incluidos em uma redução
            :param num-comp: (int) -> numero de componentes que serão gerados em cada linha
            :param pooling_ratio:  (int) -> Razão na redução do numero de canais
        Outputs:
            :return: (Array[
        '''
        batch_activations = []

        #Escolhe um batch de imagens aletórias
        batch = np.random.choice(images_names, batch_size)

        # Extrai as camadas desejadas da rede
        #Passa as imagens pela rede e extrai as camadas desejadas
        maps = np.squeeze(np.asarray(self.extract_layers(images_path, images_names)))
        # Passa pelo algoritmo de redução de dimensionalisdade
        maps = self.dimRed(maps, stride, num_comp)

        #Itera pelas imagens da batch
        for j, image in enumerate(maps):
            #Itera pelas ativações de cada imagem
            for i, layer in enumerate(image):
                #Redimensiona a ativação para passar pelo pooling
                layer = np.reshape(layer, (1, np.shape(layer)[0], np.shape(layer)[1], 1))
                maps[j][i] = np.squeeze(
                    tf.nn.max_pool2d(
                        tf.convert_to_tensor((el), dtype=tf.float32),
                        ksize=[1, pooling_ratio],
                        strides=[1, pooling_ratio],
                        padding='SAME',
                    )
                )
        maps = np.asarray(maps)
        batch_activations.append(maps)

        return batch_activations




