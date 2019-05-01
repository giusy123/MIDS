from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten
from keras.models import Model
from keras import regularizers
import numpy as np


np.random.seed(12)
from tensorflow import set_random_seed

set_random_seed(12)


class DNN():
    def __init__(self, n_classes):
        self._nClass = n_classes

        # A deep autoecndoer model

    def deepAutoEncoder(self, x_train, params):
        n_col = x_train.shape[1]
        input = Input(shape=(n_col,))
        encoded = Dense(params['first_layer'], activation=params['first_activation'],
                        kernel_initializer=params['kernel_initializer'],
                        name='encoder1')(input)
        encoded = Dense(params['second_layer'], activation=params['first_activation'],
                        kernel_initializer=params['kernel_initializer'],

                        name='encoder2')(encoded)
        encoded = Dense(params['third_layer'], activation=params['first_activation'],
                        kernel_initializer=params['kernel_initializer'], activity_regularizer=regularizers.l1(10e-5),

                        name='encoder3')(encoded)

        encoded = Dropout(.5)(encoded)
        decoded = Dense(params['second_layer'], activation=params['first_activation'],
                        kernel_initializer=params['kernel_initializer'], name='decoder1')(encoded)
        decoded = Dense(params['first_layer'], activation=params['second_activation'],
                        kernel_initializer=params['kernel_initializer'], name='decoder2')(decoded)
        decoded = Dense(n_col, activation=params['third_activation'], kernel_initializer=params['kernel_initializer'],
                        name='decoder')(decoded)
        # serve per L2 normalization?
        # encoded1_bn = BatchNormalization()(encoded)

        autoencoder = Model(input=input, output=decoded)
        autoencoder.summary
        learning_rate = 0.001
        autoencoder.compile(loss=params['losses'],
                            optimizer=params['optimizer'](), metrics=['accuracy'])

        return autoencoder

    #model with fixed weights autoencoder
    def MLP_WeightFixed(self, encoder, x_train, params):

        n_col = x_train.shape[1]
        input_dim = Input(shape=(n_col,))
        # freeze all layers
        for layer in encoder.layers:
            layer.trainable = False

        layer = encoder(input_dim)
        layer = BatchNormalization()(layer)
        layer = Dropout(.5)(layer)
        softmax = Dense(self._nClass, activation='softmax')(layer)
        model = Model(input_dim, softmax)

        model.summary()
        model.compile(loss=params['losses'],
                      optimizer=params['optimizer'](),
                      metrics=['acc'])


        return model