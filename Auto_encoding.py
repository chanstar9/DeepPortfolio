from keras import regularizers
from keras.layers import Input, Dense, BatchNormalization, Activation
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
from keras.backend import sigmoid
from keras.callbacks import EarlyStopping


def swish(x, beta=0.5):
    return x * sigmoid(beta * x)


get_custom_objects().update({'swish': Activation(swish)})


def Autoencoder(encoding_dim, input_dim, encode_activation, decode_activation, optimizer, loss, x_train, shuffle, epoch,
                batch_size, validation_split):
    input_stock = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation=encode_activation, kernel_regularizer=regularizers.l1(10e-5))(input_stock)
    decoded = Dense(input_dim, activation=decode_activation, kernel_regularizer=regularizers.l1(10e-5))(encoded)
    autoencoder = Model(input_stock, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    autoencoder.fit(x_train, x_train,
                    shuffle=shuffle,
                    epochs=epoch,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[EarlyStopping(patience=10)],
                    validation_split=validation_split
                    )
    return autoencoder
