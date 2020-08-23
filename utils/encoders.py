from tensorflow import keras
from tensorflow.keras import layers


def AlexNet(input_shape):

    inputs = keras.Input(input_shape)

    x = layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(256, (11, 11), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    outputs = layers.GlobalMaxPooling2D()(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='alexnet_encoder')

    return model
