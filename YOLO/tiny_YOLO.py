
""""
build by : Lokesh
"""

from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Activation
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K


class tinyYOLO:

    """
    tiny YOLO model
    """

    @staticmethod
    def build(width,height,depth,classes=None):
        input_shape = (height,width,depth)
        model = Sequential()

        if K.image_data_format() == "channels_first":
            input_shape = (depth,height,width)

        #start adding layers to model

        model.add(Conv2D(16,(3,3),padding='same',input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(32,(3,3),padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64,(3,3),padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(128,(3,3),padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(256,(3,3),padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(512,(3,3),padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(1024,(3,3),padding='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(1024,(3,3),padding='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(1024,(3,3),padding='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(1470))

        return model



