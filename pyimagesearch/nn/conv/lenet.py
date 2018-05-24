
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import keras.backend as K

class LeNet:

    @staticmethod
    def build(width,height,depth,classes):
        inputshape = (height,width,depth)
        model = Sequential()

        if K.image_data_format()=="channels_first":
            inputshape = (depth,height,width)

        model.add(Conv2D(20,(5,5),padding='same',input_shape=inputshape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(50,(5,5),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
