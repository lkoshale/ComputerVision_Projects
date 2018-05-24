from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
import keras.backend as K

class MiniVGGNet:

    @staticmethod
    def build(width,height,depth,classes):

        image_shape = (height,width,depth)
        model = Sequential()
        chanDim = -1

        if K.image_data_format()== 'channels_first':
            image_shape = (depth,height,width)
            chanDim = 1

        model.add(Conv2D(32,(3,3),input_shape=image_shape,padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(rate=0.25))

        model.add(Conv2D(64,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(rate=0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model