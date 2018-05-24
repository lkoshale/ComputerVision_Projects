
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K


class ShallowNet:

    @staticmethod
    def build(width,height,depth,classes):  #classes is numbe of claasses to predict

        model = Sequential()
        inputshape = (height,width,depth)

        if(K.image_data_format()=='channels_first'):
            inputshape = (depth,height,width)

        #TODO 32 filters of size 3x3
        model.add(Conv2D(32,(3,3),padding='same',input_shape=inputshape))
        model.add(Activation('relu'))
        model.add(Flatten())                        #flatten before going in final FC layer
        model.add(Dense(classes))
        model.add(Activation('softmax'))            #final activation layer softmax

        return model
