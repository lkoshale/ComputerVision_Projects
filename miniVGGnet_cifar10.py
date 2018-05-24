
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras import backend as k
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
import argparse
import numpy as np
import imutils
import argparse

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.datasets import cifar10

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")
((trainX,trainY),(testX,testY)) = cifar10.load_data()

trainX = trainX.astype(dtype='float32') / 255.0
testX = testX.astype(dtype='float32') / 255.0

le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]

print("[INFO] compiling model...")

model = MiniVGGNet.build(width=32,height=32,depth=3,classes=10)
opt = SGD(lr=0.01,decay=0.01/40,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=opt)

print("[INFO] training network...")
H = model.fit(trainX,trainY,validation_data=(testX,testY),epochs=40,batch_size=64,verbose=1)

print("[INFO] evaluating network...")

predict = model.predict(testX,batch_size=64)
print(classification_report(testY.argmax(axis=1),predict.argmax(axis=1),target_names=labelNames))


#ploting
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
