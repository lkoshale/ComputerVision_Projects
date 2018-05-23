from pyimagesearch.nn.conv.ShallowNet import ShallowNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreProcessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader

from keras.optimizers import SGD
import numpy as np
import imutils.paths as paths
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreProcessor(32,32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader([sp,iap])
(data,labels) = sdl.load(imagePaths,verbose=500)
data = data.astype(dtype='float32')/255.0

(trainX,testX,trainY,testY)= train_test_split(data,labels,test_size=0.25,random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")
opt = SGD(lr=0.005)

model = ShallowNet.build(width=32,height=32,depth=3,classes=2)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=['accuracy'])

print("[INFO] training network...")

H = model.fit(trainX,trainY,validation_data=(testX,testY),epochs=100,batch_size=32,verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

