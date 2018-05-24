from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.lenet import LeNet
from sklearn import datasets
from keras import backend as K
import numpy as np
import argparse
from keras.optimizers import SGD
import matplotlib.pyplot as plt


dataset = datasets.fetch_mldata("MNIST Original")
data = datasets.data

if K.image_data_format() == 'channels_first':
    data = data.reshape(data.shape[0],1,28,28)
else:
    data = data.reshape(data.shape[0],28,28,1)

(trainX,testX,trainY,testY)= train_test_split(data/255.0,dataset.target.astype('int'),test_size=0.25,random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("[INFO] compiling model...")
model = LeNet.build(28,28,1,10)
model.compile(optimizer=SGD(lr=0.1),loss="categorical_crossentropy",metrics=['accuracy'])

print("[INFO] training network...")
H = model.fit(trainX,trainY,validation_data=(testX,testY),epochs=20,batch_size=28,verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=[str(x) for x in lb.classes_]))


#plot

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
