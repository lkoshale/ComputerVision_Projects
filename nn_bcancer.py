import numpy as np

from pyimagesearch.nn.NeuralNetwork import NeuralNetwork
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

print("[INFO] loading Breast Cancer (sample) dataset...")

cancer = datasets.load_breast_cancer()
data = cancer.data.astype("float")

data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

(trainX,testX,trainY,testY) = train_test_split(data,cancer.target,test_size=0.25)

trainYt = trainY
testYt = testY

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] training network...")

nn = NeuralNetwork([ trainX.shape[1],15,15,1])
print("[INFO] {}".format(nn))

nn.fit(trainX,trainY,epochs=500)

print("[INFO] evaluating network...")

pred = nn.predict(testX)
pred = pred.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), pred))

# boosted decision tree for comparison with neural net
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(learning_rate=0.07,n_estimators=300)
clf.fit(trainX,trainYt)
print(clf.score(testX,testYt))



