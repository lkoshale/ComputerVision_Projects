from pyimagesearch.nn.NeuralNetwork import NeuralNetwork
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

data = datasets.load_iris()

(trainX,testX,trainY,testY) = train_test_split(data.data,data.target,test_size=0.30)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

nn = NeuralNetwork([trainX.shape[1],4,4,4,3],alpha=0.03)

print("[INFO] {}".format(nn))

nn.fit(trainX,trainY,epochs=1000)

pred = nn.predict(testX)
pred = pred.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), pred))
