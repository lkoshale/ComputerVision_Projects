from keras.models import load_model
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreProcessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
import numpy as np
import imutils.paths as paths
import matplotlib.pyplot as plt
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
ap.add_argument("-m", "--model", required=True,help="path to pre-trained model")
args = vars(ap.parse_args())

# initialize the class labels
classLabels = ["cat", "dog"]

print("[INFO] sampling images...")

#grab the list of images in the dataset then randomly sample
imagePaths = np.array(list(paths.list_images(args['dataset'])))
idxs = np.random.randint(0,len(imagePaths),size=(10,))
imagePaths = imagePaths[idxs]

sp = SimplePreProcessor(32,32)
ima = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader([sp,ima])
(data,labels) = sdl.load(imagePaths,verbose=500)
data = data.astype(dtype='float32')/255.0

#TODO *********************************
# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args['model'])

# **************************************

print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

for (i,imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    cv2.putText(image,"Label: {}".format(classLabels[preds[i]]),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image",image)
    cv2.waitKey(0)


