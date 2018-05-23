import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Image',image)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

## getting hsitogram gray
# hist = cv2.calcHist([gray],[0],None,[256],[0,256])
#
# plt.figure()
# plt.title('GrayScale Image')
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(hist)
# plt.xlim([0, 256])
# plt.show()


chans = cv2.split(image)
colors = ('b','g','r')
plt.figure()
plt.title('RGB Image')
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features=[]

for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    features.extend(hist)
    # plot the histogram
    plt.plot(hist, color=color)
    plt.xlim([0, 256])


plt.show()
print("flattened feature vector size: %d" % (np.array(features).flatten().shape))
