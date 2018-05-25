from YOLO.tiny_YOLO import tinyYOLO
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import plot_model
from utils.util import load_weights,yolo_net_out_to_car_boxes,draw_box
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = tinyYOLO.build(width=448,height=448,depth=3)
print(model.summary())
# plot_model(model,to_file='tinyYolo.png',show_shapes=True)

load_weights(model,'./yolo-tiny.weights')


imagePath = 'test1.jpg'
image = plt.imread(imagePath)
image_crop = image[300:650,500:,:]
resized = cv2.resize(image_crop,(448,448))


batch = np.transpose(resized,(2,0,1))
batch = 2*(batch/255.) - 1
batch = np.expand_dims(batch, axis=0)
out = model.predict(batch)

boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.17)

f,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
ax1.imshow(image)
ax2.imshow(draw_box(boxes,plt.imread(imagePath),[[500,1280],[300,650]]))