import cv2

image = cv2.imread('0001.jpg')
print(image.shape)
cv2.imshow('Image', image)
cv2.waitKey(0)
print( image[20,40] )
