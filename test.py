
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreProcessor
import cv2

class Box:

    def __init__(self,x,y,w,h,conf=1):

        if x<0 or y<0 or w<0 or h<0:
            print("[INFO] Error in box shapes less than 0")
            return

        if x>1 or y>1 or w >1 or h>1:
            print("[INFO] Error in box shapes greater than 1")
            return

        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.conf = conf


def draw_grid(image,grid_size=(7,7)):
    (X,Y,Z)= image.shape
    divx = int(X/grid_size[0])
    divy = int(Y/grid_size[1])
    x = 0
    y = 0
    Xlist =[]
    Ylist = []
    for i in range(0,grid_size[0]):
        x+= divx
        Xlist.append(x)
    for i in range(0,grid_size[0]):
        y+=divy
        Ylist.append(y)

    for i in range(0,grid_size[0]):
        cv2.line(image,(Xlist[i],0),(Xlist[i],Ylist[-1]),(255, 0, 0),1,1)

    for i in range(0, grid_size[0]):
        cv2.line(image, (0,Ylist[i]), (Xlist[-1], Ylist[i]), (255, 0, 0), 1, 1)

    return image


def draw_boxes_rect(image,box,image_shape,grid_size,chanFirst=True):

    (Z,X,Y) = image_shape
    if not chanFirst:
        (X,Y,Z) = image_shape

    glen = X / grid_size[0]
    cx = ( box.x * glen )+ glen
    cy = (box.y * glen) + glen

    print(glen,cx,cy)
    actW = int(box.w* Y)
    actH = int(box.h* X)

    cordx1 = int(cx - (actW/2))
    cordy1 = int(cy - (actH/2))
    cordx2 = int(cx + (actW/2))
    cordy2 = int(cy + (actH/2))

    print(cordx1,cordy1,cordx2,cordy2)
    cv2.rectangle(image,(cordx1,cordy1),(cordx2,cordy2),(0,0,0),3,1)
    cv2.imshow("new image", image)
    return image




sp = SimplePreProcessor(height=448,width=448)
image = cv2.imread('test1.jpg')
rim = sp.preprocess(image)

# cv2.imshow("Original",image)
cv2.imshow("Changed",rim)
grim = draw_grid(rim)
box = Box(0.5,0.5,0.32,0.62)
nm = draw_boxes_rect(grim,box,grim.shape,(7,7),chanFirst=False)
cv2.imshow("new image",nm)
cv2.waitKey(0)


