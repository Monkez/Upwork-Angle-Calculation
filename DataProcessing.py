import cv2
import numpy as np
from glob import glob as gl
import config as cf
import imutils
import os

cf.points = [(0, 0), (0, 0)]
cf.stt = 0
cf.img = None
cf.img_show = None
cf.angle = 0

def padding_to_square(img, W, H):
    h, w = img.shape[:2]
    out = np.zeros((H, W), np.uint8)
    r = min(W/w, H/h)
    img = cv2.resize(img, (int(r*w), int(r*h)))
    h, w = img.shape[:2]
    out[(H-h)//2:(H-h)//2+h, (W-w)//2:(W-w)//2+w] = img
    return out

def record_points(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if cf.stt == 0:
            cf.img_show  = cf.img.copy()
        cf.points[cf.stt] = (x, y)
        color = (0, 255, 0) if cf.stt==0 else (255, 0, 0)
        cv2.circle(cf.img_show,(x,y), 3, color,-1)

        if cf.stt == 1:
            x1, y1, x2, y2 = cf.points[0][0], cf.points[0][1], cf.points[1][0], cf.points[1][1]
            angle = 180*np.arctan((y1-y2)/(x2-x1))/np.pi
            cf.angle = angle
            cf.img_show = cv2.putText(cf.img_show, str(round(angle, 2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 0), 1, cv2.LINE_AA)

            cf.img_show = cv2.line(cf.img_show, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cf.stt +=1
        cf.stt = cf.stt%2

def make_label(files):
    for i, file in enumerate(files):
        img = cv2.imread(file, 1)
        cf.img = img.copy()
        cf.img_show = cf.img.copy()
        cf.points = [(0, 0), (0, 0)]
        cf.stt = 0
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',record_points)
        while(1):
            cv2.imshow('image', cf.img_show)
            if cv2.waitKey(20) & 0xFF == 32:
                break

            f = open(file[:-3]+"txt", 'w')
            f.write(str(cf.angle))
            f.close()
        cv2.destroyAllWindows()

def show(img, name = "show"):
    max_height = 400
    max_width = 800
    h, w = img.shape[:2]
    r1 = max_height/h
    r2 = max_width/w
    r = min(r1, r2)
    img = cv2.resize(img, (int(w*r), int(h*r)))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_lable(file):
    label_file = file[:-3]+"txt"
    f = open(label_file, 'r')
    a = f.readline()
    angle = float(a)
    img = cv2.imread(file)
    h, w = img.shape[:2]
    p1 = (w//2, h//2)
    p2 = (w//2+200, h//2 - int(200*np.tan(angle*np.pi/180)))
    
    img = cv2.line(img, p1, p2, (0, 255, 0), 3)

    img = cv2.putText(img, str(round(angle, 2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 0), 1, cv2.LINE_AA)
    show(img)

def augument(file, num = 5):
    label_file = file[:-3]+"txt"
    f = open(label_file, 'r')
    a = f.readline()
    angle = float(a)
    img = cv2.imread(file)
    h, w = img.shape[:2]
    # show(img, "img")
    for i in range(num):
        angle_shiff = np.random.randint(-12, 12)
        rotated = imutils.rotate_bound(img, -angle_shiff)
        rotated = rotated[h//7:-h//7, w//7:-w//7]
        target = "aug/"+ os.path.basename(file)[:-3]+"#"+str(i)+".jpg"
        f = open(target[:-3]+"txt", 'w')
        f.write(str(angle+angle_shiff))
        f.close()
        cv2.imwrite(target, rotated)
        # show(rotated)


if __name__ == "__main__":
    files = gl("crops/*.jpg")
    random_index = np.random.permutation(len(files))
    files = np.array(files)[random_index]

    # make_label(files)
    for file in files:
        augument(file)

