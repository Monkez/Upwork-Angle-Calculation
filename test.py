from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from glob import glob as gl
import PIL
W = 224
H = 224

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def show(img, name = "show"):
    max_height = 400
    max_width = 800
    h, w = img.shape[:2]
    r1 = max_height/h
    r2 = max_width/w
    r = min(r1, r2)
    img = cv2.resize(img, (int(w*r), int(h*r)))
    cv2.imshow(name, img)
    

def crop_object(file):
    im = Image.open(file)
    img = np.array(im)
    h, w = img.shape[:2]
    img = img[h//3: 2*h//3]
    img0 = img.copy()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
    # show(thresh, "thresh")
    kernel = np.ones((7, 7),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # show(opening, "opening")

    blur = cv2.GaussianBlur(opening,(315, 115),0)
    # show(blur, "blur")

    ret, thresh2 = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY)
    ratio = np.sum(thresh2)/(255*img.shape[0]*img.shape[1])
    spliter = int(img.shape[1]/(img.shape[0]*(ratio+0.1)))
    thresh2[:, -50:]=0
    for i in range(spliter):
        pos = i*img_gray.shape[1]//(spliter)
        thresh2[:, pos:pos+50] = 0

    
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    
    areas = []
    for contour in contours:
        areas.append(cv2.contourArea(contour))
    arg_max = np.argsort(np.array(areas))[::-1]
    num = min(spliter-1, len(contours))
    out = []
    boxes = []
    for i in range(num):
        index = arg_max[i]
        rect = cv2.minAreaRect(contours[index])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,0,255), 15)
        warped = four_point_transform(img0, box)
        # show(warped, "warped "+str(i))
        boxes.append(order_points(box))
        out.append(warped)
    show(img)
    out = np.array(out, dtype=object)
    boxes= np.array(boxes)
    return out, boxes

def sort_box(out, boxes):
    arg_sort = np.argsort(boxes[:, 0, 0])
    return out[arg_sort]

def padding_to_square(img, W, H):
    h, w = img.shape[:2]
    out = np.zeros((H, W, 3), np.uint8)
    r = min(W/w, H/h)
    img = cv2.resize(img, (int(r*w), int(r*h)))
    h, w = img.shape[:2]
    out[(H-h)//2:(H-h)//2+h, (W-w)//2:(W-w)//2+w] = img
    return out


def predict(model, img):
    img = padding_to_square(img, W, H)
    # img = np.expand_dims(img, axis=2)
    out = model.predict(np.array([img]), verbose=0)[0][0]
    return out


if __name__ == "__main__":
    files = gl("Data/2020_Nr63_Garne von Janßen/alpha130/*")
    model = load_model("models/model.h5")
    random = np.random.choice(files)
    out, boxes = crop_object(random)
    crops = sort_box(out, boxes)
    angles = []
    for i in range(crops.shape[0]-1):
        angle = predict(model, crops[i])
        angles.append(angle)

    print(angles)
    print("Result:", np.mean(angles))
    cv2.waitKey(0)
    cv2.destroyAllWindows()