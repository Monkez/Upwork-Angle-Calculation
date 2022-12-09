import numpy as np
import cv2
from PIL import Image, ImageEnhance
from glob import glob as gl
import PIL
import config as cf
cf.stt = 398
cf.show = False


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
    if cf.show :
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
    return np.array(out), np.array(boxes)

def sort_box(out, boxes):
    arg_sort = np.argsort(boxes[:, 0, 0])
    print(arg_sort)

    return out[arg_sort]

def padding_to_square(img, W, H):
    h, w = img.shape[:2]
    out = np.zeros((H, W, 3), np.uint8)
    r = min(W/w, H/h)
    img = cv2.resize(img, (int(r*w), int(r*h)))
    h, w = img.shape[:2]
    out[(H-h)//2:(H-h)//2+h, (W-w)//2:(W-w)//2+w] = img
    return out
    


def angle_calculate(img, i):
    img = img[20:-20]
    h, w = img.shape[:2]
    r = 163/h
    img = cv2.resize(img, (int(w*r), 163))
    image = Image.fromarray(img)
    new_image = np.array(ImageEnhance.Contrast(image).enhance(2))
    img_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    th2 = cv2.adaptiveThreshold(img_gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,15, 5)
    show(th2, "th2")

    img_gray = cv2.GaussianBlur(img_gray, (7,7), 0) 
    edges = cv2.Canny(image=img_gray, threshold1= 30, threshold2=30)
    cv2.imwrite("edges/" + str(cf.stt)+".jpg", edges)
    cv2.imwrite("crops/" + str(cf.stt)+".jpg", img)
    cf.stt+=1
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # kernel = np.ones((5,5),np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # show(morph, "morph")
    angles = []
    lines = cv2.HoughLinesP(
            edges, # Input edge image
            5, # Distance resolution in pixels
            np.pi/720, # Angle resolution in radians
            threshold=75, # Min number of votes for valid line
            minLineLength=70, # Min allowed length of line
            maxLineGap=5 # Max allowed gap between line for joining them
            )

    # Iterate over points
    if lines is not None:
        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            angle = 180*np.arctan((y1-y2)/(x2-x1))/np.pi
            # if y_min < h//3 and y_max>2*h//3:
            if 10<angle <80:
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0), 1)
                cv2.line(edges_rgb,(x1,y1),(x2,y2),(0,255,0), 1)

                angles.append(angle)
    img = cv2.resize(img, (w, h))
    show(edges_rgb, "edges rgb " )
    show(img, "img " + str(i))
    print(np.mean(angles))
    return angles

def sort_box(out, boxes):
    arg_sort = np.argsort(boxes[:, 0, 0])
    print(arg_sort)

    return out[arg_sort]

if __name__ == "__main__":
    files = gl("2020_Nr63_Garne von Janßen/alpha130/*")
    # random = np.random.choice(files)
    for file in files:
        out, boxes = crop_object(file)
        out = sort_box(out, boxes)
        for i in range(out.shape[0]):
            angle_calculate(out[i], i)
    print(cf.stt)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()




