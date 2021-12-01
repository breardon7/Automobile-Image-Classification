import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import os
import pandas as pd

import numpy as np

#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
'''
# Train Dataset creation
SAMPLE_SIZE = 1000
module_dir = os.path.dirname(__file__)  # Set path to current directory
train_meta_data_file_path = os.path.join(module_dir, 'Metadata/train-meta.xlsx')
train_data = pd.read_excel(train_meta_data_file_path).head(SAMPLE_SIZE)
train_images_file_path = os.path.join(module_dir, 'Dataset/Train/')
image_path = data[5].replace("'", "")
image_file_path = os.path.join(img_dir, image_path)
'''

# image = cv2.imread(r'C:\Users\brear\OneDrive\Documents\GitHub\Computer-Vision\Code\Dataset\Train\02753.jpg')
# image = cv2.imread(r"/home/ubuntu/Computer-Vision/Code/Dataset/Train02753.jpg")
#cv2.imshow('test image', image)

# Load image, grayscale, Otsu's threshold
image = cv2.imread("Dataset/Train/02753.jpg")
plt.imshow(image)
plt.show()
image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
blurred_float = image_blurred.astype(np.float32) / 255.0
edgeDetector = cv2.ximgproc.createStructuredEdgeDetection('model.yml.gz')
edges = edgeDetector.detectEdges(blurred_float) * 255.0
cv2.imwrite('edge-raw.jpg', edges)

def SaltPepperNoise(edgeImg):

    count = 0
    lastMedian = edgeImg
    median = cv2.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0
    count = count + 1
    if count > 70:
        return median
    '''lastMedian = median
    median = cv2.medianBlur(edgeImg, 3)'''
edges_ = np.asarray(edges, np.uint8)
SaltPepperNoise(edges_)
cv2.imwrite('edge.jpg', edges_)
image_display('edge.jpg')

def findSignificantContour(edgeImg):
    image, contours, hierarchy = cv2.findContours(
        edgeImg,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
        # Find level 1 contours
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)
# From among them, find the contours with large surface area.
    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])
    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour
contour = findSignificantContour(edges_u)
# Draw the contour on the original image
contourImg = np.copy(src)
cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
cv2.imwrite('contour.jpg', contourImg)
image_display('contour.jpg')

mask = np.zeros_like(edges_u)
cv2.fillPoly(mask, [contour], 255)
# calculate sure foreground area by dilating the mask
mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)
# mark inital mask as "probably background"
# and mapFg as sure foreground
trimap = np.copy(mask)
trimap[mask == 0] = cv2.GC_BGD
trimap[mask == 255] = cv2.GC_PR_BGD
trimap[mapFg == 255] = cv2.GC_FGD
# visualize trimap
trimap_print = np.copy(trimap)
trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
trimap_print[trimap_print == cv2.GC_FGD] = 255
cv2.imwrite('trimap.png', trimap_print)
image_display('trimap.png')






'''
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Morph open to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours and remove small noise
cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 50:
        cv2.drawContours(opening, [c], -1, 0, -1)

# Invert and apply slight Gaussian blur
result = 255 - opening
result = cv2.GaussianBlur(result, (3,3), 0)

# Perform OCR
data = pytesseract.image_to_string(result, lang='eng', config='--psm 6')
print(data)

cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imshow('result', result)
cv2.waitKey()
'''