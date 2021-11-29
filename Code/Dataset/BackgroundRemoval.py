import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import os
import pandas as pd

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
# Load image, grayscale, Otsu's threshold
# image = cv2.imread(r'C:\Users\brear\OneDrive\Documents\GitHub\Computer-Vision\Code\Dataset\Train\02753.jpg')
# image = cv2.imread(r"/home/ubuntu/Computer-Vision/Code/Dataset/Train02753.jpg")
#cv2.imshow('test image', image)

image = Image.open(r'C:\Users\brear\OneDrive\Documents\GitHub\Computer-Vision\Code\Dataset\Train\02753.jpg')
plt.imshow(image)
plt.show()
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
cv2.waitKey()'''