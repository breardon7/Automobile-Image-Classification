import matplotlib.pyplot as plt

import numpy as np
from cv2 import cv2


def SaltPepperNoise(edgeImg):
    count = 0
    lastMedian = edgeImg
    median = cv2.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0
        count = count + 1
        if count > 1000:
            return median


def findSignificantContour(edgeImg):
    contours, hierarchy = cv2.findContours(
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


def remove_image_background(image_vec):
    image_blurred = cv2.GaussianBlur(image_vec, (5, 5), 0)
    blurred_float = image_blurred.astype(np.float32) / 255.0
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection('Dataset/BackgroundModel/model.yml.gz')
    edges = edgeDetector.detectEdges(blurred_float) * 255.0
    # cv2.imwrite('edge-raw.jpg', edges)
    edges_ = np.asarray(edges, np.uint8)
    SaltPepperNoise(edges_)
    # cv2.imwrite('edge.jpg', edges_)
    contour = findSignificantContour(edges_)
    # Draw the contour on the original image
    contourImg = np.copy(image_vec)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    # cv2.imwrite('contour.jpg', contourImg)

    mask = np.zeros_like(edges_)
    cv2.fillPoly(mask, [contour], 255)
    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)
    # mark initial mask as "probably background"
    # and mapFg as sure foreground
    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 225] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD
    # visualize trimap
    trimap_print = np.copy(trimap)
    trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
    trimap_print[trimap_print == cv2.GC_FGD] = 255
    #plt.imshow(trimap_print)
    #plt.show()
    # cv2.imwrite('trimap.png', trimap_print)

    image_without_background = cv2.bitwise_and(image_vec, image_vec, mask=trimap_print)
    plt.imshow(image_without_background)
    plt.show()
    return image_without_background
    # cv2.imwrite('test.png', image_vec)


