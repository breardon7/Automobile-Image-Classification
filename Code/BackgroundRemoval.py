import numpy as np
from cv2 import cv2


def salt_pepper_noise(image_edges):
    count = 0
    lastMedian = image_edges
    median = cv2.medianBlur(image_edges, 3)
    while not np.array_equal(lastMedian, median):
        zeroed = np.invert(np.logical_and(median, image_edges))
        image_edges[zeroed] = 0
        count = count + 1
        if count > 1000:
            return median


def find_significant_contour(image_edge):
    contours, hierarchy = cv2.findContours(
        image_edge,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    # Find level 1 contours
    level1Meta = []
    for contour_index, contour_tuple in enumerate(hierarchy[0]):
        # Filter the ones without parent
        if contour_tuple[3] == -1:
            contour_tuple = np.insert(contour_tuple.copy(), 0, [contour_index])
            level1Meta.append(contour_tuple)
    # From among them, find the contours with large surface area.
    contours_with_area = []
    for contour_tuple in level1Meta:
        contour_index = contour_tuple[0]
        contour = contours[contour_index]
        area = cv2.contourArea(contour)
        contours_with_area.append([contour, area, contour_index])
    contours_with_area.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contours_with_area[0][0]
    return largestContour


def remove_image_background(image_vec):
    blurred_image = cv2.GaussianBlur(image_vec, (5, 5), 0)
    blurred_float = blurred_image.astype(np.float32) / 255.0
    edge_detector = cv2.ximgproc.createStructuredEdgeDetection('Dataset/BackgroundModel/model.yml.gz')
    edges = edge_detector.detectEdges(blurred_float) * 255.0
    edges_array = np.asarray(edges, np.uint8)
    salt_pepper_noise(edges_array)
    contour = find_significant_contour(edges_array)
    # Draw the contour on the original image
    contourImg = np.copy(image_vec)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)

    mask = np.zeros_like(edges_array)
    cv2.fillPoly(mask, [contour], 255)
    # calculate sure foreground area by dilating the mask
    map_foreground = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)
    # mark initial mask as "probably background"
    # and mapFg as sure foreground
    tri_map = np.copy(mask)
    tri_map[mask == 0] = cv2.GC_BGD
    tri_map[mask == 225] = cv2.GC_PR_BGD
    tri_map[map_foreground == 255] = cv2.GC_FGD
    tri_map_print = np.copy(tri_map)
    tri_map_print[tri_map_print == cv2.GC_PR_BGD] = 128
    tri_map_print[tri_map_print == cv2.GC_FGD] = 255
    image_without_background = cv2.bitwise_and(image_vec, image_vec, mask=tri_map_print)
    return image_without_background


