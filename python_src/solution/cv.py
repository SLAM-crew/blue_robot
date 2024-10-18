import cv2
import numpy as np

def align_histogram(frame):
    frame_rgb = cv2.split(frame)
    mean1 = np.median(frame_rgb)
    desired_mean = 60
    alpha = mean1 / desired_mean
    Inew_RGB = []
    for layer in frame_rgb:
        Imin = layer.min()
        Imax = layer.max()
        Inew = ((layer - Imin) / (Imax - Imin)) ** alpha
        Inew_RGB.append(Inew)
    Inew = cv2.merge(Inew_RGB)
    Inew_1 = (255*Inew).clip(0, 255).astype(np.uint8)
    return Inew_1

def follow_cube(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    blurVal = 3
    frame_hsv = cv2.medianBlur(frame_hsv, 1 + blurVal * 2)

    mask = cv2.inRange(frame_hsv, (0, 100, 80), (15, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    high_thresh, _ = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5 * high_thresh
    edges = cv2.Canny(mask, lowThresh, high_thresh)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, )
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 100:
            # cv2.drawContours(frame, contours[i], -1, (8, 255, 8), 3)
            M = cv2.moments(contours[i])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return [cx, cy]
            # cv2.putText(frame, str(i) + '_(' + str(cx) + ':' + str(cy) + ')', (cx + 10, cy - 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    # return frame