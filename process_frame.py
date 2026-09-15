import cv2 as cv
import numpy as np
import adaptive_bilateral as ab

# helper functions for processing frames
# for part 2
# use only process_frame_pn() when n is a num 
# other two are tested and not used in final version

def process_frame(img, kSize, thresh_multiplier, min_contour_area):
    # rolling guided filter
    def rolling_guided(img):
        # 0.100
        return cv.ximgproc.rollingGuidanceFilter(src=img, d=8, sigmaColor=25, sigmaSpace=3, numOfIter=4)

    output_img = img.copy()

    # convert image to grayscale
    img_gray_u8 = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)


    ###
    # pass image through high pass filter (hpf)
    # to help edge detection
    ###

    # rolling guided filter to smooth grass texture but keep shape edges
    img_blur = rolling_guided(img)

    # canny thresholds, hysteresis thresholding, through otsu method
    # https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
    otsu_thresh_val, thresh_img = cv.threshold(
        img_gray_u8, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    higher_thresh = otsu_thresh_val
    lower_thresh = higher_thresh * thresh_multiplier

    # canny edge detection
    def canny_channel(channel_u8, lower, higher):
        return cv.Canny(channel_u8, lower, higher)

    img_canny_edges = cv.Canny(img_blur, lower_thresh, higher_thresh)

    # morphing
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kSize, kSize))
    img_canny_edges = cv.morphologyEx(img_canny_edges, cv.MORPH_CLOSE, kernel)

    # invert canny edges 
    inverted_edges = cv.bitwise_not(img_canny_edges)

    # find contours
    contours, hierarchy = cv.findContours(inverted_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # clean up small noise 
    contours = [c for c in contours if cv.contourArea(c) > min_contour_area]

    # outline found contours
    cv.drawContours(output_img, contours, -1, (255, 255, 255), 2)

    for contour in contours:
        M = cv.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.circle(output_img, (cX, cY), 2, (255, 255, 255), -1)
            cv.putText(output_img, f"({cX}, {cY})", (cX - 35, cY + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return np.clip(output_img, 0, 255).astype(np.uint8)

def process_frame_p4(img, kSize, thresh_multiplier, min_contour_area):
    # rolling guided filter
    def rolling_guided(img):
        # 0.100
        return cv.ximgproc.rollingGuidanceFilter(src=img, d=-1, sigmaColor=25, sigmaSpace=3, numOfIter=4)

    def calculate_depth(K, digital_length, real_length):
        # camera focal length
        fx = K[0][0]

        depth = (fx * real_length) / digital_length
        return depth

    def calculate_x_coord(K, x_pixel, depth):
        # camera principal point x coord
        cx = K[0][2]
        # camera focal length
        fx = K[0][0]
        # calculate x coordinate in inches
        x_coord = depth * (x_pixel - cx) / fx
        return x_coord

    def calculate_y_coord(K, y_pixel, depth):
        # camera principal point y coord
        cy = K[1][2]
        # camera focal length
        fy = K[1][1]
        # calculate y coordinate in inches
        y_coord = depth * (y_pixel - cy) / fy
        return y_coord

    # intrinsic camera matrix K
    K = [[2564.3186869, 0, 0],
	[0, 2569.70273111, 0],
	[0,			      0, 1]]

    # real circle radius in inches
    REAL_CIRCLE_RADIUS = 10.0

    output_img = img.copy()

    # convert image to grayscale
    img_gray_u8 = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)


    ###
    # pass image through high pass filter (hpf)
    # to help edge detection
    ###

    # rolling guided filter to smooth grass texture but keep shape edges
    img_blur = rolling_guided(img)

    # canny thresholds, hysteresis thresholding, through otsu method
    # https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
    otsu_thresh_val, thresh_img = cv.threshold(
        img_gray_u8, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    higher_thresh = otsu_thresh_val
    lower_thresh = higher_thresh * thresh_multiplier

    # canny edge detection
    def canny_channel(channel_u8, lower, higher):
        return cv.Canny(channel_u8, lower, higher)

    img_canny_edges = cv.Canny(img_blur, lower_thresh, higher_thresh)

    # morphing
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kSize, kSize))
    img_canny_edges = cv.morphologyEx(img_canny_edges, cv.MORPH_CLOSE, kernel_close)

    # invert canny edges 
    inverted_edges = cv.bitwise_not(img_canny_edges)

    # find contours
    contours, hierarchy = cv.findContours(inverted_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # clean up small noise 
    contours = [c for c in contours if cv.contourArea(c) > min_contour_area]

    # outline found contours
    cv.drawContours(output_img, contours, -1, (255, 255, 255), 2)

    most_circular_contour = None
    highest_circularity = 0.0
    depth = 0.0

    for contour in contours:
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        if circularity > highest_circularity:
            highest_circularity = circularity
            most_circular_contour = contour

    if most_circular_contour is not None:
        (x, y), digital_radius = cv.minEnclosingCircle(most_circular_contour)
        depth = calculate_depth(K, digital_radius, REAL_CIRCLE_RADIUS)

        for contour in contours:
            M = cv.moments(contour)

            if M["m00"] != 0:
                # retrieve the center pixel coordinates of the contour
                center_px_x = int(M["m10"] / M["m00"])
                center_px_y = int(M["m01"] / M["m00"])
                # calculate real world coordinates wrt to camera based on real depth
                real_x_coord = round(calculate_x_coord(K, center_px_x, depth))
                real_y_coord = round(calculate_y_coord(K, center_px_y, depth))
                # keep 2 decimal places for depth
                depth = round(depth, 2)
                # draw the center point
                cv.circle(output_img, (center_px_x, center_px_y), 2, (255, 255, 255), -1)
                # write the coordinates: x, y, and depth
                cv.putText(output_img, f"({real_x_coord}, {real_y_coord}, depth: {depth})", (center_px_x - 35, center_px_y + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return output_img

def process_frametestgauss(img):
    #gauss pass 1
    output_img = img.copy()

    # convert image to grayscale
    img_gray_u8 = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)


    ###
    # pass image through high pass filter (hpf)
    # to help edge detection
    ###

    # gaussian blur to reduce noise
    img_blur = cv.GaussianBlur(img_gray_u8, (5, 5), 0)

    # canny thresholds, hysteresis thresholding, through otsu method
    # https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
    otsu_thresh_val, thresh_img = cv.threshold(
        img_gray_u8, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    higher_thresh = otsu_thresh_val
    lower_thresh = higher_thresh * 0.5

    # canny edge detection
    def canny_channel(channel_u8, lower, higher):
        return cv.Canny(channel_u8, lower, higher)

    img_canny_edges = cv.Canny(img_blur, lower_thresh, higher_thresh)

    # morphing
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
    img_canny_edges = cv.morphologyEx(img_canny_edges, cv.MORPH_CLOSE, kernel)

    # invert canny edges 
    inverted_edges = cv.bitwise_not(img_canny_edges)

    # find contours
    contours, hierarchy = cv.findContours(inverted_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # clean up small noise 
    contours = [c for c in contours if cv.contourArea(c) > 3000]

    # outline found contours
    cv.drawContours(output_img, contours, -1, (255, 255, 255), 2)

    contours_on_white = np.ones_like(img) * 255
    cv.drawContours(contours_on_white, contours, -1, (0, 255, 0), 2)

    for contour in contours:
        M = cv.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.circle(output_img, (cX, cY), 2, (255, 255, 255), -1)
            cv.putText(output_img, f"({cX}, {cY})", (cX, cY + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return output_img

def process_frametestab(img):
    #ab single channel
    output_img = img.copy()
    ###
    # pass image through high pass filter (hpf)
    # to help edge detection
    ###

    # bilateral filter to smooth grass texture but keep shape edges
    # https://www.youtube.com/watch?v=LjbYKWAQA5s
    # img_blur = cv.bilateralFilter(img_gray, 9, 255, 150)
    # https://stackoverflow.com/questions/45137319/adaptive-bilateral-filter-in-opencv-python2-7-implimentation
    # https://arxiv.org/pdf/1811.02308
    # convert image to grayscale
    img_gray_u8 = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    img_gray = img_gray_u8.astype(np.float64)
    theta = img_gray.copy()
    sigma = np.full_like(img_gray, 30.0)
    img_blur = ab.adaptive_bilateral_filter(img_gray, theta, sigma, rho=5)
    img_blur = np.clip(img_blur, 0, 255).astype(np.uint8)

    # canny thresholds, hysteresis thresholding, through otsu method
    # https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
    otsu_thresh_val, thresh_img = cv.threshold(
        img_gray_u8, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    higher_thresh = otsu_thresh_val
    lower_thresh = higher_thresh * 0.5

    # canny edge detection
    def canny_channel(channel_u8, lower, higher):
        return cv.Canny(channel_u8, lower, higher)

    img_canny_edges = cv.Canny(img_blur, lower_thresh, higher_thresh)

    # morphing
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
    img_canny_edges = cv.morphologyEx(img_canny_edges, cv.MORPH_CLOSE, kernel)

    # invert canny edges 
    inverted_edges = cv.bitwise_not(img_canny_edges)

    # find contours
    contours, hierarchy = cv.findContours(inverted_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # clean up small noise 
    contours = [c for c in contours if cv.contourArea(c) > 3000]

    # outline found contours
    cv.drawContours(output_img, contours, -1, (255, 255, 255), 2)

    contours_on_white = np.ones_like(img) * 255
    cv.drawContours(contours_on_white, contours, -1, (0, 255, 0), 2)

    for contour in contours:
        M = cv.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.circle(output_img, (cX, cY), 2, (255, 255, 255), -1)
            cv.putText(output_img, f"({cX}, {cY})", (cX, cY + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return np.clip(output_img, 0, 255).astype(np.uint8)