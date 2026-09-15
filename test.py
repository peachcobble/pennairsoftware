# import libraries
import cv2 as cv
import numpy as np
import adaptive_bilateral as ab
import time

# https://opencv.org/edge-detection-using-opencv/

# not all these func are used
# they are just tested to see best
# reserved here to retry
# this is a scrappy test file for all parts
# please do not evaluate

def three_channel_canny(img_blur):
    #gauss blur img_blur
    b, g, r = cv.split(img_blur)  # already uint8 from filtering above

    def canny_channel(channel_u8, lower, higher):
        return cv.Canny(channel_u8, lower, higher)

    edges_b = canny_channel(b, lower_thresh, higher_thresh)
    edges_g = canny_channel(g, lower_thresh, higher_thresh)
    edges_r = canny_channel(r, lower_thresh, higher_thresh)

    return cv.bitwise_or(cv.bitwise_or(edges_b, edges_g), edges_r)

def three_channel_ab(img):
    # 8.789
    b, g, r = cv.split(img)
    out_b = ab.adaptive_bilateral_filter(b, b.copy(), np.full_like(b, 30.0), rho=5)
    out_g = ab.adaptive_bilateral_filter(g, g.copy(), np.full_like(g, 30.0), rho=5)
    out_r = ab.adaptive_bilateral_filter(r, r.copy(), np.full_like(r, 30.0), rho=5)
    return cv.merge([out_b, out_g, out_r])

def single_channel_ab(img_gray):
    # 3.048
    #img_gray should be float64
    theta = img_gray.copy()
    sigma = np.full_like(img_gray, 30.0)
    return ab.adaptive_bilateral_filter(img_gray, theta, sigma, rho=5)

def guided_filter(img):
    # 0.100
# Parameters:
    # radius (r): Controls the window size (larger = more smoothing)
    # eps (ε): Controls the edge preservation (larger = acts more like a standard blur)
    radius = 8
    eps = 0.02 * 255 * 255  # Regularization parameter squared for 8-bit images

    # Apply Guided Filter
    # Syntax: cv2.ximgproc.guidedFilter(guide, src, radius, eps)
    return cv.ximgproc.guidedFilter(guide=img, src=img, radius=radius, eps=eps)

def faster_bilateral(img):
    # 0.079
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv.cvtColor(np.array(img), cv.COLOR_BGRA2BGR)
    elif img.ndim == 2:
        img = cv.cvtColor(np.array(img), cv.COLOR_GRAY2BGR)

    return cv.bilateralFilter(img, d=9, sigmaColor=150, sigmaSpace=150)

def rolling_guided(img):
    # 0.100
    return cv.ximgproc.rollingGuidanceFilter(src=img, d=-1, sigmaColor=25, sigmaSpace=3, numOfIter=4)

img_folder_path = 'test_outputs/Imgs/'
input_img_path = 'resources/part3test1.png'

# start timing
start_time = time.perf_counter()

# import image
img = cv.imread(input_img_path, cv.IMREAD_COLOR)
# output image
output_img = img.copy()

# convert image to grayscale
img_gray_u8 = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

###
# pass image through high pass filter (hpf)
# to help edge detection
###

# gaussian blur to reduce noise
# img_blur = cv.GaussianBlur(img_gray_u8, (5, 5), 0)

# bilateral filter to smooth grass texture but keep shape edges
# https://www.youtube.com/watch?v=LjbYKWAQA5s

# img_blur = cv.bilateralFilter(img_gray_u8y, 9, 255, 150)

# https://stackoverflow.com/questions/45137319/adaptive-bilateral-filter-in-opencv-python2-7-implimentation
# https://arxiv.org/pdf/1811.02308
img_gray_f64 = cv.cvtColor(img, cv.COLOR_BGRA2GRAY).astype(np.float64)

# img_blur = three_channel_ab(img)
# img_blur = single_channel_ab(img_gray_f64)
# img_blur = guided_filter(img)
# img_blur = faster_bilateral(img)
img_blur = rolling_guided(img)
# img_blur = np.clip(img_blur, 0, 255).astype(np.uint8)

# canny thresholds, hysteresis thresholding, through otsu method
# https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
otsu_thresh_val, thresh_img = cv.threshold(
    img_gray_u8, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
higher_thresh = otsu_thresh_val
lower_thresh = higher_thresh * 0.1

# img_canny_edges = three_channel_canny(img_blur)
# canny edge detection
img_canny_edges = cv.Canny(img_blur, lower_thresh, higher_thresh)

# morphing
# kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
# img_kernel_open = cv.morphologyEx(img_canny_edges, cv.MORPH_OPEN, kernel_open)

# kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (41, 41))
# img_kernel_close = cv.morphologyEx(img_canny_edges, cv.MORPH_CLOSE, kernel_close)

# img_canny_edges = cv.subtract(img_kernel_close,img_kernel_open)

kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (41, 41))
img_canny_edges = cv.morphologyEx(img_canny_edges, cv.MORPH_CLOSE, kernel_close)

# invert canny edges 
inverted_edges = cv.bitwise_not(img_canny_edges)

# find contours
contours, hierarchy = cv.findContours(inverted_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# clean up small noise 
contours = [c for c in contours if cv.contourArea(c) > 8000]

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
        cv.putText(output_img, f"({cX}, {cY})", (cX - 35, cY + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

cv.imwrite(img_folder_path + 'output_gray.jpg', np.clip(img_gray_u8, 0, 255).astype(np.uint8))
cv.imwrite(img_folder_path + 'output_blur.jpg', np.clip(img_blur, 0, 255).astype(np.uint8))
cv.imwrite(img_folder_path + 'output_edges.jpg', np.clip(img_canny_edges, 0, 255).astype(np.uint8))
cv.imwrite(img_folder_path + 'output_contours.jpg', np.clip(contours_on_white, 0, 255).astype(np.uint8))
cv.imwrite(img_folder_path + 'output_img.jpg', np.clip(output_img, 0, 255).astype(np.uint8))

# end timing and report
end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.3f} seconds")
