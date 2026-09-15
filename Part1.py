# import libraries
import cv2 as cv
import numpy as np
import time
import adaptive_bilateral as ab

# used as a starting point:
# https://opencv.org/edge-detection-using-opencv/

# timer to see how much time it takes to process the img
start_time = time.time()

# function for using three channel adaptive bilateral(ab) filtering
def three_channel_ab(img):
    # split the image into bgr channels
    b, g, r = cv.split(img)
    # collect outputs of channels filtered with ab filtering
    out_b = ab.adaptive_bilateral_filter(b, b.copy(), np.full_like(b, 30.0), rho=5)
    out_g = ab.adaptive_bilateral_filter(g, g.copy(), np.full_like(g, 30.0), rho=5)
    out_r = ab.adaptive_bilateral_filter(r, r.copy(), np.full_like(r, 30.0), rho=5)
    # merge the filtered channels back into a single image
    return cv.merge([out_b, out_g, out_r])

# define the general folder path for output images
img_folder_path = 'Part1/'
# define the input image path with filename and extension
input_img_path = 'resources/PennAir 2024 App Static.png'

# import image
img = cv.imread(input_img_path, cv.IMREAD_COLOR)
# output image
output_img = img.copy()

# convert image to grayscale in uint8 format
img_gray_u8 = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

# pass image through high pass filter (hpf) 
# to help edge detection
# according to opencv documentation !

# i was unsatisifed with the results of just
# using gaussian blur
# found bilateral filtering through: 
# https://www.youtube.com/watch?v=LjbYKWAQA5s
# i wanted automatic values for the parameters
# so i wanted to use adaptive bilateral filtering
# https://stackoverflow.com/questions/45137319/adaptive-bilateral-filter-in-opencv-python2-7-implimentation
# https://arxiv.org/pdf/1811.02308
# i generated adaptive_bilateral.py from the paper
# found through the stackoverflow link above
# later when i found faster bilateral filtering
# i found that it was significantlyfaster than adaptive 
# bilateral filtering, but it was not as great
# at smoothing still images
# hence why i used a different algo for parts 2-4

# ab filtering applied
img_blur = three_channel_ab(img)

# convert to u8 
img_blur = np.clip(img_blur, 0, 255).astype(np.uint8)

# canny thresholds, hysteresis thresholding, through otsu method
# https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
otsu_thresh_val, thresh_img = cv.threshold(
    img_gray_u8, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
higher_thresh = otsu_thresh_val
lower_thresh = higher_thresh * 0.5

# canny edge detection
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

# draw contours on a white background to better visualize sometimes
contours_on_white = np.ones_like(img) * 255
cv.drawContours(contours_on_white, contours, -1, (0, 255, 0), 2)

# iterate through contours
for contour in contours:
    # find centers of each
    # and draw them on the output image
    # and label coordinates
    M = cv.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv.circle(output_img, (cX, cY), 2, (255, 255, 255), -1)
        cv.putText(output_img, f"({cX}, {cY})", (cX - 35, cY + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# outputs, excess outputs commented out to reduce extra outputs
# cv.imwrite(img_folder_path + 'part1output_gray.jpg', np.clip(img_gray_u8, 0, 255).astype(np.uint8))
# cv.imwrite(img_folder_path + 'part1output_blur.jpg', np.clip(img_blur, 0, 255).astype(np.uint8))
# cv.imwrite(img_folder_path + 'part1output_edges.jpg', np.clip(img_canny_edges, 0, 255).astype(np.uint8))
# cv.imwrite(img_folder_path + 'part1output_contours.jpg', np.clip(contours_on_white, 0, 255).astype(np.uint8))
cv.imwrite(img_folder_path + 'part1output_img.jpg', np.clip(output_img, 0, 255).astype(np.uint8))

end_time = time.time()
# calculates and prints the elapsed time to process the video
print(f"\nTotal time: {end_time - start_time:.2f} seconds")