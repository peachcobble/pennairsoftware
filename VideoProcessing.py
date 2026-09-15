# import libraries
import os
import time
import cv2 as cv
import numpy as np
import process_frame as pf

def process_video(frame_proc_func, input_video_path, video_folder_path, output_video_name, kSize, thresh_multiplier=None, min_contour_area=None):
    # timer to see how much time it takes to process the video
    start_time = time.time()

    # set up file paths for input and output

    # make sure this folder exists, creates if it doesn't
    os.makedirs(video_folder_path, exist_ok=True)

    # output video path with filename and extension
    output_video_path = os.path.join(video_folder_path, output_video_name)

    # import input video
    video = cv.VideoCapture(input_video_path)
    # check if video was opened successfully
    if not video.isOpened():
        raise FileNotFoundError(f'could not open video: {input_video_path}')

    # retrieves the fps or defaults to 30.0 
    fps = video.get(cv.CAP_PROP_FPS) or 30.0
    # retrieves the width and height of the video frames
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    # prints values for checking
    print(f"width: {width}, height: {height}, fps: {fps}")

    # specifices H.264 codec for video compression
    fourcc = cv.VideoWriter_fourcc(*'avc1')
    # initalizes the video writer itself
    writer = cv.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

    # frame counter to act as a 'progress bar'
    frame_count = 0
    try:
        while True:
            # increment frame counter
            frame_count += 1
            # print frame number to console and replacing the last line
            # to see the latest number of frames processed
            print(f"\rFrame: {frame_count}", end="", flush=True)
            # reads the frame and a bool to check if there is a next frame to read
            still_running, frame = video.read()
            # if there is no next frame, break the while loop
            if not still_running:
                break

            # process the frame and output the frame with contour edges and center labelling
            processed_frame = frame_proc_func(frame, kSize, thresh_multiplier, min_contour_area)

            # ensure size matches writer settings
            processed_frame_h, processed_frame_w = processed_frame.shape[:2]
            # if the size doesn't match, resize to match
            if (processed_frame_w, processed_frame_h) != (width, height):
                processed_frame = cv.resize(processed_frame, (width, height))

            # write the processed frame to the output video
            writer.write(processed_frame)
    finally:
        # release the video and writer objects
        video.release()
        writer.release()
        # ends timer
        end_time = time.time()
        # calculates and prints the elapsed time to process the video
        print(f"\nTotal time: {end_time - start_time:.2f} seconds")
