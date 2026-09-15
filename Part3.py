import VideoProcessing as vp
import process_frame as pf

# parameters for process_frame
kSize = 45
min_contour_area = 8000
thresh_multiplier = 0.5

# function used to process each frame
frame_proc_func = pf.process_frame

# general video output folder path
video_folder_path = 'Part3/'

# input video path with filename and extension
input_video_path = 'resources/PennAir 2024 App Dynamic Hard.mp4'

# output video filename and extension
output_video_name = 'part3output.mp4'

vp.process_video(frame_proc_func, input_video_path, video_folder_path, output_video_name,
                  kSize, thresh_multiplier, min_contour_area)