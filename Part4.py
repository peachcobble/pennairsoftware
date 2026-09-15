import VideoProcessing as vp
import process_frame as pf

# parameters for process_frame
kSize = 45
min_contour_area = 10000
thresh_multiplier = 0.5

# function used to process each frame
# not abstracted because too much work
# to fit in the center calculation code + depth printing
frame_proc_func = pf.process_frame_p4

# general video output folder path
video_folder_path = 'Part4/'

# input video path with filename and extension
input_video_path = 'resources/PennAir 2024 App Dynamic Hard.mp4'

# output video filename and extension
output_video_name = 'part4output.mp4'

# parameters for process_frame passed through process_video not used hence 0, 0, 0
vp.process_video(frame_proc_func, input_video_path, video_folder_path, output_video_name,
                  kSize, thresh_multiplier, min_contour_area)
