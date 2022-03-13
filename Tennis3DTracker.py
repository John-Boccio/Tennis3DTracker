import argparse
import os
import logging
import collections
import cv2

import Tennis3DTrackerHelper

# Set up logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] : %(message)s', level=logging.DEBUG)


# Get arguments and validate them
parser = argparse.ArgumentParser(description='Predict 3D positions of a tennis ball during a video of a tennis match point')
parser.add_argument('--input_video', type=str, required=True, help='Path to an input mp4 video file of a tennis match point')
parser.add_argument('--output_video', type=str, required=False, help='Output path of mp4 with predictions, defaults to <input_video_path>_3D.mp4')

args = parser.parse_args()

def file_exists(file_path):
    exists = os.path.exists(file_path)
    if not exists:
        logging.warning(f'File {file_path} does not exist')
    return exists

def file_extension_matches(file_path, extension):
    file_name, file_ext = os.path.splitext(file_path)
    if not (matches := (file_ext == extension)):
        logging.warning(f'{file_name + file_ext} does not have extension {extension}')
    return matches

input_video_path = args.input_video
assert file_exists(input_video_path)
input_video_path_no_extension, input_video_path_extension = os.path.splitext(input_video_path)
assert file_extension_matches(input_video_path, '.mp4')

output_video_path = args.output_video
if output_video_path is None:
    output_video_path = input_video_path_no_extension + '_3DPredictions.mp4'
output_video_path_no_extension, output_video_path_extension = os.path.splitext(output_video_path)
assert file_extension_matches(output_video_path, '.mp4')

logging.debug(f'Successfully retrieved input mp4 {input_video_path} and output mp4 {output_video_path}')


# Create TrackNet model
tracknet_dimensions = (640, 360)
tracknet = Tennis3DTrackerHelper.get_tracknet(*tracknet_dimensions)

logging.debug(f'Successfully loaded tracknet with weights')


# Get input video and output video which we will create
input_video, output_video, fps, video_dimensions = Tennis3DTrackerHelper.setup_input_output_videos(input_video_path, output_video_path)
input_video = Tennis3DTrackerHelper.OpenCVInputVideoWrapper(input_video)
output_video = Tennis3DTrackerHelper.OpenCVOutputVideoWrapper(output_video, video_dimensions)

# TrackNet uses 3 consecutive frames to predict where the ball is so the first 2 frames have no prediction
assert input_video.totalFrames >= 3, f'Input video {input_video_path} is does not have the minimum number of frames ({input_video.totalFrames} <= 3)'

image_buffer = collections.deque(maxlen=3)
for _ in range(2):
    image = input_video.read_next()
    output_video.write_next(image)
    image = Tennis3DTrackerHelper.image_to_float_and_reshape(image, tracknet_dimensions)
    image_buffer.append(image)

ball_position_buffer = collections.deque(maxlen=15)
while not input_video.atEndOfVideo:
    image = input_video.read_next()
    out_image = image
    image = Tennis3DTrackerHelper.image_to_float_and_reshape(image, tracknet_dimensions)
    image_buffer.append(image)

    ball_position = Tennis3DTrackerHelper.predict_2D_ball_position(tracknet, image_buffer, video_dimensions, tracknet_dimensions)
    ball_position_buffer.append(ball_position)
    for i, position in enumerate(ball_position_buffer):
        if position is None:
            continue
        color = Tennis3DTrackerHelper.ROYGBIVP_BGR_COLORS[i % len(Tennis3DTrackerHelper.ROYGBIVP_BGR_COLORS)]
        out_image = cv2.circle(out_image, position, radius=2, color=color, thickness=2)
    output_video.write_next(out_image)
    logging.debug(f'Ball position in frame[{input_video.currentFrame}] = {ball_position}')
