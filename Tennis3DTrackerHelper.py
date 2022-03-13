import os
import cv2
import numpy as np
from TrackNet.Code_Python3.TrackNet_Three_Frames_Input import Models


ROYGBIVP_BGR_COLORS = [
    [0, 0, 255],
    [0, 127, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [130, 0, 75],
    [211, 0, 148],
    [203, 192, 255]
]


def get_tracknet(width, height):
    tracknet_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TrackNet/Code_Python3/TrackNet_Three_Frames_Input/weights/model.3')
    tracknet = Models.TrackNet.TrackNet(256, input_height=height, input_width=width)
    tracknet.compile(loss='categorical_crossentropy', optimizer= 'adadelta' , metrics=['accuracy'])
    tracknet.load_weights(tracknet_weights_path)
    return tracknet


def setup_input_output_videos(input_video_path, output_video_path):
    # Get info about input video
    input_video = cv2.VideoCapture(input_video_path)
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup output video with same parameters
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))

    return input_video, output_video, fps, (video_width, video_height)


def image_to_float_and_reshape(image, dimensions):
    return cv2.resize(image, dimensions).astype(np.float32)


def predict_2D_ball_position(m, image_buffer, video_dimensions, tracknet_dimensions):
    video_width, video_height = video_dimensions
    tracknet_width, tracknet_height = tracknet_dimensions

    # Mostly copied code from tracknet...

    # image_buffer goes oldest --> newest, but tracknet expects newest --> oldest when stacked
    # combine three imgs to  (width , height, rgb*3)
    X = np.concatenate((image_buffer[2], image_buffer[1], image_buffer[0]), axis=2)
    # since the odering of TrackNet  is 'channels_first', so we need to change the axis
    X = np.rollaxis(X, 2, 0)
	# prdict heatmap
    pr = m.predict( np.array([X]) )[0]
    #since TrackNet output is ( net_output_height*model_output_width , n_classes )
	#so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
	#.argmax( axis=2 ) => select the largest probability as class
    pr = pr.reshape(( tracknet_height ,  tracknet_width , 256 ) ).argmax( axis=2 )

	#cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
    pr = pr.astype(np.uint8) 

	#reshape the image size as original input image
    heatmap = cv2.resize(pr  , (video_width, video_height ))

	#heatmap is converted into a binary image by threshold method.
    _, heatmap = cv2.threshold(heatmap,127,255,cv2.THRESH_BINARY)

	#find the circle in image with 2<=radius<=7
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=50,param2=2,minRadius=2,maxRadius=7)

    if circles is not None and len(circles) == 1:
        return (int(circles[0][0][0]), int(circles[0][0][1]))
    else:
        return None


class OpenCVInputVideoWrapper:
    def __init__(self, video):
        self._video = video
        self.currentFrame = 0
        self.totalFrames = self._video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.atEndOfVideo = self.totalFrames == 0

    def read_next(self):
        if self.atEndOfVideo:
            raise Exception('Read past end of video')
        
        self._video.set(cv2.CAP_PROP_POS_FRAMES, self.currentFrame)
        status, image = self._video.read()
        if not status:
            raise Exception('Error occurred while reading video')
        
        self.currentFrame += 1
        self.atEndOfVideo = self.currentFrame >= self.totalFrames

        return image


class OpenCVOutputVideoWrapper:
    def __init__(self, video, dimensions):
        self._video = video
        self.totalFrames = 0
        self.dimensions = dimensions

    def write_next(self, image):
        resized_image = cv2.resize(image, self.dimensions)
        self._video.write(resized_image)
        self.totalFrames += 1
