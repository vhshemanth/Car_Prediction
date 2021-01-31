import cv2
import argparse
# import orien_lines
import datetime
# from imutils.video import VideoStream
# from utils import detector_utils as detector_utils
# import pandas as pd
from datetime import date
import xlrd
from xlwt import Workbook
from xlutils.copy import copy
import numpy as np
from utils import detector_utils
from utils.detector_utils import detection_graph

lst1=[]
lst2=[]


detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.70
    cap = cv2.VideoCapture('cars1.mp4')
    Orientation = 'bt'
    Line_Perc1 = float(15)
    Line_Perc2 = float(30)

    # max number of cars we want to detect/track
    num_cars_detect = 2

    # Used to calculate fps
    num_frames = 0

    im_height, im_width = (None,None)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if im_height == None:
            im_height, im_width = frame.shape[:2]
        if ret == True:
            frame = np.array(frame)
            # Run image through tensorflow graph
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # Draw bounding boxeses and text
            detector_utils.draw_box_on_image(num_cars_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame)
            # lst1.append(a)
            # lst2.append(b)

            # Calculate Frames per second (FPS)
            num_frames += 1

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
