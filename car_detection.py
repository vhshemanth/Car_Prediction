import cv2
import argparse
import datetime

import numpy as np
from utils import detector_utils

lst1=[]
lst2=[]

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())


detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.70
    cap = cv2.VideoCapture('cars1.mp4')
    Orientation = 'bt'
    Line_Perc1 = float(15)
    Line_Perc2 = float(30)

    # max number of cars we want to detect/track
    num_cars_detect =2

    start_time = datetime.datetime.now()

    # Used to calculate fps
    num_frames = 0

    im_height, im_width = (None,None)


    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            frame = np.array(frame)
            if im_height == None:
                im_height, im_width = frame.shape[:2]
            # Run image through tensorflow graph
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # Draw bounding boxeses and text
            detector_utils.draw_box_on_image(num_cars_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame)

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time


            detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)


            #Display the resulting frame
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

