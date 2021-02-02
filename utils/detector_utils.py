
import tensorflow as tf

import cv2
from utils import label_map_util
from utils import alertcheck
import numpy as np


detection_graph = tf.Graph()

TRAINED_MODEL_DIR = 'frozen_graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/output_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/labelmap.pbtxt'

NUM_CLASSES = 1
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

a = b = 0
global x,y,w,z

# Load a frozen infrerence graph into memory
def load_inference_graph():
    # load frozen tensorflow model into memory

    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess

def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)

    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)


def draw_box_on_image(num_cars_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np):

    global a, b
    car_cnt=0
    color0 = (255, 0, 0)
    color1 = (0, 50, 255)

    centeriX=tuple()
    centeriY=tuple()

    for i in range(num_cars_detect):

        if (scores[i] > score_thresh):

            if classes[i] == 1:
                id = 'Car'

            if i == 0:
                color = color0


                (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
                p1 = (int(left), int(top))
                p2 = (int(right), int(bottom))

                cv2.rectangle(image_np, p1, p2, color, 3, 1)

                cv2.putText(image_np, 'Car ' + str(i) + ': ' + id, (int(left), int(top) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.putText(image_np, 'confidence: ' + str("{0:.2f}".format(scores[i])),
                        (int(left), int(top) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                              boxes[i][0] * im_height, boxes[i][2] * im_height)
                centeriX = (left/2,bottom/ 2)

            else:
                color=color1
                (left1, right1, top1, bottom1) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                              boxes[i][0] * im_height, boxes[i][2] * im_height)
                p3 = (int(left1), int(top1))
                p4 = (int(right1), int(bottom1))

                cv2.rectangle(image_np, p3, p4, color, 3, 1)

                cv2.putText(image_np, 'Car ' + str(i) + ': ' + id, (int(left1), int(top1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.putText(image_np, 'confidence: ' + str("{0:.2f}".format(scores[i])),
                            (int(left1), int(top1) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                centeriY = (left1/2,bottom1/ 2)


    a=alertcheck.drawboxtosafeline(centeriX,centeriY)

    if car_cnt == 0:
        b = 0
    else:
        b = 1
    return a, b
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)




