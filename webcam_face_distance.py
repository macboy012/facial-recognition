import dlib
import os
import sys
import numpy as np
import cv2
from collections import OrderedDict
from imutils import face_utils
import time
import utils
import face_recognition


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def facefeature(image):
    rects = detector(image, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
        #print shape

        #face_locations = face_recognition.face_locations(image)
        #print 'face locations', face_locations
        #print 'rects', rects
        #print 'rect', rect
        new_rect = rect.top(), rect.right(), rect.bottom(), rect.left()
        face_encodings = face_recognition.face_encodings(image, [new_rect], num_jitters=10)
        #print face_encodings

        dist = face_recognition.face_distance(face_encodings, encoding)
        if len(face_encodings) == 0:
            continue

        use_dist = None
        if len(dist) > 0:
            use_dist = sorted(dist)[0]


        """
        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            clone = image.copy()
            print name
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)

            # loop over the subset of facial landmarks, drawing the
            # specific face part
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

                # extract the ROI of the face region as a separate image
                (x, y, w, h) = cv2.boundingRect(numpy.array([shape[i:j]]))
                roi = image[y:y + h, x:x + w]
                print roi.shape
                #roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

                # show the particular face part
                cv2.imshow("ROI", roi)
                cv2.imshow("Image", clone)
                cv2.waitKey(200)
        """
        # visualize all facial landmarks with a transparent overlay
        output = face_utils.visualize_facial_landmarks(image, shape)

        cv2.rectangle(output, (rect.tl_corner().x, rect.tl_corner().y), (rect.br_corner().x, rect.br_corner().y), (0, 255, 0), 4)

        if use_dist is not None:
            draw_xcentered_text(output, str(round(use_dist, 3)), 20)

        cv2.imshow("Image", output)
        cv2.waitKey(1)


#image = cv2.imread('imgs/2018-11-14_17-55-38/97.png')
def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


video_capture = cv2.VideoCapture(0)
def load_from_webcam():
    if not video_capture.isOpened():
        print('Unable to load camera.')
        time.sleep(5)

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    return frame

DRAWING_COLOR = (100,0,255)
def draw_xcentered_text(frame, text, height):
    fheight, fwidth, _ = frame.shape
    #base_font_scale = 6.0
    font_scale = 6.0
    #base_thickness = 10
    thickness = 10
    increment = 0.5
    while True:
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        twidth = size[0][0]
        theight = size[0][1]
        baseline = size[1]
        if twidth <= fwidth * 0.98:
            break
        font_scale -= increment

    x = (fwidth - twidth) / 2

    if height >= 0:
        y = theight+height
    else:
        y = fheight - baseline - theight - height

    if font_scale < 3.5:
        thickness -= 2

    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, DRAWING_COLOR, thickness)
    return frame




compare_file = sys.argv[1]
encoding = utils.compute_image_encoding(os.path.dirname(compare_file), os.path.basename(compare_file))
if encoding is None:
    print 'no face found in comparison image'
    sys.exit(1)

while True:
    facefeature(load_from_webcam())
