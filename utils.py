import os
import json
import numpy
import math
import cv2
from collections import defaultdict
import dlib
from imutils import face_utils
import imutils

MODEL_SAVE_LOCATION = "model"
MODEL_BIN_FILE = "face_recognition.bin"
MODEL_LABEL_FILE = "face_recognition_labels.json"

def save_model(recognizer, labels):
    if not os.path.isdir(MODEL_SAVE_LOCATION):
        os.mkdir(MODEL_SAVE_LOCATION, 0755)
    recognizer.save(os.path.join(MODEL_SAVE_LOCATION, MODEL_BIN_FILE))
    with open(os.path.join(MODEL_SAVE_LOCATION, MODEL_LABEL_FILE), "wb") as f:
        json.dump(labels, f)

def load_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(MODEL_SAVE_LOCATION, MODEL_BIN_FILE))
    with open(os.path.join(MODEL_SAVE_LOCATION, MODEL_LABEL_FILE), "rb") as f:
        labels = json.load(f)
    return recognizer, labels

def get_names_faces_dict(directory):
    name_face_map = defaultdict(list)
    for dir_name in os.listdir(directory):
        full_dir = os.path.join(directory, dir_name)
        if os.path.isdir(full_dir):
            names = os.listdir(full_dir)
            if "label.txt" in names:
                with open(os.path.join(full_dir, "label.txt"), 'r') as f:
                    label = f.read().strip()
                for filename in sorted(names):
                    if filename == 'label.txt':
                        continue
                    if not filename.endswith(".png"):
                        continue
                    fullname = os.path.join(full_dir, filename)
                    img = cv2.imread(fullname, 0)
                    if img is None:
                        print 'img was none'
                        continue
                    refined = refine_image(img)
                    if refined is None:
                        'rejected'
                        continue
                    name_face_map[label].append(img)
                    assert img is not None, fullname
    return name_face_map

def format_table(stats):
    table = [
        "   | T | F",
        "-----------",
        " P |{tp}|{fp}",
        "-----------",
        " N |{tn}|{fn}",
    ]
    table = "\n".join(table)
    vals = {}
    for key, value in stats.items():
        as_str = str(value)
        if len(as_str) == 1:
            as_str = " %s " % as_str
        elif len(as_str) == 2:
            as_str = " %s" % as_str
        elif len(as_str) == 3:
            as_str = "%s" % as_str
        vals[key] = as_str
    return table.format(**vals)

def format_graph(stats):
    graph = [
        "TP: {tp}",
        "TN: {tn}",
        "FP: {fp}",
        "FN: {fn}",
    ]
    graph = "\n".join(graph)
    graph_stats = {}
    for key, val in stats.items():
        graph_stats[key] = "#"*val

    return graph.format(**graph_stats)

def translate_to_center_relative(dims, coordinate):
    x, y = dims
    x_c = x/2.0
    y_c = y/2.0
    #print 'center', x_c, y_c
    return (coordinate[0]-x_c, coordinate[1]-y_c)

def translate_to_topleft_relative(dims, coordinate):
    x, y = dims
    x_c = x/2.0
    y_c = y/2.0
    #print 'center', x_c, y_c
    return (coordinate[0]+x_c, coordinate[1]+y_c)

def translate_rotation(original_dims, new_dims, rotation, coordinates):
    cr_coordinates = numpy.array([translate_to_center_relative(original_dims, coordinate) for coordinate in coordinates])

    theta = numpy.radians(rotation)
    c, s = numpy.cos(theta), numpy.sin(theta)
    R = numpy.array(((c, -s), (s, c)))

    rot_coordiantes = numpy.matmul(cr_coordinates, R)

    not_cr_coordinates = numpy.array([translate_to_topleft_relative(new_dims, coord) for coord in rot_coordiantes])

    return not_cr_coordinates

FEATURE_PREDICTOR = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def refine_image(image):
    x_len, y_len = image.shape[:2]
    image_rect = dlib.rectangle(left=0, top=0, right=x_len, bottom=y_len)

    image_shape = image.shape
    shape = FEATURE_PREDICTOR(image, image_rect)
    shape = face_utils.shape_to_np(shape)

    left_avg = get_feature_average(shape, "left_eye")
    right_avg = get_feature_average(shape, "right_eye")

    eye_difference  = right_avg - left_avg
    rads = math.atan(eye_difference[1]/eye_difference[0])
    degrees = math.degrees(rads)

    orig_dims = image.shape[0:2]
    rot_image = imutils.rotate(image, degrees)

    new_dims = rot_image.shape[:2]

    rot_shape = translate_rotation(orig_dims, new_dims, degrees, shape)
    rot_shape = numpy.array([(int(x[0]), int(x[1])) for x in rot_shape])

    jaw_eye_ratio = calculate_eye_jaw_sides_ratio(rot_shape)
    if abs(1-jaw_eye_ratio) > 0.15:
        return None

    first, last = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
    jaw = rot_shape[first:last]
    left_jaw = jaw[0]
    right_jaw = jaw[-1]

    max_right = rot_image.shape[1]
    max_down = rot_image.shape[1]

    max_down = min(max_down, max([y for x, y in jaw]))

    first, last = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    top_most = min([x[1] for x in rot_shape[first:last]])
    first, last = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    top_most2 = min([x[1] for x in rot_shape[first:last]])

    left =  max(int(left_jaw[0]), 0)
    right = min(int(right_jaw[0]), max_right)
    top = min(top_most, top_most2)
    bottom = max_down
    final = rot_image[top:bottom, left:right]

    return final

def get_feature_average(shape, feature_name):
    i, j = face_utils.FACIAL_LANDMARKS_IDXS[feature_name]
    locs = shape[i:j]
    return locs.mean(axis=0)

def calculate_eye_jaw_sides_ratio(shape):
    # Only uses x component, so should probably be done after leveling.
    # or changed to use full xy distance

    # Calc ratio between eye to jaw left and right sides.
    first, last = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
    jaw = shape[first:last]
    left_jaw = jaw[0]
    right_jaw = jaw[-1]

    left_avg = get_feature_average(shape, "left_eye")

    right_avg = get_feature_average(shape, "right_eye")

    jaw_eye_ratio = (right_avg[0]-right_jaw[0])/(left_jaw[0]-left_avg[0])

    return jaw_eye_ratio


