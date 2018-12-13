import os
import json
import numpy as np
import math
import cv2
from collections import defaultdict
import dlib
from imutils import face_utils
import imutils
import face_recognition

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
        if len(key) == 2:
            graph_stats[key] = "#"*val

    return graph.format(**graph_stats)




MATCH_DATA_DIR = "match_data"
def get_people_list():
    with open(os.path.join(MATCH_DATA_DIR, "people.json"), "rb") as f:
        people = json.load(f)
    return people

def write_people_list(people):
    with open(os.path.join(MATCH_DATA_DIR, "people.json"), "wb") as f:
        json.dump(people, f)

def read_match_data():
    all_data = {}
    for dir_name in os.listdir(MATCH_DATA_DIR):
        full_dir = os.path.join(MATCH_DATA_DIR, dir_name)
        if os.path.isdir(full_dir):
            person_info = load_person(full_dir)
            assert person_info['name'] not in all_data
            all_data[person_info['name']] = person_info

    return all_data

def load_person(directory):
    file_names = os.listdir(directory)
    assert 'info.json' in file_names
    with open(os.path.join(directory, "info.json"), 'r') as f:
        info = json.load(f)
    info['encodings'] = []
    for name in file_names:
        if name.endswith(".png") or name.endswith(".jpg"):
            encoding = load_and_cache_encoding(directory, name[:-4])
            info['encodings'].append(encoding)
    return info

def load_and_cache_encoding(directory, prefix, jitters=100):
    bin_file = os.path.join(directory, prefix)
    if os.path.isfile(bin_file + ".npy"):
        encoding = np.load(bin_file + ".npy")
        if not encoding.any():
            return None
    else:
        if os.path.isfile(os.path.join(directory, prefix + ".png")):
            encoding = compute_image_encoding(directory, prefix + ".png", jitters)
        else:
            encoding = compute_image_encoding(directory, prefix + ".jpg", jitters)
        if encoding is None:
            encoding = np.array(0)
        np.save(bin_file, encoding)
    return encoding

def compute_image_encoding(directory, filename, jitters=100):
    """
    Assumes it will receive an image with a single face in it.
    No more, no less.
    """
    fullname = os.path.join(directory, filename)
    img = face_recognition.load_image_file(fullname)
    face_encodings = face_recognition.face_encodings(img, num_jitters=jitters)
    if len(face_encodings) > 0:
        return face_encodings[0]
    else:
        return None


def get_names_faces_lists(all_data):
    names = []
    encoded_faces = []
    for name, person in all_data.items():
        for ef in person['encodings']:
            names.append(name)
            encoded_faces.append(ef)
    return names, encoded_faces


def get_identified_people(cv2_img, known_faces, names):

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2_img[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=10)
    if len(face_encodings) == 0:
        return {}
    face_encoding = face_encodings[0]
    matches = face_recognition.compare_faces(known_faces, face_encoding)

    hits = defaultdict(int)
    for match, name in zip(matches, names):
        # FML that 'match' is a np bool, not 'is True'
        if match == True:
            hits[name] += 1

    return hits

def get_face_distances(cv2_img, known_faces):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2_img[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=10)
    if len(face_encodings) == 0:
        return None
    return get_face_distances_with_encoding(face_encodings[0], known_faces)

def get_face_distances_with_encoding(face_encoding, known_faces):
    #matches = face_recognition.compare_faces(known_faces, face_encoding)
    distances = face_recognition.face_distance(known_faces, face_encoding)
    return distances


TOLERANCE = 0.35
def get_best_match(cv2_img, known_faces, names):
    distances = get_face_distances(cv2_img, known_faces)
    if distances is None:
        return None

    min_dist = 1
    use_name = None
    for name, dist in zip(names, distances):
        if dist < min_dist:
            min_dist = dist
            use_name = name


    if min_dist > TOLERANCE:
        return None
    else:
        return use_name

