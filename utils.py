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
from six.moves import cPickle as pickle
from sklearn.neighbors import KDTree

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


class NoFaceException(Exception):
    pass
class NoMatchException(Exception):
    pass


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
    else:
        if os.path.isfile(os.path.join(directory, prefix + ".png")):
            encoding = compute_image_encoding(directory, prefix + ".png", jitters)
        else:
            encoding = compute_image_encoding(directory, prefix + ".jpg", jitters)
        if encoding is None:
            encoding = np.array(0)
        np.save(bin_file, encoding)

    if not encoding.any():
        return None
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

def get_encoding_from_cv2_img(cv2_img):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2_img[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=10)
    if len(face_encodings) == 0:
        return None

    return face_encodings[0]

def get_face_distances(cv2_img, known_faces):
    face_encoding = get_encoding_from_cv2_img(cv2_img)
    if face_encoding is not None:
        return get_face_distances_with_encoding(face_encoding, known_faces)
    else:
        return None

def get_face_distances_with_encoding(face_encoding, known_faces):
    #matches = face_recognition.compare_faces(known_faces, face_encoding)
    distances = face_recognition.face_distance(known_faces, face_encoding)
    return distances



def get_best_match_with_encoding(encoding, known_faces, names):
    distances = get_face_distances_with_encoding(encoding, known_faces)
    return _get_best_match(distances, names)

def get_best_match(cv2_img, known_faces, names):
    distances = get_face_distances(cv2_img, known_faces)
    return _get_best_match(distances, names)

TOLERANCE = 0.35
def _get_best_match(distances, names):
    if distances is None:
        raise NoFaceException()
        #return None

    min_dist = 1
    use_name = None
    for name, dist in zip(names, distances):
        if dist < min_dist:
            min_dist = dist
            use_name = name


    if min_dist > TOLERANCE:
        raise NoMatchException()
        #return None
    else:
        return use_name




def prompt_person(people_list):
    OFFSET = 3
    print " 1) Other"
    print " 2) Never match"
    i = 0
    i_name = {}
    for i, person in enumerate(sorted(people_list)):
        i_name[i+OFFSET] = person
        print " %s) %s" % (i+OFFSET, person['name'])

    while True:
        num = raw_input("number? ")
        try:
            num = int(num)
            if num > i+OFFSET or num < 1:
                continue
        except ValueError:
            continue
        break
    if num == 1:
        while True:
            name = raw_input("name? ")
            email = raw_input("email? ")
            if email == '':
                email = name.lower()
            if prompt_yn("%s - %s correct?" % (name, email)):
                break
        people_list.append({
            'name': name,
            'email': email+"@athinkingape.com",
        })
        return name
    elif num == 2:
        return 'nevermatch'
    else:
        return i_name[num]['name']

def prompt_yn(text):
    while True:
        inp = raw_input("%s ([y]/n)" % text)
        if inp == '' or inp == 'y':
            return True
        elif inp == 'n':
            return False

def write_name(directory, dir_name, name):
    label_path = os.path.join(directory, dir_name, "name.txt")
    with open(label_path, 'w') as f:
        f.write(name+"\n")


def load_people(directory):
    """
    returns a dictionary of name to list of face encodings
    ignores all nevermatch image sets
    """
    people = defaultdict(list)
    for dir_name in os.listdir(directory):
        fullpath = os.path.join(directory, dir_name)
        if os.path.isdir(fullpath) and dir_name != "__pycache__":
            names = os.listdir(fullpath)
            if "name.txt" not in names:
                continue
            with open(os.path.join(fullpath, "name.txt"), "rb") as f:
                truth_name = f.read().strip()
            if truth_name == 'nevermatch':
                continue
            for img_name in names:
                if img_name == 'name.txt':
                    continue
                if img_name.endswith(".png") or img_name.endswith(".jpg"):
                    encoding = load_and_cache_encoding(fullpath, img_name[:-4], jitters=10)

                    if encoding is None:
                        continue
                    people[truth_name].append((encoding, os.path.join(fullpath, img_name)))
                else:
                    continue

    return people


class FaceNameStorage(object):
    def __init__(self, names, faces, filepaths=None):
        self.names = names
        self.faces = faces
        self.filepaths = filepaths
        self._tree = None


class TreeModel(object):
    MAX_DISTANCE = 0.32
    NEIGHBOUR_COUNT = 10
    MIN_NEIGHBOUR_COUNT = 5
    def __init__(self, face_name_storage, max_distance=MAX_DISTANCE, neighbour_count=NEIGHBOUR_COUNT):
        self.face_name_storage = face_name_storage
        self.faces = self.face_name_storage.faces
        self.names = self.face_name_storage.names
        self.tree = KDTree(self.faces)
        self.max_distance = max_distance
        self.neighbour_count = neighbour_count

    def get_predictions(self, faces):

        results = []

        #distances_s, indices_s = self.tree.query(faces, k=self.neighbour_count)
        indices_s = self.tree.query_radius(faces, r=self.max_distance)
        #for distances, indices in zip(distances_s, indices_s):
        for indices in indices_s:
            votes = defaultdict(int)
            #for distance, index in zip(distances, indices):
            for index in indices:
                #if distance <= self.max_distance:
                votes[self.names[index]] += 1

            total_votes = sum(votes.values())
            if total_votes < self.MIN_NEIGHBOUR_COUNT:
                results.append(None)
                continue
            # If we didn't get at least 1/4 of our nearest neighbours inside our distance, no match
            #if total_votes < self.NEIGHBOUR_COUNT/4:
                #results.append(None)
                #continue

            best_person_name, best_person_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)[0]
            # Make sure our vote leader has more than half of the total votes
            if (best_person_votes / float(total_votes)) < 0.5:
                results.append(None)
                continue

            results.append(best_person_name)


        return results

def save_directory_to_model(directory, model_path):
    people = load_people(directory)

    serialized = []
    index_map = []
    files = []

    test_data = []
    for name, faces in people.items():
        for face, filepath in faces:
            #if random.random() < (TEST_PERCENT/100.0):
                #test_data.append((name, face))
            #else:
            index_map.append(name)
            serialized.append(face)
            files.append(filepath)

    fns = FaceNameStorage(index_map, serialized, files)

    with open(model_path, "wb") as f:
        pickle.dump(fns, f)

    return fns

def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)
