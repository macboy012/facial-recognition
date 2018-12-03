import os
import sys
import json
import cv2
import numpy
from collections import defaultdict
import random
import dlib
from imutils import face_utils
import utils

CONFIDENCE_CUTOFF = 10
def run(dir_name):
    name_face_map = utils.get_names_faces_dict(dir_name)

    holdout_count = 0#len(name_face_map)/2
    holdouts = {}
    for i in range(holdout_count):
        holdout_name = random.choice(name_face_map.keys())
        holdouts[holdout_name] = name_face_map.pop(holdout_name)

    test_data = defaultdict(list)
    for name, name_faces in name_face_map.items():
        sample_count = len(name_faces)/10
        randomed = set()
        for i in range(sample_count):
            index = 0
            while index in randomed:
                index = random.randint(0, len(name_faces)-1)
            randomed.add(index)
            assert name_faces[index] is not None
            test_data[name].append(name_faces[index])
            name_faces[index] = None


    faces = []
    label_indexes = []
    labels = []
    name_map = {}
    for i, (name, name_faces) in enumerate(name_face_map.items()):
        labels.append(name)
        name_map[name] = i
        for face in name_faces:
            if face is None:
                continue
            faces.append(face)
            label_indexes.append(i)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, numpy.array(label_indexes))

    for name, name_faces in test_data.items():
        print '-----------------'
        print name
        print 'sample_count: %s' % len(name_face_map[name])
        print 'test_count: %s' % len(name_faces)

        fp = 0
        fn = 0
        tp = 0
        tn = 0
        confs = []
        for name_face in name_faces:
            label_id, confidence = face_recognizer.predict(name_face)

            if confidence < CONFIDENCE_CUTOFF:
                if name == labels[label_id]:
                    confs.append(confidence)
                    tp += 1
                else:
                    fp += 1
            else:
                if name == labels[label_id]:
                    fn += 1
                else:
                    tn += 1

        if len(confs) == 0:
            print 'confidence: -'
        else:
            print 'confidence: %s' % (sum(confs)/len(confs))

        stats = {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
        }
        print utils.format_graph(stats)
        print utils.format_table(stats)

    holdout_stats = defaultdict(int)
    for holdout_name, holdout_faces in holdouts.items():
        print 'holdout face test - %s' % holdout_name
        for face in holdout_faces:
            label_id, confidence = face_recognizer.predict(face)
            if confidence < CONFIDENCE_CUTOFF:
                holdout_stats[labels[label_id]] += 1
                #print 'false positive', confidence, label_map[label_id]
            #else:
                #print 'true negative'
            #print label_id, confidence
    print holdout_stats

    return face_recognizer, labels

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "directory to use requiried"
        sys.exit(1)
    recognizer, labels = run(sys.argv[1])
    utils.save_model(recognizer, labels)
