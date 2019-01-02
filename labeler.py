import cv2
import os
import sys
import shutil
import time

import utils

import json


def run(directory):

    match_data = utils.read_match_data()
    names, face_encodings = utils.get_names_faces_lists(match_data)
    people_list = utils.get_people_list()

    todo = []
    for dir_name in os.listdir(directory):
        fullpath = os.path.join(directory, dir_name)
        if not os.path.isdir(fullpath) or dir_name == "__pycache__":
            continue

        filenames = os.listdir(fullpath)
        if "name.txt" in filenames:
            with open(os.path.join(directory, dir_name, "name.txt")) as f:
                name = f.read().strip()
                if name not in [p['name'] for p in people_list] and name != 'nevermatch':
                    print "Unknown label", name
            continue
        todo.append(dir_name)

    for i, dir_name in enumerate(todo):
        print "%s/%s" % (i+1, len(todo)), dir_name
    #for dir_name in os.listdir(directory):
        fullpath = os.path.join(directory, dir_name)
        filenames = os.listdir(fullpath)
        image_count = sum([fname.endswith(".png") for fname in filenames])
        no_face = 0
        no_match = 0
        for i, fname in enumerate(filenames):
            if not fname.endswith(".png"):
                continue
            img_path = os.path.join(directory, dir_name, fname)
            img = cv2.imread(img_path, 1)
            cv2.imshow('image', img)
            cv2.waitKey(1)

            bm = None
            try:
                bm = utils.get_best_match(img, face_encodings, names)
            except utils.NoFaceException:
                no_face += 1
            except utils.NoMatchException:
                no_match += 1

            if no_face > (image_count/2)+1:
                print 'writing', 'nevermatch'
                utils.write_name(directory, dir_name, 'nevermatch')
                break

            if no_match > (image_count/2)+1:
                name = utils.prompt_person(people_list)
                print 'writing', name
                utils.write_name(directory, dir_name, name)
                break

            if bm is None:
                continue

            name = None
            if utils.prompt_yn("Is this %s?" % bm):
                name = bm

            if name is None:
                name = utils.prompt_person(people_list)
            print 'writing', name
            utils.write_name(directory, dir_name, name)
            break
        else:
            # We didn't take any action because we didn't hit any of the flags, this is a nevermatch
            print 'writing', 'nevermatch'
            utils.write_name(directory, dir_name, 'nevermatch')

    utils.write_people_list(people_list)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        print "directory needed"
