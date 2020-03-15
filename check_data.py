from __future__ import print_function
import os
import sys
import cv2

import utils

def run(directory):

    match_data = utils.read_match_data()
    names, face_encodings = utils.get_names_faces_lists(match_data)

    dirs = os.listdir(directory)
    for i, dir_name in enumerate(sorted(dirs, reverse=True)):
        print(dir_name, i+1, len(dirs))
        fullpath = os.path.join(directory, dir_name)
        if not os.path.isdir(fullpath) or dir_name == "__pycache__":
            continue

        filenames = os.listdir(fullpath)
        if "name.txt" not in filenames:
            continue
        with open(os.path.join(directory, dir_name, "name.txt")) as f:
            name = f.read().strip()
        if name == 'nevermatch':
            continue

        matched = False
        for fname in filenames:
            if not fname.endswith(".png"):
                continue

            encoding = utils.load_and_cache_encoding(os.path.join(directory, dir_name), fname[:-4], jitters=10)
            try:
                bm = utils.get_best_match_with_encoding(encoding, face_encodings, names)
            except Exception:
                continue
            if bm == name:
                matched = True
                break

            if bm is None:
                continue
            # This means we got a result and it wasn't a match
            break

        if matched is True:
            continue


        for fname in filenames:
            if fname.endswith(".png"):
                break

        img = cv2.imread(os.path.join(directory, dir_name, fname), 1)
        print(os.path.join(directory, dir_name, fname))
        cv2.imshow('image', img)
        cv2.waitKey(1)

        if utils.prompt_yn("Not %s?" % name):
            os.remove(os.path.join(directory, dir_name, "name.txt"))

if __name__ == "__main__":
    run(sys.argv[1])
