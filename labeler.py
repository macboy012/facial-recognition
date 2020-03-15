from __future__ import print_function
import cv2
import os
import sys
import shutil
import time
from collections import defaultdict

import json

import utils


def run(directory):
    tree_model = utils.TreeModel(utils.load_model("modelv2_testing.pkl"))

    match_data = utils.read_match_data()
    names, face_encodings = utils.get_names_faces_lists(match_data)
    people_list = utils.get_people_list()
    existing_people = set([x['name'] for x in people_list])

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
                    print("Unknown label", name)
            continue
        todo.append(dir_name)

    for i, dir_name in enumerate(sorted(todo)):
        print("%s/%s" % (i+1, len(todo)), dir_name)
    #for dir_name in os.listdir(directory):
        fullpath = os.path.join(directory, dir_name)
        filenames = os.listdir(fullpath)
        image_count = sum([fname.endswith(".png") for fname in filenames])
        no_face = 0
        no_match = 0

        representative_img = None
        encodings = []
        for i, fname in enumerate(filenames):
            if not fname.endswith(".png"):
                continue

            img = cv2.imread(os.path.join(fullpath, fname), 1)
            cv2.imshow('image', img)
            cv2.waitKey(1)

            encoding = utils.load_and_cache_encoding(fullpath, fname[:-4], jitters=10)
            if encoding is None:
                continue
            else:
                encodings.append(encoding)
                representative_img = os.path.join(fullpath, fname)
                continue
            # Dead

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
                print('writing', 'nevermatch')
                utils.write_name(directory, dir_name, 'nevermatch')
                break

            if no_match > (image_count/2)+1:
                name = utils.prompt_person(people_list)
                print('writing', name)
                utils.write_name(directory, dir_name, name)
                break

            if bm is None:
                continue

            name = None
            if utils.prompt_yn("Is this %s?" % bm):
                name = bm

            if name is None:
                name = utils.prompt_person(people_list)
            print('writing', name)
            utils.write_name(directory, dir_name, name)
            break

        #else:
            # We didn't take any action because we didn't hit any of the flags, this is a nevermatch
            #print 'writing', 'nevermatch'
            #utils.write_name(directory, dir_name, 'nevermatch')

        if len(encodings) == 0:
            print('writing', 'nevermatch')
            utils.write_name(directory, dir_name, 'nevermatch')
            continue

        preds = tree_model.get_predictions(encodings, max_distance=0.4)
        print(preds)
        votes = defaultdict(int)
        for p in preds:
            if p is None:
                continue
            votes[p] += 1


        img = cv2.imread(representative_img, 1)
        cv2.imshow('image', img)
        cv2.waitKey(50)

        # for ambiguous directories, just say we'll never match. 
        print(votes)
        #continue
        #if len(votes) > 1:
            #print 'writing', 'nevermatch'
            #utils.write_name(directory, dir_name, 'nevermatch')
            #continue
        if len(votes) == 1:
            if sum(votes.values()) >= 2:
                final_name = votes.keys()[0]
            else:
                final_name = votes.keys()[0]
                if not utils.prompt_yn("Is this %s?" % final_name):
                    final_name = utils.prompt_person(people_list)
            print('writing', final_name)
            utils.write_name(directory, dir_name, final_name)
        elif len(votes) > 1:
            answer = utils.prompt_list([x[0] for x in sorted(votes.items(), reverse=True, key=lambda x: x[1])])
            if answer is None:
                answer = utils.prompt_person(people_list)
            print('writing', answer)
            utils.write_name(directory, dir_name, answer)
        else:
            final_name = utils.prompt_person(people_list)
            print('writing', final_name)
            utils.write_name(directory, dir_name, final_name)

        continue
        total_votes = sum(votes.values())

        # if not enough votes, prompt
        if total_votes < 2:
            if len(votes) > 0:
                final_name = votes.keys()[0]
                if not utils.prompt_yn("Is this %s?" % final_name):
                    final_name = utils.prompt_person(people_list)
            else:
                final_name = utils.prompt_person(people_list)

            print('writing', final_name)
            utils.write_name(directory, dir_name, final_name)
        else:
            final_name = votes.keys()[0]
            print('writing', final_name)
            utils.write_name(directory, dir_name, final_name)


    utils.write_people_list(people_list)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        print("directory needed")
