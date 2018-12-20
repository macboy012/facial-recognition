import cv2
import os
import sys
import shutil

import utils

import json



def prompt_person(people_list):
    OFFSET = 3
    print " 1) Other"
    print " 2) Never match"
    i = 0
    for i, person in enumerate(people_list):
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
        name = raw_input("name? ")
        email = raw_input("email? ")
        if email == '':
            email = name.lower()
        people_list.append({
            'name': name,
            'email': email+"@athinkingape.com",
        })
        return name
    elif num == 2:
        return 'nevermatch'
    else:
        return people_list[num-OFFSET]['name']

def prompt_yn(name):
    while True:
        inp = raw_input("Is this %s? ([y]/n)" % name)
        if inp == '' or inp == 'y':
            return name
        elif inp == 'n':
            return None

def write_name(directory, dir_name, name):
    label_path = os.path.join(directory, dir_name, "name.txt")
    with open(label_path, 'w') as f:
        f.write(name+"\n")

def run(directory):

    match_data = utils.read_match_data()
    names, face_encodings = utils.get_names_faces_lists(match_data)
    people_list = utils.get_people_list()

    for dir_name in os.listdir(directory):
        fullpath = os.path.join(directory, dir_name)
        if os.path.isdir(fullpath) and dir_name != "__pycache__":
            filenames = os.listdir(fullpath)
            if "name.txt" in filenames:
                with open(os.path.join(directory, dir_name, "name.txt")) as f:
                    name = f.read().strip()
                    if name not in [p['name'] for p in people_list]:
                        print "Unknown label", name
                continue
            for i, fname in enumerate(filenames):
                if not fname.endswith(".png"):
                    continue
                img_path = os.path.join(directory, dir_name, fname)
                img = cv2.imread(img_path, 1)
                cv2.imshow('image', img)
                cv2.waitKey(1)
                bm = utils.get_best_match(img, face_encodings, names)
                if bm is None:
                    if i < 4:
                        continue
                    else:
                        name = prompt_person(people_list)
                        write_name(directory, dir_name, name)
                        break

                name = prompt_yn(bm)
                if name is None:
                    name = prompt_person(people_list)

                write_name(directory, dir_name, name)
                break

    utils.write_people_list(people_list)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        print "directory needed"
