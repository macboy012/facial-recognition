import cv2
import os
import sys
import shutil

import utils

import json


def get_people_list():
    with open("match_data/people.json", "rb") as f:
        people = json.load(f)
    return people

def write_people_list(people):
    with open("match_data/people.json", "wb") as f:
        json.dump(people, f)


def prompt_person(people_list):
    OFFSET = 3
    print " 1) Other"
    print " 2) Delete"
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
        return 'delete'
    else:
        return people_list[num-OFFSET]['name']

def run(directory):
    people_list = get_people_list()

    for dir_name in os.listdir(directory):
        fullpath = os.path.join(directory, dir_name)
        if os.path.isdir(fullpath) and dir_name != "__pycache__":
            names = os.listdir(fullpath)
            if "name.txt" in names:
                continue
            print dir_name
            img_path = os.path.join(directory, dir_name, names[0])
            img = cv2.imread(img_path, 0)
            cv2.imshow('image', img)
            cv2.waitKey(1)
            name = prompt_person(people_list)
            if name == 'delete':
                shutil.rmtree(os.path.join(directory, dir_name))
            else:
                label_path = os.path.join(directory, dir_name, "name.txt")
                with open(label_path, 'w') as f:
                    f.write(name+"\n")

    write_people_list(people_list)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        print "directory needed"
