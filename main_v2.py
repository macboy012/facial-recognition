from __future__ import print_function
from collections import deque, Counter
import sys
import cv2
import datetime
import functools
import json
from multiprocessing import Pool, TimeoutError
import numpy as np
import os
import subprocess
import time
import utils
import requests
import socket
import logging
from watchdog import Watchdog
from wakeup import Wakeup
from graphics_utils import TimedFrameModify, draw_xcentered_text
import capture_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s: %(pathname)s:line %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

FACE_OPEN_COOKIE = r'ANNX/~?v\O(b9PIJJ_bX,Rkn-Fai*IX4VdoOP?_PmInt+ll/'

pool = Pool(4)
ID_PHOTO_COUNT = 6
FACE_CAPTURE_DIRECTORY = "imgs"
class FaceIdentifier(object):
    MIN_NEIGHBOURS = 10
    CASCPATH = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(CASCPATH)

    def __init__(self):
        self.pool = pool
        self.active_faces = []
        self.active_groups = []

        self.match_data = utils.read_match_data()
        self.names, self.face_encodings = utils.get_names_faces_lists(self.match_data)
        self.model_storage = utils.load_model("modelv2_testing.pkl")
        self.tree_model = utils.TreeModel(self.model_storage)

    def look_for_faces(self, frame):
        shrunk = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        faces = FaceIdentifier.faceCascade.detectMultiScale(
            cv2.cvtColor(shrunk, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=FaceIdentifier.MIN_NEIGHBOURS,
            minSize=(30, 30)
        )
        if len(faces) != 1:
            return None
        else:
            return faces[0]*4

    def get_email_for_name(self, name):
        return self.model_storage.name_email[name]

    def track_face(self, frame, face_loc, frame_counter):
        if len(self.active_groups) >= 1:
            return

        if len(self.active_faces) > 0:
            if frame_counter != self.active_faces[-1]['frame_counter'] + 1:
                self.active_faces = []

        x, y, w, h = face_loc

        x1 = x
        x2 = x+w
        y1 = y
        y2 = y+h

        # All this down to the copy is just expanding the area we capture
        # out so we get an extra 25% on all directions
        width_plus = int(w/4.0)
        height_plus = int(h/4.0)
        x1 -= width_plus
        x2 += width_plus
        y1 -= height_plus
        y2 += height_plus

        y_max, x_max = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x_max, x2)
        y2 = min(y_max, y2)

        face = frame[y1:y2, x1:x2]
        face = np.copy(face)

        async_result = self.pool.apply_async(utils.get_encoding_from_cv2_img, (face,))

        self.active_faces.append({
            "face": face,
            "async_result": async_result,
            "frame_counter": frame_counter,
            "prediction": None,
            "done": False,
        })

    #so, what are the conditions in which i throw stuff away?
    #i get frames with some time difference or fc count difference between them

    #this just keeps slapping me with images
    #or doesn't

    #i need to make descisions and save images during this.

    def check_stuff(self):
        if len(self.active_faces) >= ID_PHOTO_COUNT:
            self.active_groups.append(self.active_faces)
            self.active_faces = []

        prediction = None
        have_match = False
        next_active_groups = []
        # this is broke as shit, 1 active group right now. NEVER MORE
        for group in self.active_groups:
            for face_check in group:
                if face_check['done'] is True:
                    continue
                try:
                    result = None
                    reason = None
                    try:
                        result = face_check['async_result'].get(timeout=0.01)
                        reason = 'match'
                    except utils.NoFaceException:
                        reason = 'no_face'
                    except utils.NoMatchException:
                        reason = 'no_match'

                    face_check['encoding'] = result
                    face_check['done'] = True
                    face_check['reason'] = reason
                except TimeoutError:
                    pass

            done = True
            for face_check in group:
                done = face_check['done'] and done
            if done:
                prediction = self.process_encodings(group)
                have_match = True

            else:
                next_active_groups.append(group)
        self.active_groups = next_active_groups
        return have_match, prediction

    def process_encodings(self, capture_group):
        encodings = [f['encoding'] for f in capture_group if f['encoding'] is not None]

        preds = self.tree_model.get_predictions(encodings)
        counter = Counter(preds)
        logging.info(counter)
        if None in counter:
            del counter[None]

        our_dir = os.path.join(FACE_CAPTURE_DIRECTORY, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f"))
        os.mkdir(our_dir, 0o755)
        for capture in capture_group:
            local_name = "%s.png" % capture['frame_counter']
            file_path = os.path.join(our_dir, local_name)
            cv2.imwrite(file_path, capture['face'])

        if len(counter) > 1:
            return None

        total_votes = sum(counter.values())

        if total_votes < 2:
            return None
        else:
            return list(counter.keys())[0]

    def process_prediction(self, capture_group):
        test_set = set()
        prediction = None

        """
        for face_check in capture_group:
            test_set.add(face_check['prediction'])
        if len(test_set) == 1:
            test_set_val = list(test_set)[0]
            if test_set_val is not None:
                prediction = test_set_val
        """

        counter = Counter([face_check['prediction'] for face_check in capture_group])
        logging.info(counter)
        reason_counter = Counter([face_check['reason'] for face_check in capture_group])
        logging.info(reason_counter)

        prediction = None
        if len(counter) == 1:
            prediction = list(counter.keys())[0]
        elif len(counter) == 2:
            if None in counter:
                keys  = counter.keys()
                keys.remove(None)
                other = keys[0]
                if counter[other] >= 2:
                    prediction = other
                #if counter[None] <= counter[other]:
                    #prediction = other

        our_dir = os.path.join(FACE_CAPTURE_DIRECTORY, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f"))
        os.mkdir(our_dir, 0o755)
        for capture in capture_group:
            local_name = "%s.png" % capture['frame_counter']
            file_path = os.path.join(our_dir, local_name)
            cv2.imwrite(file_path, capture['face'])
            capture['face'] = local_name
            del capture['async_result']
            del capture['done']
            del capture['frame_counter']

        with open(os.path.join(our_dir, "prediction.json"), 'wb') as f:
            json.dump(capture_group, f)

        return prediction

def do_frames_differ(frame1, frame2, difference):
    # subsample
    small_frame1 = cv2.resize(frame1, (0, 0), fx=0.1, fy=0.1)
    small_frame2 = cv2.resize(frame2, (0, 0), fx=0.1, fy=0.1)

    # subtract and turn into a datatype that can actually hold negatives
    diff = np.subtract(small_frame1, small_frame2, dtype='int16')
    diff = np.abs(diff)
    differing = np.vectorize(lambda x: x > difference)(diff)
    # Must have at least 100 pixels counting as different
    return np.sum(differing) > 100

def match_action(email, timed_frame_modify):
    logging.info(email)
    try:
        socket.gethostbyname("frontdoor")
    except:
        pass
    else:
        try:
            response = requests.get("http://frontdoor/api/face_open_door", params={"email":email}, cookies={"Auth":FACE_OPEN_COOKIE}, timeout=3)
        except requests.exceptions.Timeout:
            resp_data = {'status': 'failure', 'message': 'Timeout'}
        else:
            logging.info(response.content)
            resp_data = response.json()
        if resp_data['status'] == 'failure':
            info = None
            if resp_data['message'] == 'Outside of working hours':
                info = functools.partial(draw_xcentered_text, text="Outside work hours", height=-50)
            elif resp_data['message'] == 'Please reauthenticate' or resp_data['message'] == 'No matching user':
                #info = functools.partial(draw_xcentered_text, text="Login: http://frontdoor/", height=-50)
                fnfn = functools.partial(draw_xcentered_text, text="Login:", height=-10)
                timed_frame_modify.add_modification(fnfn, 10, 'infotext', exclusive=True)
                fnfn = functools.partial(draw_xcentered_text, text="http://frontdoor/", height=-100)
                timed_frame_modify.add_modification(fnfn, 10, 'infotext2', exclusive=True)
            elif resp_data['message'] == 'Timeout':
                info = functools.partial(draw_xcentered_text, text="Timeout", height=-50)
            if info is not None:
                timed_frame_modify.add_modification(info, 3, 'infotext', exclusive=True)

DRAWING_COLOR = (100,0,255)

class FrameWorker:
    def process_frame(self, frame):
        raise NotImplemented("Must implement in subclass")

#class DifferenceWakeup

def main_loop(watchdog):
    face_identifier = FaceIdentifier()
    timed_frame_modify = TimedFrameModify()

    last_frame = None
    frame_counter = 0
    while True:
        watchdog.stroke_watchdog()

        frame = capture_utils.load_frame_from_webcam()

        if last_frame is not None:
            if do_frames_differ(frame, last_frame, 100):
                Wakeup.wakeup()
        last_frame = frame

        face = face_identifier.look_for_faces(frame)
        if face is not None:
            Wakeup.wakeup()
            face_identifier.track_face(frame, face, frame_counter)

            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x+w, y+h), DRAWING_COLOR, 15)

        have_match, prediction = face_identifier.check_stuff()
        if have_match:
            if prediction is None:
                action = functools.partial(draw_xcentered_text, text='No match', height=100)

            else:
                action = functools.partial(draw_xcentered_text, text=prediction, height=100)

                email = face_identifier.get_email_for_name(prediction)
                match_action(email, timed_frame_modify)

            timed_frame_modify.add_modification(action, 3, 'nametext', exclusive=True)

        frame_counter += 1

        if Wakeup.are_we_awake():
            frame = cv2.flip(frame, flipCode=1)
            frame = timed_frame_modify.process_frame(frame)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imshow('Video', frame)

if __name__ == "__main__":
    watchdog = Watchdog(action=Wakeup.run_wakeup_command)
    watchdog.start_watchdog()
    try:
        main_loop(watchdog)
    finally:
        watchdog.stop_watchdog()
