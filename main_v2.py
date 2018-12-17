import atexit
from collections import deque
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

class TimedFrameModify(object):
    def __init__(self):
        self.actions = []
        pass

    def process_frame(self, frame):
        now = time.time()
        save_actions = []
        for action, endtime in self.actions:
            if endtime < time.time():
                continue
            save_actions.append((action, endtime))
            frame = action(frame)

        self.actions = save_actions

        # Technically we don't have to return it, we're modify it in place *shrug*
        # * when it's a numpy array/cv2 image
        return frame

    def add_modification(self, action, seconds):
        self.actions.append((action, time.time()+seconds))

class Wakeup(object):
    def __init__(self):
        self.wake_process = None
        self.last_wakeup = 0

    def are_we_awake(self):
        return self.last_wakeup+30 > time.time()

    def wakeup(self):
        if self.are_we_awake():
            return

        self.wake_process = subprocess.Popen("caffeinate -u", shell=True)
        time.sleep(0.1)

        self.last_wakeup = time.time()
        self.wake_process.terminate()
        self.wake_process.wait()
        self.wake_process = None

pool = Pool(1)
ID_PHOTO_COUNT = 4
FACE_CAPTURE_DIRECTORY = "imgs"
class FaceIdentifier(object):
    def __init__(self):
        self.pool = pool
        self.active_faces = []
        self.active_groups = []

        self.match_data = utils.read_match_data()
        self.names, self.face_encodings = utils.get_names_faces_lists(self.match_data)

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

        async_result = self.pool.apply_async(utils.get_best_match, (face, self.face_encodings, self.names))
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
                    result = face_check['async_result'].get(timeout=0.01)
                    face_check['prediction'] = result
                    face_check['done'] = True
                except TimeoutError:
                    pass
            done = True
            for face_check in group:
                done = face_check['done'] and done
            if done:
                prediction = self.process_prediction(group)
                have_match = True

            else:
                next_active_groups.append(group)
        self.active_groups = next_active_groups
        return have_match, prediction

    def process_prediction(self, capture_group):
        test_set = set()
        prediction = None
        for face_check in capture_group:
            test_set.add(face_check['prediction'])
        if len(test_set) == 1:
            test_set_val = list(test_set)[0]
            if test_set_val is not None:
                prediction = test_set_val

        our_dir = os.path.join(FACE_CAPTURE_DIRECTORY, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f"))
        os.mkdir(our_dir, 0755)
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

#    on recognition
#    write to screen
#    open door



VIDEO_CAPTURE = cv2.VideoCapture(0)

@atexit.register
def cleanup():
    if VIDEO_CAPTURE is not None:
        VIDEO_CAPTURE.release()

def load_from_webcam():
    if not VIDEO_CAPTURE.isOpened():
        print('Unable to load camera.')
        time.sleep(5)

    ret, frame = VIDEO_CAPTURE.read()
    return frame

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

MIN_NEIGHBOURS = 10
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
def look_for_faces(frame):
    shrunk = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    faces = faceCascade.detectMultiScale(
        cv2.cvtColor(shrunk, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=MIN_NEIGHBOURS,
        minSize=(30, 30)
    )
    if len(faces) != 1:
        return None
    else:
        return faces[0]*4

DRAWING_COLOR = (100,0,255)

def text_action(frame, prediction):
    cv2.putText(frame, prediction, (150,450), cv2.FONT_HERSHEY_DUPLEX, 6.0, DRAWING_COLOR, 12)
    return frame

def main_loop():
    wakeup = Wakeup()
    face_identifier = FaceIdentifier()
    timed_frame_modify = TimedFrameModify()

    last_frame = None
    frame_counter = 0
    while True:
        frame = load_from_webcam()

        if last_frame is not None:
            if do_frames_differ(frame, last_frame, 100):
                wakeup.wakeup()
        last_frame = frame

        face = look_for_faces(frame)
        if face is not None:
            wakeup.wakeup()
            face_identifier.track_face(frame, face, frame_counter)

            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x+w, y+h), DRAWING_COLOR, 15)

        have_match, prediction = face_identifier.check_stuff()
        if have_match:
            if prediction is None:
                action = functools.partial(text_action, prediction='No match')
            else:
                action = functools.partial(text_action, prediction=prediction)
                # open sesame
            timed_frame_modify.add_modification(action, 2)

        frame_counter += 1

        if wakeup.are_we_awake():
            frame = cv2.flip(frame, flipCode=1)
            frame = timed_frame_modify.process_frame(frame)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imshow('Video', frame)

if __name__ == "__main__":
    main_loop()

#get a frame...

#look for faces in the frame.
