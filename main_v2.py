import atexit
from collections import deque, Counter
import sys
import cv2
import datetime
import functools
import json
from multiprocessing import Pool, TimeoutError, Queue, Process
from Queue import Empty
import numpy as np
import os
import subprocess
import time
import utils
import requests
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s: %(pathname)s:line %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

FACE_OPEN_COOKIE = r'ANNX/~?v\O(b9PIJJ_bX,Rkn-Fai*IX4VdoOP?_PmInt+ll/'


def watchdog_action():
    wake_process = subprocess.Popen("caffeinate -u", shell=True)
    time.sleep(0.1)
    wake_process.terminate()
    wake_process.wait()

def watchdog(queue):
    parent_pid = os.getppid()

    # Expect to be terminated, no nice shutdown.
    wakeup_count = 0
    while True:
        try:
            queue.get(timeout=10)
            wakeup_count = 0
        except Empty:
            wakeup_count += 1
            logging.error("No heartbeat from parent, doing wakeup number %s" % wakeup_count)
            watchdog_action()

        # Exit on reparent (we'd die on SIGHUP, right?)
        if parent_pid != os.getppid():
            return
        time.sleep(1)

def start_watchdog():
    q = Queue()
    process = Process(target=watchdog, args=(q,))
    process.start()
    return q, process

class TimedFrameModify(object):
    def __init__(self):
        self.actions = []
        pass

    def process_frame(self, frame):
        now = time.time()
        save_actions = []
        for mod in self.actions:
            if mod['endtime'] < time.time():
                continue
            save_actions.append(mod)
            frame = mod['action'](frame)

        self.actions = save_actions

        # Technically we don't have to return it, we're modify it in place *shrug*
        # * when it's a numpy array/cv2 image
        return frame

    def add_modification(self, action, seconds, mtype, exclusive=False):
        mod = {
            'action': action,
            'endtime': time.time()+seconds,
            'mtype': mtype,
        }

        if exclusive:
            for tmod in self.actions:
                if tmod['mtype'] == mtype:
                    tmod['action'] = action
                    tmod['endtime'] = time.time()+seconds
                    break
            else:
                self.actions.append(mod)
        else:
            self.actions.append(mod)

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

pool = Pool(4)
ID_PHOTO_COUNT = 6
FACE_CAPTURE_DIRECTORY = "imgs"
class FaceIdentifier(object):
    def __init__(self):
        self.pool = pool
        self.active_faces = []
        self.active_groups = []

        self.match_data = utils.read_match_data()
        self.names, self.face_encodings = utils.get_names_faces_lists(self.match_data)
        self.model_storage = utils.load_model("modelv2_testing.pkl")
        self.tree_model = utils.TreeModel(self.model_storage)

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
        os.mkdir(our_dir, 0755)
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
            return counter.keys()[0]

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
            prediction = counter.keys()[0]
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

#putText( ., text, scale, color, chickness)
#getTextSize(text, font, fontscale, thickness)

def draw_xcentered_text(frame, text, height):
    fheight, fwidth, _ = frame.shape
    #base_font_scale = 6.0
    font_scale = 6.0
    #base_thickness = 10
    thickness = 10
    increment = 0.5
    while True:
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        twidth = size[0][0]
        theight = size[0][1]
        baseline = size[1]
        if twidth <= fwidth * 0.98:
            break
        font_scale -= increment

    x = (fwidth - twidth) / 2

    if height >= 0:
        y = theight+height
    else:
        y = fheight - baseline - theight - height

    if font_scale < 3.5:
        thickness -= 2

    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, DRAWING_COLOR, thickness)
    return frame


def main_loop(watchdog_queue):
    wakeup = Wakeup()
    face_identifier = FaceIdentifier()
    timed_frame_modify = TimedFrameModify()

    last_frame = None
    frame_counter = 0
    last_bump = 0
    while True:
        now_ts = time.time()
        if last_bump < now_ts-2:
            watchdog_queue.put("ping", timeout=1)
            last_bump = now_ts

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
                action = functools.partial(draw_xcentered_text, text='No match', height=100)

            else:
                action = functools.partial(draw_xcentered_text, text=prediction, height=100)

                email = face_identifier.get_email_for_name(prediction)
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

            timed_frame_modify.add_modification(action, 3, 'nametext', exclusive=True)

        frame_counter += 1

        if wakeup.are_we_awake():
            frame = cv2.flip(frame, flipCode=1)
            frame = timed_frame_modify.process_frame(frame)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imshow('Video', frame)

if __name__ == "__main__":
    queue, process = start_watchdog()
    try:
        main_loop(queue)
    finally:
        process.terminate()
