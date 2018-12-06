import cv2
import json
import os
import sys
import logging
import requests
import datetime
import time
import glob
import numpy as np
import subprocess
from collections import defaultdict

import utils


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
logging.basicConfig(level=logging.INFO)

anterior = 0

class Face(object):
    def __init__(self, location, face_frame, frame_counter):
        self.location = location
        self.face_frame = face_frame
        self.frame_counter = frame_counter
        self.timestamp = datetime.datetime.now()

    def get_surface_area(self):
        x, y, w, h = self.location
        return w*h

MAX_FRAMES_BETWEEN_CAPTURES = 10
FACE_CAPTURE_DIRECTORY = "imgs"
AREA_RATIO_MIN = 0.8
AREA_RATIO_MAX = 1.2
X_DIFF_PERCENT = 0.2
Y_DIFF_PERCENT = 0.2
MIN_NEIGHBOURS = 20

MIN_CAPTURE_FRAMES = 10
FRAMES_COUNT_TO_SAVE = 10

ACTIVATE_MEAN_DIFF = 8

TEXT_DISPLAY_TIME = 60

DRAWING_COLOR = (100,0,255)

face_recognizer, labels = utils.load_model()

def try_label(face):
    refined = utils.refine_image(face.face_frame)
    if refined is None:
        return None
    label, confidence = face_recognizer.predict(refined)
    if confidence > 20:
        return None
    #print labels[label], confidence
    return labels[label]

class FaceCapture(object):
    def __init__(self, video_capture, frame_generator=None):
        self.video_capture = video_capture
        self.in_capture = False
        self.capture_buffer = []

        self.frame_counter = 0
        if frame_generator is None:
            self.frame_generator = self.load_from_webcam()
        else:
            self.frame_generator = frame_generator

        self.draw_wanted_start_frame = -100000
        self.wake_process = None
        self.last_face_frame = 0
        self.last_wakeup = 0
        self.thank_person = None

        self.found_people = defaultdict(int)

        self.match_data = utils.read_match_data()

        self.names, self.face_encodings = utils.get_faces_names_lists(self.match_data)

    def flush_capture_buffer(self):
        self.end_wakeup()
        if len(self.capture_buffer) < MIN_CAPTURE_FRAMES:
            logging.info("Emptied buffer of %s images without saving" % len(self.capture_buffer))
            self.capture_buffer = []
            self.found_people.clear()
            return

        our_dir = os.path.join(FACE_CAPTURE_DIRECTORY, self.capture_buffer[0].timestamp.strftime("%Y-%m-%d_%H-%M-%S"))
        os.mkdir(our_dir, 0755)
        for capture in self.capture_buffer:
            file_path = os.path.join(our_dir, "%s.png" % capture.frame_counter)
            cv2.imwrite(file_path, capture.face_frame)
        logging.info("Wrote buffer of %s images to %s" % (len(self.capture_buffer), our_dir))

        self.capture_buffer = []
        self.draw_wanted_start_frame = self.frame_counter

        max_hits = 0
        max_person = None
        for person, hits in self.found_people.items():
            if hits > max_hits:
                max_person = person
                max_hits = hits

        self.thank_person = max_person
        self.found_people.clear()

    def wakeup(self):
        if self.wake_process is None:
            if self.last_wakeup+30 > time.time():
                return

            self.wake_process = subprocess.Popen("caffeinate -u", shell=True)
            time.sleep(0.1)
            self.end_wakeup()
        else:
            # If we have a process, we're definitely awake.
            self.last_wakeup = time.time()

    def end_wakeup(self):
        if self.wake_process is not None:
            self.last_wakeup = time.time()
            self.wake_process.terminate()
            self.wake_process.wait()
            self.wake_process = None

    def capture_face(self, face):
        self.wakeup()
        # If we have nothing to check against, always capture.
        if len(self.capture_buffer) == 0:
            self.capture_buffer.append(face)
            return

        last_face = self.capture_buffer[-1]

        size_ratio = float(face.get_surface_area()) / last_face.get_surface_area()
        if size_ratio < AREA_RATIO_MIN or size_ratio > AREA_RATIO_MAX:
            logging.info("Area ratio of %s too different" % size_ratio)
            return

        x, y, w, h = last_face.location
        x_min = x - w * X_DIFF_PERCENT
        x_max = x + w * X_DIFF_PERCENT

        y_min = y - w * Y_DIFF_PERCENT
        y_max = y + w * Y_DIFF_PERCENT

        x, y, w, h = face.location
        if x < x_min or x > x_max:
            logging.info("X %s outside allowable %s - %s" % (x, x_min, x_max))
            return
        if y < y_min or y > y_max:
            logging.info("Y %s outside allowable %s - %s" % (y, y_min, y_max))
            return

        self.capture_buffer.append(face)

    def load_from_webcam(self):
        if not self.video_capture.isOpened():
            print('Unable to load camera.')
            time.sleep(5)

        while True:
            # Capture frame-by-frame
            ret, frame = self.video_capture.read()
            yield frame

    @staticmethod
    def load_from_file(filename):
        files = glob.glob(filename)
        for f in files:
            yield cv2.imread(f, 1)

    @staticmethod
    def load_from_request(url):
        while True:
            response = requests.get(url, stream=True)
            filename = "/tmp/tmp_file"
            with open(filename, "wb") as handle:
                for chunk in response.iter_content(chunk_size=512):
                    if chunk:  # filter out keep-alive new chunks
                        handle.write(chunk)

            v = next(FaceCapture.load_from_file(filename))
            yield v

    def run(self):

        """
        get frame
        process frame
        handle result of processing
        process output
        """

        last_mean = 0
        st = time.time()
        sframe = 0
        while True:
            if time.time()-1 > st:
                st = time.time()
                #print 'fps', self.frame_counter - sframe
                sframe = self.frame_counter

            self.frame_counter += 1
            frame = next(self.frame_generator)

            xMax = frame.shape[1]
            yMax = frame.shape[0]

            capture_area = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean, stddev = cv2.meanStdDev(capture_area)
            mean = mean[0][0]
            stddev = stddev[0][0]

            if abs(mean-last_mean) > ACTIVATE_MEAN_DIFF:
                self.wakeup()

            last_mean = mean

            faces = []
            if abs(self.frame_counter - self.last_face_frame) < 20 or self.frame_counter % 5 == 0:
                faces = faceCascade.detectMultiScale(
                    capture_area,
                    scaleFactor=1.1,
                    minNeighbors=MIN_NEIGHBOURS,
                    minSize=(30, 30)
                )

            if len(faces) == 1:
                self.last_face_frame = self.frame_counter
                face = faces[0]
                x, y, w, h = face

                x1 = x
                x2 = x+w
                y1 = y
                y2 = y+h

                # expand_area
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

                colour_face = frame[y1:y2, x1:x2]
                colour_face = np.copy(colour_face)

                face_obj = Face(face, colour_face, self.frame_counter)
                self.capture_face(face_obj)

                people = utils.get_identified_people(colour_face, self.face_encodings, self.names)
                for name, count in people.items():
                    self.found_people[name] += count

            # do flush if we have enough frames
            if len(self.capture_buffer) >= FRAMES_COUNT_TO_SAVE:
                self.flush_capture_buffer()

            # clear buffer if we never got enough frames
            if len(self.capture_buffer) > 0:
                if self.frame_counter - self.capture_buffer[-1].frame_counter > MAX_FRAMES_BETWEEN_CAPTURES:
                    self.flush_capture_buffer()

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), DRAWING_COLOR, 15)

            # Display the resulting frame
            frame = cv2.flip(frame, flipCode=1)

            if self.draw_wanted_start_frame > self.frame_counter - TEXT_DISPLAY_TIME:
                cv2.putText(frame, "Thanks!", (150,250), cv2.FONT_HERSHEY_DUPLEX, 8.0, DRAWING_COLOR, 14)
                if self.thank_person is not None:
                    cv2.putText(frame, self.thank_person, (150,450), cv2.FONT_HERSHEY_DUPLEX, 6.0, DRAWING_COLOR, 12)

            # When the screen goes off, we hang on waitKey, so don't do it if we haven't done a wakeup recently
            # Also no point in updating the screen if it is off.
            if self.last_wakeup + 40 > time.time():
                cv2.imshow('Video', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Display the resulting frame
                cv2.imshow('Video', frame)

def get_frames():
    files = glob.glob("/Users/mackenzie/frontdoor/20*/*.jpg")
    for filename in files:
        yield next(FaceCapture.load_from_file(filename))


if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)
    #video_capture = None
    #fc = FaceCapture(video_capture, frame_generator=get_frames())
    #gen = FaceCapture.load_from_request('http://frontdoor:8080/?action=snapshot')
    #fc = FaceCapture(video_capture, frame_generator=gen)
    fc = FaceCapture(video_capture)
    try:
        fc.run()
    finally:
        # When everything is done, release the capture
        if video_capture is not None:
            video_capture.release()
        cv2.destroyAllWindows()

