import cv2
import os
import sys
import logging
import requests
import datetime
import time
import glob
import numpy
import subprocess


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
#logging.basicConfig(filename='webcam.log',level=log.INFO)
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

MIN_CAPTURE_FRAMES = 20
FRAMES_COUNT_TO_SAVE = 20

CAPTURE_AREA_XY1 = (600, 500)
CAPTURE_AREA_XY2 = (1000, 1000)

WANTED_TIME = 60

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

    def flush_capture_buffer(self):
        self.end_wakeup()
        if len(self.capture_buffer) < MIN_CAPTURE_FRAMES:
            logging.info("Emptied buffer of %s images without saving" % len(self.capture_buffer))
            self.capture_buffer = []
            return

        our_dir = os.path.join(FACE_CAPTURE_DIRECTORY, self.capture_buffer[0].timestamp.strftime("%Y-%m-%d_%H-%M-%S"))
        os.mkdir(our_dir, 0755)
        for capture in self.capture_buffer:
            file_path = os.path.join(our_dir, "%s.png" % capture.frame_counter)
            cv2.imwrite(file_path, capture.face_frame)
        logging.info("Wrote buffer of %s images to %s" % (len(self.capture_buffer), our_dir))

        self.capture_buffer = []
        self.draw_wanted_start_frame = self.frame_counter

    def wakeup(self):
        if self.wake_process is None:
            self.wake_process = subprocess.Popen(["caffeinate", "-u"], shell=True)

    def end_wakeup(self):
        if self.wake_process is not None:
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
            st = time.time()
            response = requests.get(url, stream=True)
            filename = "/tmp/tmp_file"
            with open(filename, "wb") as handle:
                for chunk in response.iter_content(chunk_size=512):
                    if chunk:  # filter out keep-alive new chunks
                        handle.write(chunk)

            v = next(FaceCapture.load_from_file(filename))
            et = time.time()
            print et-st
            yield v

    def run(self):

        """
        get frame
        process frame
        handle result of processing
        process output
        """

        while True:
            self.frame_counter += 1
            frame = next(self.frame_generator)

            xMax = frame.shape[1]
            yMax = frame.shape[0]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print CAPTURE_AREA_XY1[1],CAPTURE_AREA_XY2[1], CAPTURE_AREA_XY1[0],CAPTURE_AREA_XY2[0]
            #capture_area = gray[CAPTURE_AREA_XY1[1]:CAPTURE_AREA_XY2[1], CAPTURE_AREA_XY1[0]:CAPTURE_AREA_XY2[0]]
            capture_area = gray

            faces = faceCascade.detectMultiScale(
                capture_area,
                scaleFactor=1.1,
                minNeighbors=MIN_NEIGHBOURS,
                minSize=(30, 30)
            )

            if len(faces) == 1:
                face = faces[0]
                x, y, w, h = face

                x1 = x
                x2 = x+w
                y1 = y
                y2 = y+h

                gray_face = capture_area[y1:y2, x1:x2]

                face_obj = Face(face, gray_face, self.frame_counter)
                self.capture_face(face_obj)

            # do flush if we have enough frames
            if len(self.capture_buffer) >= FRAMES_COUNT_TO_SAVE:
                self.flush_capture_buffer()

            # clear buffer if we never got enough frames
            if len(self.capture_buffer) > 0:
                if self.frame_counter - self.capture_buffer[-1].frame_counter > MAX_FRAMES_BETWEEN_CAPTURES:
                    self.flush_capture_buffer()

            """
            putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
            .   @brief Draws a text string.
            .
            .   The function cv::putText renders the specified text string in the image. Symbols that cannot be rendered
            .   using the specified font are replaced by question marks. See #getTextSize for a text rendering code
            .   example.
            .
            .   @param img Image.
            .   @param text Text string to be drawn.
            .   @param org Bottom-left corner of the text string in the image.
            .   @param fontFace Font type, see #HersheyFonts.
            .   @param fontScale Font scale factor that is multiplied by the font-specific base size.
            .   @param color Text color.
            .   @param thickness Thickness of the lines used to draw a text.
            .   @param lineType Line type. See #LineTypes
            .   @param bottomLeftOrigin When true, the image data origin is at the bottom-left corner. Otherwise,
            .   it is at the top-left corner.
            """
            if self.draw_wanted_start_frame > self.frame_counter - WANTED_TIME:
                cv2.putText(frame, "Thanks!", (300,150), cv2.FONT_HERSHEY_DUPLEX, 4.0, (0,255,0), 7)


            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                #x += CAPTURE_AREA_XY1[0]
                #y += CAPTURE_AREA_XY1[1]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                #cv2.fillPoly(frame, numpy.array([[(x, y), (x+w, y), (x, y+h), (x+w, y+h)]], dtype=numpy.int32), (255, 255, 0))

            #cv2.rectangle(frame, CAPTURE_AREA_XY1, CAPTURE_AREA_XY2, (255, 255, 0), 2)
            #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

            # Display the resulting frame
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

