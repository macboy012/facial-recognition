import cv2
import atexit

VIDEO_CAPTURE = cv2.VideoCapture(0)
@atexit.register
def cleanup():
    if VIDEO_CAPTURE is not None:
        VIDEO_CAPTURE.release()

def load_frame_from_webcam():
    if not VIDEO_CAPTURE.isOpened():
        print('Unable to load camera.')
        time.sleep(5)

    ret, frame = VIDEO_CAPTURE.read()
    return frame
