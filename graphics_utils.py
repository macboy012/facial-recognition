import time

DRAWING_COLOR = (100,0,255)

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

    cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, int(font_scale), DRAWING_COLOR, thickness)
    return frame
