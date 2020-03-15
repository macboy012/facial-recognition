import subprocess
import time

class Wakeup(object):
    def __init__(self):
        self.wake_process = None
        self.last_wakeup = 0

    def are_we_awake(self):
        return self.last_wakeup+30 > time.time()

    def wakeup(self, force=False):
        if not force and self.are_we_awake():
            return

        self.wake_process = subprocess.Popen("caffeinate -u", shell=True)
        time.sleep(0.1)

        self.last_wakeup = time.time()
        self.wake_process.terminate()
        self.wake_process.wait()
        self.wake_process = None
