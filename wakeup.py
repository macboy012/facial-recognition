import subprocess
import time

class WakeupSingle(object):
    def __init__(self):
        self.last_wakeup = 0

    def are_we_awake(self):
        return self.last_wakeup+30 > time.time()

    @staticmethod
    def run_wakeup_command():
        wake_process = subprocess.Popen("caffeinate -u", shell=True)
        time.sleep(0.1)

        wake_process.terminate()
        wake_process.wait()
        wake_process = None

    def wakeup(self, force=False):
        if not force and self.are_we_awake():
            return

        WakeupSingle.run_wakeup_command()
        self.last_wakeup = time.time()

Wakeup = WakeupSingle()
