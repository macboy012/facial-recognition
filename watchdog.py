import logging
from queue import Empty
import time
import os
from multiprocessing import Queue, Process

class Watchdog:
    def __init__(self, action):
        self.action = action
        self.last_watchdog_stroke = 0

    def watchdog_function(self):
        parent_pid = os.getppid()

        # Expect to be terminated, no nice shutdown.
        wakeup_count = 0
        while True:
            try:
                self.queue.get(timeout=10)
                wakeup_count = 0
            except Empty:
                wakeup_count += 1
                logging.error("No heartbeat from parent, doing action number %s" % wakeup_count)
                self.action()

            # Exit on reparent
            if parent_pid != os.getppid():
                return
            time.sleep(1)

    def start_watchdog(self):
        self.queue = Queue()
        self.process = Process(target=self.watchdog_function)
        self.process.start()

    def stroke_watchdog(self):
        now_ts = time.time()
        if self.last_watchdog_stroke < now_ts-2:
            self.queue.put("ping", timeout=1)
            self.last_watchdog_stroke = now_ts

    def stop_watchdog(self):
        self.process.terminate()
