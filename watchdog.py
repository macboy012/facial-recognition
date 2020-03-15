import subprocess
import logging
import time
import os
from multiprocessing import Queue, Process

def watchdog_take_action():
    wake_process = subprocess.Popen("caffeinate -u", shell=True)
    time.sleep(0.1)
    wake_process.terminate()
    wake_process.wait()

def watchdog_function(queue):
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
            watchdog_take_action()

        # Exit on reparent (we'd die on SIGHUP, right?)
        if parent_pid != os.getppid():
            return
        time.sleep(1)

def start_watchdog():
    q = Queue()
    process = Process(target=watchdog_function, args=(q,))
    process.start()
    return q, process
