import time
import cv2
import numpy as np
import mss
from queue import Queue
import threading


def capture_frame(queue: Queue, selected_monitor = 1, screen_coverage = False, screen_coverage_size = 0.7):
    """
    Thread function: continuously capture frames and put them in the queue.
    Runs in the background.

    Args:
        queue (Queue): Where the frames will be storaged to be consumed.
        selected_monitor (int): Index into ``mss.mss().monitors``. 1 is the
            first physical monitor; 0 (the combined virtual screen) is treated
            as invalid and falls back to the primary monitor. Any out-of-range
            value also falls back to the primary monitor.
        screen_coverage (bool): When True, capture only the top-left fraction of
            the monitor given by ``screen_coverage_size`` instead of the whole
            screen.
        screen_coverage_size (float): Fraction (0-1) of the monitor's width and
            height to capture when ``screen_coverage`` is True.
    """
    # Asserting the params
    assert isinstance(selected_monitor, int) and not isinstance(selected_monitor, bool), \
        "selected_monitor must be an int"
    assert selected_monitor >= 0, "selected_monitor must be >= 0"
    assert isinstance(screen_coverage, bool), "screen_coverage must be a bool"
    assert isinstance(screen_coverage_size, (int, float)) and not isinstance(screen_coverage_size, bool), \
        "screen_coverage_size must be a number"
    assert 0 < screen_coverage_size <= 1, "screen_coverage_size must be in the range (0, 1]"
    assert isinstance(queue, Queue), "The queue must be a valid one"

    # With MSS get the selected monitor and converting into a nump array for OpenCV
    with mss.MSS as sct:
        monitors = sct.monitors

        # Part of the screen to capture
        if selected_monitor < len(monitors) and selected_monitor != 0:
            if screen_coverage == False:
                analyzed_monitor = monitors[selected_monitor]
            else:
                analyzed_monitor = {
                    "left": monitors[selected_monitor]["left"],
                    "top": monitors[selected_monitor]["top"],
                    "width": int(monitors[selected_monitor]["width"]*screen_coverage_size),
                    "height": int(monitors[selected_monitor]["height"]*screen_coverage_size)
                }
        else:
            print("Recording the computer defined 'main monitor'")
            analyzed_monitor = None
            for monitor in sct.monitors[1:]:          # skip the combined-screen entry
                if monitor.get("is_primary"):    
                    analyzed_monitor = monitor
            if analyzed_monitor is None:                
                analyzed_monitor = monitors[1]

        # Getting the frames and putting them on the queue
        while True:
            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(analyzed_monitor))

            # Put frame in queue (if queue is full, wait until it has space)
            queue.put(img)
            
def display_frames(queue: Queue, screen_name = "Pirairuchess"):
    """
    """
    assert isinstance(queue, Queue), "The queue must be a valid one"
    assert isinstance(screen_name, str) and screen_name, "screen_name must be a non-empty str"
    cv2.namedWindow(screen_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(screen_name, 1000, 700)

    while True:
            # Get frame from queue (waits if queue is empty)
            img = queue.get()

            # Display it
            cv2.imshow(screen_name, img)

            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

def screen_capture(selected_monitor = 1, screen_name = "Pirairuchess", screen_coverage = False, screen_coverage_size = 0.7):
    """
    """
    # Create a bounded queue (max 30 frames waiting)
    frame_queue = Queue(maxsize=30)

    # Creates two threads, for getting frames and displaying them
    capture_thread = threading.Thread(
        target=capture_frame,
        args=(frame_queue, selected_monitor, screen_coverage, screen_coverage_size),
        daemon=True
    )
    display_thread = threading.Thread(
        target=display_frames,
        args=(frame_queue, screen_name),
        daemon=True
    )

    # Start both threads
    capture_thread.start()
    display_thread.start()

    # Main thread
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stoping the capture...")