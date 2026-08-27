import time
import cv2
import numpy as np
import mss


def screen_capture(selected_monitor = 1, screen_name = "Pirairuchess", screen_coverage = False, screen_coverage_size = 0.7):
    """
    Continuously capture a monitor (or a region of it) and show it in a
    resizable OpenCV window until "q" is pressed.

    Args:
        selected_monitor (int): Index into ``mss.mss().monitors``. 1 is the
            first physical monitor; 0 (the combined virtual screen) is treated
            as invalid and falls back to the primary monitor. Any out-of-range
            value also falls back to the primary monitor.
        screen_name (str): Label for the capture.
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
    assert isinstance(screen_name, str) and screen_name, "screen_name must be a non-empty str"
    assert isinstance(screen_coverage, bool), "screen_coverage must be a bool"
    assert isinstance(screen_coverage_size, (int, float)) and not isinstance(screen_coverage_size, bool), \
        "screen_coverage_size must be a number"
    assert 0 < screen_coverage_size <= 1, "screen_coverage_size must be in the range (0, 1]"

    # The paramether content
    with mss.MSS() as sct:
        monitors = sct.monitors
        first_check = False

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


        while "Screen capturing":
            last_time = time.time() # For FPS

            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(analyzed_monitor))

            # Display the picture
            cv2.namedWindow(screen_name, cv2.WINDOW_NORMAL)
            if first_check == False:
                cv2.resizeWindow(screen_name, 1000, 700)                            
            cv2.imshow(screen_name, img)
            first_check = True

            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                #print(sct.monitors)
                cv2.destroyAllWindows()
                break

screen_capture(selected_monitor=1, screen_coverage=True)