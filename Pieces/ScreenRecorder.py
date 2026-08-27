import time
import cv2
import numpy as np
import mss


with mss.MSS() as sct:
    # Part of the screen to capture
    monitor = {"top": 60, "left": 100, "width": 1000, "height": 700}

    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))

        # Display the picture
        cv2.namedWindow('Pirairuchess', cv2.WINDOW_NORMAL)
        cv2.imshow("Pirairuchess", img)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            print(sct.monitors)
            cv2.destroyAllWindows()
            break