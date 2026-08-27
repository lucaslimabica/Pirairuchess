import time
import cv2
import numpy as np
import mss


with mss.MSS() as sct:
    # Part of the screen to capture
    monitors = sct.monitors
    main_monitor = None
    print(monitors)
    #main_monitor = monitors[1]
    for monitor in sct.monitors[1:]:          # skip the combined-screen entry
        if monitor.get("is_primary"):    
            if monitor["is_primary"] == True:
                main_monitor = monitor

    if main_monitor is None:                
        main_monitor = {"top": 60, "left": 100, "width": 1000, "height": 700}


    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(main_monitor))

        # Display the picture
        cv2.namedWindow('Pirairuchess', cv2.WINDOW_NORMAL)
        cv2.imshow("Pirairuchess", img)
        cv2.resizeWindow('Pirairuchess', 1000, 700)


        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            #print(sct.monitors)
            cv2.destroyAllWindows()
            break