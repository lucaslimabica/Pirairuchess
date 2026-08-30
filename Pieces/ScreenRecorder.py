import queue
import threading
import time
from datetime import datetime
import cv2
import numpy as np
import mss


TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS


def _resolve_monitor(
        sct,
        selected_monitor,
        screen_coverage,
        screen_coverage_size
    ):
    """Pick the mss monitor dict to grab, optionally cropped to a centered fraction."""
    monitors = sct.monitors

    if selected_monitor != 0 and selected_monitor < len(monitors):
        target = monitors[selected_monitor]
    else:
        print("Recording the computer defined 'main monitor'")
        target = next(
            (m for m in monitors[1:] if m.get("is_primary")),
            monitors[1],
        )

    if not screen_coverage:
        return target

    return {
        "left": target["left"],
        "top": target["top"],
        "width": int(target["width"] * screen_coverage_size),
        "height": int(target["height"] * screen_coverage_size),
    }


def _put_drop_oldest(q, item):
    """Put `item` on the queue; if it's full, discard the oldest entry and retry"""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass

def _drain(q):
    """Remove and discard every item currently in the queue"""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return

def capture_frame(
        frame_queue,
        recording_queue,
        recording_event,
        stop_event,
        selected_monitor=1,
        screen_coverage=False,
        screen_coverage_size=0.7
    ):
    """Grab screen frames at TARGET_FPS, feeding the display queue always and the
    recording queue while `recording_event` is set. Runs until `stop_event` is set"""
    assert isinstance(selected_monitor, int) and not isinstance(selected_monitor, bool), \
        "selected_monitor must be an int"
    assert selected_monitor >= 0, "selected_monitor must be >= 0"
    assert isinstance(screen_coverage, bool), "screen_coverage must be a bool"
    assert isinstance(screen_coverage_size, (int, float)) and not isinstance(screen_coverage_size, bool), \
        "screen_coverage_size must be a number"
    assert 0 < screen_coverage_size <= 1, "screen_coverage_size must be in the range (0, 1]"

    with mss.mss() as sct:
        monitor = _resolve_monitor(sct, selected_monitor, screen_coverage, screen_coverage_size)

        next_frame = time.perf_counter() # Starts the timer for each frame capture

        while not stop_event.is_set():
            img = np.array(sct.grab(monitor))

            # Feeds the queues
            _put_drop_oldest(frame_queue, img)
            if recording_event.is_set():
                _put_drop_oldest(recording_queue, img)

            # Sleep until the next tick. If we're already behind
            # (grab was slow), reset the clock instead of keeping incrementing it
            next_frame += FRAME_INTERVAL
            sleep_for = next_frame - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_frame = time.perf_counter() # Reset the clock if we're behind schedule


def display_frames(
        frame_queue,
        stop_event,
        recording_queue,
        recording_event,
        screen_name="Pirairuchess"
    ):
    """Show frames from the display queue in an OpenCV window and handle keys:
    `q` quits, `r` toggles recording"""
    assert isinstance(screen_name, str) and screen_name, "screen_name must be a non-empty str"

    cv2.namedWindow(screen_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(screen_name, 1000, 700)
    recording_thread = None

    try:
        while not stop_event.is_set():
            try:
                img = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            cv2.imshow(screen_name, img)

            # ----------End of Display Logic-------------
            # |                                         |
            # ------------User's Inputs------------------

            key = cv2.waitKey(25) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                if recording_event.is_set():
                    # ---- Stop recording ----
                    recording_event.clear()          # capture stops feeding frames
                    recording_queue.put(None)        # sentinel: writer flushes and exits
                    if recording_thread is not None:
                        recording_thread.join(timeout=5)
                        recording_thread = None
                    print("Recording stopped")
                else:
                    # ---- Start recording ----
                    _drain(recording_queue) # Get a clear queue

                    path = datetime.now().strftime("recording_%Y%m%d_%H%M%S.mp4")
                    recording_thread = threading.Thread(
                        target=record_screen,
                        args=(recording_queue, stop_event),
                        kwargs={"fps": TARGET_FPS, "path": path},
                        daemon=True,
                    )
                    recording_thread.start()
                    recording_event.set()            # capture starts feeding frames
                    print(f"Recording started -> {path}")

    finally:
        stop_event.set()
        recording_event.clear()
        if recording_thread is not None:
            recording_queue.put(None)
            recording_thread.join(timeout=5)
        cv2.destroyAllWindows()

def record_screen(recording_queue, stop_event, fps=TARGET_FPS, path="videotest.mp4"):
    """Consume frames from the recording queue and write them to `path`.
    Stops on the `None` sentinel or when `stop_event` is set, then releases the file."""
    writer = None
    frames_written = 0

    try:
        while True:
            try:
                frame = recording_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_event.is_set(): # Stop if the main thread gets the flag
                    break
                continue

            if frame is None:  # sentinel from display_frames
                break

            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if writer is None:
                height, width = bgr.shape[:2]
                writer = cv2.VideoWriter(
                    path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (width, height),
                )

            writer.write(bgr)
            frames_written += 1

    finally:
        if writer is not None:
            writer.release()
            print(f"Saved {path} ({frames_written} frames)")


def screen_capture(
        selected_monitor=1,
        screen_name="Pirairuchess",
        screen_coverage=False,
        screen_coverage_size=0.7
    ):
    """Wire up the queues/events, start the capture thread, and run the display
    loop on the main thread until the user quits"""
    # Display queue: tiny, drop-oldest.
    frame_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    # Recording queue: ~5s of bufferstop a slow writer from eating all RAM
    recording_queue = queue.Queue(maxsize=TARGET_FPS * 5)
    recording_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_frame,
        args=(frame_queue, recording_queue, recording_event, stop_event, selected_monitor,
              screen_coverage, screen_coverage_size),
        daemon=True,
    )
    capture_thread.start()

    try:
        display_frames(
            frame_queue=frame_queue,
            recording_queue=recording_queue,
            stop_event=stop_event,
            recording_event=recording_event,
            screen_name=screen_name
        )
    except KeyboardInterrupt:
        print("Stopping the capture...")
    finally:
        stop_event.set()
        capture_thread.join(timeout=2)
