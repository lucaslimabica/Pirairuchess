import os
import queue
import threading
import time

import cv2
import numpy as np
import mss


def _resolve_monitor(
        sct,
        selected_monitor,
        screen_coverage,
        screen_coverage_size
    ):
    """
    Translate the user's monitor choice into the concrete bounding box that
    ``mss`` needs for a grab.

    ``mss`` exposes ``sct.monitors`` as a list where index ``0`` is the virtual
    screen spanning every physical display and indices ``1..N`` are the
    individual monitors. This helper selects one of those entries and, when a
    coverage crop is requested, shrinks it to the top-left rectangle of the
    monitor so downstream processing works on a smaller image.

    Args:
        sct (mss.base.MSSBase): An open ``mss`` instance. Only ``sct.monitors``
            is read; the object is never mutated.
        selected_monitor (int): Desired index into ``sct.monitors``. ``1`` is
            the first physical monitor. ``0`` (the combined virtual screen) and
            any value ``>= len(sct.monitors)`` are rejected and trigger the
            primary-monitor fallback: the first entry whose ``is_primary`` flag
            is set, or ``sct.monitors[1]`` if none advertises it.
        screen_coverage (bool): When ``False`` the full monitor rectangle is
            returned unchanged. When ``True`` the returned rectangle keeps the
            monitor's ``left``/``top`` origin but its ``width``/``height`` are
            scaled down by ``screen_coverage_size``.
        screen_coverage_size (float): Scaling factor in the range ``(0, 1]``
            applied to width and height when ``screen_coverage`` is ``True``.
            ``0.7`` captures the top-left 70% of the monitor in each axis
            (49% of its area). Ignored when ``screen_coverage`` is ``False``.

    Returns:
        dict: A monitor rectangle with integer ``left``, ``top``, ``width`` and
        ``height`` keys, ready to hand to ``sct.grab()``. When
        ``screen_coverage`` is ``False`` this is the exact dict from
        ``sct.monitors`` (do not mutate it); otherwise it is a fresh dict.

    Side effects:
        Prints a notice to stdout when the primary-monitor fallback is taken.
    """
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


def capture_frame(
        frame_queue,
        stop_event,
        selected_monitor=1,
        screen_coverage=False,
        screen_coverage_size=0.7
    ):
    """
    Producer thread: grab the selected screen region in a tight loop and hand
    each frame to the consumer through ``frame_queue``.

    This is the ``target`` of the capture ``threading.Thread`` started by
    :func:`screen_capture`. It owns its own ``mss`` instance because ``mss``
    objects are not safe to share between threads. The loop runs until
    ``stop_event`` is set, re-checking the flag once per grab, so shutdown
    latency is one frame time.

    Back-pressure policy: the queue is bounded and this function never blocks on
    it. When the queue is full (the consumer has fallen behind) the oldest
    frame is discarded and the newest one takes its place, so the consumer
    always sees near-real-time frames instead of a growing backlog. Frames that
    cannot be enqueued are simply dropped; the next iteration grabs a fresher
    one.

    Args:
        frame_queue (queue.Queue): Bounded queue receiving BGRA frames as
            ``numpy.ndarray`` of shape ``(H, W, 4)`` and dtype ``uint8`` (the
            raw layout returned by ``numpy.array(sct.grab(...))``). Must be
            created with a finite ``maxsize`` for the drop-oldest policy to
            take effect.
        stop_event (threading.Event): Shared shutdown signal. While it stays
            clear the loop keeps capturing; once any thread sets it the loop
            exits, the ``with mss.mss()`` block closes, and the thread returns.
        selected_monitor (int): Monitor index passed through to
            :func:`_resolve_monitor`. ``1`` is the first physical monitor;
            ``0`` and out-of-range values fall back to the primary monitor.
            Defaults to ``1``.
        screen_coverage (bool): When ``True`` only the top-left
            ``screen_coverage_size`` fraction of the monitor is grabbed;
            when ``False`` the whole monitor is grabbed. Defaults to ``False``.
        screen_coverage_size (float): Crop factor in ``(0, 1]`` used only when
            ``screen_coverage`` is ``True``. Defaults to ``0.7``.

    Raises:
        AssertionError: If any argument fails its type or range check. The
            checks run before the capture loop starts, so a bad call fails
            fast inside the thread.

    Returns:
        None: The function only returns when ``stop_event`` is set.
    """
    assert isinstance(selected_monitor, int) and not isinstance(selected_monitor, bool), \
        "selected_monitor must be an int"
    assert selected_monitor >= 0, "selected_monitor must be >= 0"
    assert isinstance(screen_coverage, bool), "screen_coverage must be a bool"
    assert isinstance(screen_coverage_size, (int, float)) and not isinstance(screen_coverage_size, bool), \
        "screen_coverage_size must be a number"
    assert 0 < screen_coverage_size <= 1, "screen_coverage_size must be in the range (0, 1]"

    with mss.mss() as sct:
        monitor = _resolve_monitor(sct, selected_monitor, screen_coverage, screen_coverage_size) # get the selected monitor

        while not stop_event.is_set():
            img = np.array(sct.grab(monitor))

            # Keep only the freshest frame: drop the stale one if the
            # consumer has fallen behind.
            try:
                frame_queue.put_nowait(img) # Tries to put into the queue
            except queue.Full:
                try:
                    frame_queue.get_nowait() # Remove the oldest frame to try again
                except queue.Empty:
                    pass
                try:
                    frame_queue.put_nowait(img) # Try again the new frame
                except queue.Full:
                    pass


def display_frames(
        frame_queue,
        stop_event,
        screen_name="Pirairuchess"
    ):
    """
    Consumer loop: pull frames from ``frame_queue`` and render them in a
    resizable OpenCV window until the user quits.

    This runs on the **main thread**, not in a worker: OpenCV's HighGUI
    (``namedWindow``/``imshow``/``waitKey``) is not thread-safe and misbehaves
    or crashes when driven from a background thread. :func:`screen_capture`
    therefore calls this directly and keeps the capture work on the spawned
    thread.

    The loop exits when any of the following happens, and in every case the
    ``finally`` block sets ``stop_event`` (so the producer stops too) and
    destroys the OpenCV windows:

    * the user presses ``q`` while the window is focused (``break``);
    * another thread sets ``stop_event`` (the ``while`` guard goes false);
    * an unhandled exception propagates out of the loop body.

    An empty queue is not an error: ``get`` times out after one second, the
    ``except`` re-enters the loop, and the ``stop_event`` guard is re-checked,
    which guarantees the loop cannot hang if the producer dies without
    enqueuing anything.

    Args:
        frame_queue (queue.Queue): Same queue the producer fills. Items are
            expected to be ``numpy.ndarray`` images accepted by
            ``cv2.imshow`` (BGRA ``uint8`` as produced by :func:`capture_frame`).
        stop_event (threading.Event): Shared shutdown signal. Read on every
            iteration to allow an external stop, and set unconditionally in the
            ``finally`` block on the way out.
        screen_name (str): Title of the OpenCV window and the key OpenCV uses to
            identify it across ``namedWindow``/``imshow``/``destroyAllWindows``.
            Must be a non-empty string. Defaults to ``"Pirairuchess"``.

    Raises:
        AssertionError: If ``screen_name`` is not a non-empty string.

    Returns:
        None
    """
    assert isinstance(screen_name, str) and screen_name, "screen_name must be a non-empty str"

    cv2.namedWindow(screen_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(screen_name, 1000, 700)

    try:
        while not stop_event.is_set():
            try:
                img = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            cv2.imshow(screen_name, img)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        cv2.destroyAllWindows()


def screen_capture(
        selected_monitor=1,
        screen_name="Pirairuchess",
        screen_coverage=False,
        screen_coverage_size=0.7
    ):
    """
    Entry point: wire up the producer/consumer pair and run a live screen
    capture session until the user quits.

    Creates the shared ``frame_queue`` and ``stop_event``, starts
    :func:`capture_frame` on a daemon thread, then runs :func:`display_frames`
    on the calling (main) thread so OpenCV's GUI stays on the main thread. When
    the display loop returns -- ``q`` pressed, ``stop_event`` set elsewhere, or
    an error -- or the user sends ``Ctrl+C``, the ``finally`` block sets
    ``stop_event`` and waits up to two seconds for the capture thread to unwind
    its ``mss`` context. The thread is a daemon, so even if that join times out
    the process can still exit.

    The queue is capped at two frames: one in flight for the consumer plus one
    spare, which is enough to hide a single slow grab without letting display
    latency build up.

    Args:
        selected_monitor (int): Monitor index to capture, forwarded to
            :func:`capture_frame` / :func:`_resolve_monitor`. ``1`` is the
            first physical monitor; ``0`` and out-of-range values fall back to
            the primary monitor. Defaults to ``1``.
        screen_name (str): Non-empty title for the preview window, forwarded to
            :func:`display_frames`. Defaults to ``"Pirairuchess"``.
        screen_coverage (bool): When ``True`` capture only the top-left
            ``screen_coverage_size`` fraction of the monitor. Defaults to
            ``False``.
        screen_coverage_size (float): Crop factor in ``(0, 1]`` used when
            ``screen_coverage`` is ``True``. Defaults to ``0.7``.

    Raises:
        AssertionError: Propagated from :func:`capture_frame` or
            :func:`display_frames` if an argument fails validation.

    Returns:
        None: Returns after both the display loop and the capture thread have
        stopped.
    """
    frame_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_frame,
        args=(frame_queue, stop_event, selected_monitor,
              screen_coverage, screen_coverage_size),
        daemon=True,
    )
    capture_thread.start()

    try:
        display_frames(frame_queue, stop_event, screen_name)
    except KeyboardInterrupt:
        print("Stopping the capture...")
    finally:
        stop_event.set()
        capture_thread.join(timeout=2)


# ---------------------------------------------------------------------------
# On-demand video recording
# ---------------------------------------------------------------------------

RECORD_FPS = 20.0


def _video_writer_loop(record_queue, output_path, fps):
    """
    Writer thread: drain ``record_queue`` to a video file until told to stop.

    Runs alongside the live preview while a recording is active. The
    ``cv2.VideoWriter`` is created lazily from the first frame so its size
    always matches the real capture resolution (a mismatch makes OpenCV write
    an empty file). Frames arrive as BGRA (``mss``'s layout) and are converted
    to the BGR that ``VideoWriter`` expects.

    The loop ends when it receives the ``None`` sentinel that
    :func:`record_screen` enqueues on "stop recording" / shutdown.

    Args:
        record_queue (queue.Queue): Unbounded queue of BGRA ``numpy.ndarray``
            frames, terminated by a single ``None`` sentinel. Unbounded so that
            a momentarily slow disk never forces a recorded frame to be
            dropped.
        output_path (str): Destination file. The container/codec pair is
            ``.mp4`` + ``mp4v``; change both together if you need another.
        fps (float): Frame rate stamped into the file. Must match the rate at
            which :func:`record_screen` actually feeds the queue, otherwise
            playback runs fast or slow.

    Returns:
        None
    """
    writer = None
    try:
        while True:
            frame = record_queue.get()
            if frame is None:            # sentinel -> stop recording
                break

            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if writer is None:
                height, width = bgr.shape[:2]
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            writer.write(bgr)
    finally:
        if writer is not None:
            writer.release()


def record_screen(
        selected_monitor=1,
        screen_name="Pirairuchess",
        screen_coverage=False,
        screen_coverage_size=0.7,
        output_dir="recordings",
        fps=RECORD_FPS,
    ):
    """
    Live preview of the captured monitor with **on-demand** video recording.

    Same producer/consumer skeleton as :func:`screen_capture` -- a daemon
    capture thread fills a small drop-oldest ``frame_queue``, this function
    displays it on the main thread -- plus a third stage that only exists while
    a recording is running:

    * press ``r`` (with the preview window focused) to start recording; a new
      timestamped ``.mp4`` is opened under ``output_dir`` and a
      :func:`_video_writer_loop` thread starts draining an unbounded
      ``record_queue``;
    * press ``r`` again to stop: a ``None`` sentinel is queued, the writer
      thread finishes the file and is joined;
    * press ``q`` (or ``Ctrl+C``) to quit, which also stops any active
      recording cleanly.

    Recording captures exactly what the preview shows (the frames coming off
    ``frame_queue``). Frames are handed to the writer at a fixed ``fps`` cadence
    -- regardless of how fast the capture loop actually spins -- so the saved
    video plays back at real-time speed. If the loop stalls, the cadence
    resynchronises to "now" instead of trying to catch up with a burst.

    Args:
        selected_monitor (int): Monitor index, forwarded to
            :func:`capture_frame` / :func:`_resolve_monitor`. ``1`` is the first
            physical monitor; ``0`` and out-of-range values fall back to the
            primary monitor. Defaults to ``1``.
        screen_name (str): Non-empty preview-window title. Defaults to
            ``"Pirairuchess"``.
        screen_coverage (bool): When ``True`` capture only the top-left
            ``screen_coverage_size`` fraction of the monitor. Defaults to
            ``False``.
        screen_coverage_size (float): Crop factor in ``(0, 1]`` used when
            ``screen_coverage`` is ``True``. Defaults to ``0.7``.
        output_dir (str): Directory for the ``.mp4`` files, created if missing.
            Defaults to ``"recordings"``.
        fps (float): Frame rate for the output files and the rate at which
            frames are fed to the writer. Must be > 0. Defaults to
            :data:`RECORD_FPS`.

    Raises:
        AssertionError: If ``fps`` is not a positive number, ``output_dir`` is
            not a non-empty string, or an argument fails a check inside
            :func:`capture_frame` / :func:`display_frames`.

    Returns:
        None: Returns once the preview loop and every worker thread have
        stopped.
    """
    assert isinstance(output_dir, str) and output_dir, "output_dir must be a non-empty str"
    assert isinstance(fps, (int, float)) and not isinstance(fps, bool), "fps must be a number"
    assert fps > 0, "fps must be > 0"

    frame_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_frame,
        args=(frame_queue, stop_event, selected_monitor,
              screen_coverage, screen_coverage_size),
        daemon=True,
    )
    capture_thread.start()

    cv2.namedWindow(screen_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(screen_name, 1000, 700)

    frame_interval = 1.0 / fps
    recorder = {"queue": None, "thread": None, "next_ts": 0.0}

    def start_recording():
        if recorder["thread"] is not None:
            return
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, time.strftime("capture_%Y%m%d_%H%M%S.mp4"))
        record_queue = queue.Queue()                     # unbounded: never drop
        thread = threading.Thread(
            target=_video_writer_loop,
            args=(record_queue, path, fps),
            daemon=True,
        )
        thread.start()
        recorder.update(queue=record_queue, thread=thread, next_ts=time.time())
        print(f"Recording -> {path}")

    def stop_recording():
        if recorder["thread"] is None:
            return
        recorder["queue"].put(None)                      # sentinel
        recorder["thread"].join(timeout=5)
        recorder.update(queue=None, thread=None)
        print("Recording stopped")

    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            cv2.imshow(screen_name, frame)

            if recorder["thread"] is not None:
                now = time.time()
                if now >= recorder["next_ts"]:
                    recorder["queue"].put(frame)
                    recorder["next_ts"] += frame_interval
                    if now - recorder["next_ts"] > frame_interval:
                        recorder["next_ts"] = now + frame_interval   # resync after a stall

            key = cv2.waitKey(25) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                stop_recording() if recorder["thread"] is not None else start_recording()
    except KeyboardInterrupt:
        print("Stopping the capture...")
    finally:
        stop_event.set()
        stop_recording()
        capture_thread.join(timeout=2)
        cv2.destroyAllWindows()
