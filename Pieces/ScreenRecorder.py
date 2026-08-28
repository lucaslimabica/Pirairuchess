import queue
import threading

import cv2
import numpy as np
import mss


def _resolve_monitor(sct, selected_monitor, screen_coverage, screen_coverage_size):
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


def capture_frame(frame_queue, stop_event, selected_monitor=1,
                  screen_coverage=False, screen_coverage_size=0.7):
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

    # mss instances are NOT thread-safe: create one inside this thread.
    with mss.mss() as sct:
        monitor = _resolve_monitor(sct, selected_monitor, screen_coverage, screen_coverage_size)

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


def display_frames(frame_queue, stop_event, screen_name="Pirairuchess"):
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


def screen_capture(selected_monitor=1, screen_name="Pirairuchess",
                   screen_coverage=False, screen_coverage_size=0.7):
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
