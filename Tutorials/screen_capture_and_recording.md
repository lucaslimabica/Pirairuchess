# Screen Capture and Recording

How the live screen is shown on a window and, on demand, saved to an `.mp4`.
The working code is at [ScreenRecorder.py](/Pieces/ScreenRecorder.py).

The whole thing is 3 jobs running at the same time, talking through queues:

```
capture thread            main thread (display)          recording thread
--------------            --------------------          ----------------
grab frame with mss  -->  frame_queue  -->  show it in an OpenCV window
                                            reads the keyboard (q / r)
grab frame with mss  -->  recording_queue  ------------>  write frame to .mp4
   (only while recording)
```

## Why threads and queues

- `mss.grab()` is slow-ish, `cv2.imshow()` is slow-ish, `VideoWriter.write()` is slow-ish.
  Doing them one after another in a single loop makes the FPS drop.
- Splitting them into threads lets each run at its own pace.
- A `queue.Queue` is the safe way for threads to hand data to each other
  (no shared list, no locks to manage by hand).

## The pieces

### 1. Choosing what to grab — `_resolve_monitor`

Turns `selected_monitor` / `screen_coverage` into the `{left, top, width, height}`
dict that `mss` wants. See [mss.md](/Tutorials/mss.md) for what `sct.monitors` looks like.

### 2. Capture thread — `capture_frame`

Loop:
1. `img = np.array(sct.grab(monitor))`
2. push to `frame_queue` (for display) — always
3. push to `recording_queue` — only while `recording_event` is set
4. sleep so the loop runs at ~`TARGET_FPS`, not as fast as the CPU allows

The sleep matters: without it the loop runs hundreds of times/sec, the video
plays back too fast, and the recording queue fills up.

### 3. Display loop — `display_frames` (runs on the main thread)

> OpenCV windows / `waitKey` must run on the main thread. That's why this is
> not a thread.

Loop:
1. `img = frame_queue.get(timeout=1)`
2. `cv2.imshow(...)`
3. `key = cv2.waitKey(25)`
4. `q` -> quit, `r` -> toggle recording (start/stop the recording thread)

### 4. Recording thread — `record_screen`

Loop:
1. `frame = recording_queue.get(timeout=0.5)`
2. stop if the frame is `None` (the "we're done" signal) or `stop_event` is set
3. convert `BGRA -> BGR` (OpenCV writes 3 channels, not 4)
4. create the `VideoWriter` on the first frame (needs the frame size)
5. `writer.write(frame)`
6. on exit: `writer.release()` so the file is valid

### 5. Wiring — `screen_capture`

Creates the queues + events, starts the capture thread, then runs the display
loop. On exit, sets `stop_event` and joins the threads.

## Key concepts to keep when redoing it

| Concept | What it does | Where |
|---|---|---|
| `threading.Event` | on/off flag shared between threads | `stop_event`, `recording_event` |
| `queue.Queue(maxsize=N)` | bounded buffer between threads (prevents RAM blowup) | both queues |
| drop-oldest on full | for *live* data, a fresh frame beats a complete history | `_put_drop_oldest` |
| sentinel value (`None`) | tell a consumer "no more data, stop" | `record_screen` |
| `get(timeout=...)` | wait for data but stay responsive to stop flags | display + recording loops |
| bounded queue | slow writer can't make memory grow forever | `recording_queue` |

## Things that bit me (don't repeat)

- **Unbounded `recording_queue`** + no FPS cap on capture = queue grows forever = 25 GB RAM.
  Fix: `maxsize` on the queue + pace the capture loop.
- **One `Event` meaning two things.** `recording_event` should mean exactly
  "capture, please feed the recording queue" — nothing else reads it.
- **FPS mismatch.** `VideoWriter` is told `30.0`; if capture doesn't actually
  produce ~30 fps, playback speed is wrong.
- **`BGRA` vs `BGR`.** `mss` gives 4 channels; `VideoWriter` wants 3.

## Ideas for the rewrite

- Measure real FPS and pass it to `VideoWriter` (or duplicate/drop frames to hit a target).
- Move all the shared state (queues, events, threads) into a `ScreenRecorder` class.
- Add a small on-screen "REC" indicator while recording.
- Let the output folder / filename be configurable.
