from Pieces.ScreenRecorder import record_screen, screen_capture

if __name__ == "__main__":
    # screen_capture(selected_monitor=1, screen_coverage=True)  # preview only
    record_screen(selected_monitor=1)  # preview + press "r" to start/stop recording
