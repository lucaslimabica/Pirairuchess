import time
from datetime import datetime
import cv2
import numpy as np
import mss
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Capture full screen screenshots for Chess dataset.")
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Monitor number to capture (default: 1 for primary monitor)"
    )
    parser.add_argument(
        "--label",
        type=str,
        default="board",
        help="The name of the output label"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=500,
        help="The duration of capture in seconds (default: 500)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=2,
        help="The interval between captures in seconds (default: 2)"
    )
    return parser.parse_args()

#TODO : Screenshoots for each X seconds to fill the dataset

def capture_full_screenshots(source_name, duration=300, interval=2):
    output_dir = f"datasets/{source_name}/"
    os.makedirs(output_dir, exist_ok=True)
    
    with mss.mss() as sct:
        monitor = sct.monitors[1] 
        start = time.time()
        count = 0
        
        while time.time() - start < duration:
            img = np.array(sct.grab(monitor))
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_%f")[:-3]
            count += 1
            filename = f"{source_name}_{count}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, bgr)
            print(f"[{source_name}] {count}: {filename} ({img.shape})")
            time.sleep(interval)
    
    print(f"[V] - Captured {count} full screenshots")

def main():
    args = parse_arguments()
    print(f"[0] - Starting full screen capture for '{args.label}' for {args.duration}s ({args.duration/args.interval})")
    capture_full_screenshots(args.label, duration=args.duration, interval=args.interval)



main()