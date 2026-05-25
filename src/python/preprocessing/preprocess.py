import cv2
import numpy as np
import os
from pathlib import Path

class PCBBatchProcessor:
    def __init__(self, target_w=240, canvas_dim=320, crop_dim=256):
        self.target_w = target_w
        self.canvas_dim = canvas_dim
        self.crop_dim = crop_dim
        # Calculate crop start point to center the 256 box in 320 canvas
        self.start_xy = (canvas_dim - crop_dim) // 2 

    def process_frame(self, frame):
        """Applies 240px scaling on 320px canvas and returns 256x256 center-crop"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([135, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        s = box.sum(axis=1); diff = np.diff(box, axis=1)
        
        rect_pts = np.zeros((4, 2), dtype="float32")
        rect_pts[0] = box[np.argmin(s)]; rect_pts[2] = box[np.argmax(s)]
        rect_pts[1] = box[np.argmin(diff)]; rect_pts[3] = box[np.argmax(diff)]
        
        d01, d03 = np.linalg.norm(rect_pts[0]-rect_pts[1]), np.linalg.norm(rect_pts[0]-rect_pts[3])
        src_pts_h = rect_pts if d01 > d03 else np.float32([rect_pts[1], rect_pts[2], rect_pts[3], rect_pts[0]])
        h_blue, w_blue = (d03, d01) if d01 > d03 else (d01, d03)
        
        # Isotropic Scaling to 240px
        aspect = h_blue / w_blue
        target_h = int(self.target_w * aspect)
        
        off_x, off_y = (self.canvas_dim - self.target_w) // 2, (self.canvas_dim - target_h) // 2
        dst_pts = np.float32([[off_x, off_y], [off_x+self.target_w, off_y], 
                             [off_x+self.target_w, off_y+target_h], [off_x, off_y+target_h]])
        
        matrix = cv2.getPerspectiveTransform(src_pts_h, dst_pts)
        warped = cv2.warpPerspective(frame, matrix, (self.canvas_dim, self.canvas_dim), flags=cv2.INTER_LANCZOS4)
        
        # Orientation (Pins Left)
        hsv_w = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        sil_mask = cv2.inRange(hsv_w, np.array([0, 0, 180]), np.array([180, 80, 255]))
        mom = cv2.moments(sil_mask[off_y:off_y+target_h, :])
        if mom["m00"] > 0 and int(mom["m01"]/mom["m00"]) < (target_h // 2):
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        
        standardized_320 = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # FINAL STEP: The 256x256 Crop
        s = self.start_xy
        e = s + self.crop_dim
        return standardized_320[s:e, s:e]

def run_batch_preprocessing():
    # Define source structure
    BASE_SRC = Path("./data")
    BASE_DST = Path("./preprocessed_data")
    
    # Target specific sub-folders based on your request
    # train/good, train/anomaly, test/good
    sub_paths = [
        Path("train/good"),
        Path("train/anomaly"),
        Path("test/good")
    ]

    processor = PCBBatchProcessor()

    for sub in sub_paths:
        src_path = BASE_SRC / sub
        dst_path = BASE_DST / sub
        
        if not src_path.exists():
            print(f"Skipping missing directory: {src_path}")
            continue
            
        dst_path.mkdir(parents=True, exist_ok=True)
        
        print(f"--- Processing {sub} ---")
        files = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for f in files:
            img = cv2.imread(str(src_path / f))
            if img is None: continue
            
            result_256 = processor.process_frame(img)
            
            if result_256 is not None:
                cv2.imwrite(str(dst_path / f), result_256)
            else:
                print(f"Failed to extract PCB from {f}")

    print("\nBatch Preprocessing Complete. Files saved in ./preprocessed_data")

if __name__ == "__main__":
    run_batch_preprocessing()
