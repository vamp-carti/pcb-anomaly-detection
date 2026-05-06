import cv2
import numpy as np
import os
from pathlib import Path

class MaskBatchProcessor:
    def __init__(self, target_w=240, canvas_dim=320, crop_dim=256):
        self.target_w = target_w
        self.canvas_dim = canvas_dim
        self.crop_dim = crop_dim
        self.start_xy = (canvas_dim - crop_dim) // 2

    def get_warp_matrix(self, frame):
        """Calculates geometry from the raw RGB frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([135, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, None
        
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
        
        aspect = h_blue / w_blue
        target_h = int(self.target_w * aspect)
        off_x, off_y = (self.canvas_dim - self.target_w) // 2, (self.canvas_dim - target_h) // 2
        
        dst_pts = np.float32([[off_x, off_y], [off_x+self.target_w, off_y], 
                             [off_x+self.target_w, off_y+target_h], [off_x, off_y+target_h]])
        
        return cv2.getPerspectiveTransform(src_pts_h, dst_pts), (off_y, target_h)

    def process_mask(self, img_path, mask_path):
        raw_img = cv2.imread(str(img_path))
        raw_mask = cv2.imread(str(mask_path))
        if raw_img is None or raw_mask is None: return None

        matrix, board_info = self.get_warp_matrix(raw_img)
        if matrix is None: return None
        off_y, target_h = board_info

        # Apply warp: Image uses LANCZOS, Mask uses NEAREST
        warped_img = cv2.warpPerspective(raw_img, matrix, (self.canvas_dim, self.canvas_dim), flags=cv2.INTER_LANCZOS4)
        warped_mask = cv2.warpPerspective(raw_mask, matrix, (self.canvas_dim, self.canvas_dim), flags=cv2.INTER_NEAREST)

        # Correct Orientation (Pins Left)
        hsv_w = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
        sil_mask = cv2.inRange(hsv_w, np.array([0, 0, 180]), np.array([180, 80, 255]))
        mom = cv2.moments(sil_mask[off_y:off_y+target_h, :])
        if mom["m00"] > 0 and int(mom["m01"]/mom["m00"]) < (target_h // 2):
            warped_mask = cv2.rotate(warped_mask, cv2.ROTATE_180)
            warped_img = cv2.rotate(warped_img, cv2.ROTATE_180) # Rotate img to stay in sync

        final_mask = cv2.rotate(warped_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Final 256x256 Crop
        s, e = self.start_xy, self.start_xy + self.crop_dim
        return final_mask[s:e, s:e]

def run_gt_batch():
    # Define absolute paths per request
    GT_IN = Path("/app/v2/data/ground_truth")
    GT_OUT = Path("/app/v2/preprocessed_data/ground_truth")
    
    # Mapping Ground Truth folders to their source Image folders
    configs = [
        {"gt_sub": "good", "img_dir": "/app/v2/data/train/good"},
        {"gt_sub": "anomaly", "img_dir": "/app/v2/data/test/anomaly"}
    ]

    processor = MaskBatchProcessor()

    for cfg in configs:
        src_gt_dir = GT_IN / cfg["gt_sub"]
        dst_gt_dir = GT_OUT / cfg["gt_sub"]
        img_dir = Path(cfg["img_dir"])

        if not src_gt_dir.exists():
            print(f"Skipping: {src_gt_dir} not found.")
            continue
            
        dst_gt_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- Processing GT: {cfg['gt_sub']} ---")

        for mask_file in os.listdir(src_gt_dir):
            if mask_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = img_dir / mask_file
                mask_path = src_gt_dir / mask_file

                if not img_path.exists():
                    print(f"Warning: No matching image for mask {mask_file}")
                    continue

                processed_gt = processor.process_mask(img_path, mask_path)
                if processed_gt is not None:
                    cv2.imwrite(str(dst_gt_dir / mask_file), processed_gt)

    print("\nGround Truth Preprocessing Complete.")

if __name__ == "__main__":
    run_gt_batch()
