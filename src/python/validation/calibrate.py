import torch
import torchvision
import torchmetrics
import numpy as np
import cv2
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from anomalib.models import Fastflow

# 1. SAFETY & CONFIG (Keeping your logic identical)
torch.serialization.add_safe_globals([
    torchvision.transforms.v2._container.Compose,
    torchvision.transforms.v2._geometry.Resize,
    torchvision.transforms.v2._misc.Normalize,
    torchvision.transforms.v2._misc.ToDtype,
    torchvision.transforms.v2._type_conversion.ToImage,
    torchvision.transforms.v2._misc.Identity,
    torchvision.transforms.functional.InterpolationMode,
    torchmetrics.classification.f_beta.F1Score,
    torchmetrics.classification.precision_recall_curve.BinaryPrecisionRecallCurve
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "/app/v2/results/R1_flow_8/pcb_fastflow_v2/version_0/checkpoints/pcb_v2_best.ckpt"
RAW_IMG_ROOT = Path("/app/v2/data/val")
RAW_GT_ROOT = Path("/app/v2/data/val/ground_truth")
BASE_SAVE_DIR = Path("/app/v2/debug_segregated_results")

X_START, X_END = 30, 90
Y_START, Y_END = 90, 160
PIN_ZONE_BIAS = 0.0 
IMAGE_THRESHOLD = -0.2620 # Your current baseline

# NEW: Score storage for statistical analysis
all_results = [] 

# 3. SYNCHRONIZED PROCESSOR (Logic Unchanged)
class SynchronizedProcessor:
    def __init__(self, target_w=240, canvas_dim=320, crop_dim=256):
        self.target_w = target_w
        self.canvas_dim = canvas_dim
        self.crop_dim = crop_dim
        self.start_xy = (canvas_dim - crop_dim) // 2 

    def get_warp_info(self, frame):
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
        target_h = int(self.target_w * (h_blue/w_blue))
        off_x, off_y = (self.canvas_dim - self.target_w) // 2, (self.canvas_dim - target_h) // 2
        dst_pts = np.float32([[off_x, off_y], [off_x+self.target_w, off_y], 
                             [off_x+self.target_w, off_y+target_h], [off_x, off_y+target_h]])
        return cv2.getPerspectiveTransform(src_pts_h, dst_pts), (off_y, target_h)

# 4. LOAD TOOLS
model = Fastflow.load_from_checkpoint(checkpoint_path=CKPT_PATH, map_location=DEVICE).to(DEVICE).eval()
processor = SynchronizedProcessor()
transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])

def run_calibration_loop(folder_name, label):
    img_folder = RAW_IMG_ROOT / folder_name
    images = sorted(list(img_folder.glob("*.png")))
    
    for img_path in tqdm(images, desc=f"Analyzing {folder_name}"):
        raw_bgr = cv2.imread(str(img_path))
        matrix, board_info = processor.get_warp_info(raw_bgr)
        if matrix is None: continue
        
        warped_img = cv2.warpPerspective(raw_bgr, matrix, (320, 320), flags=cv2.INTER_LANCZOS4)
        off_y, target_h = board_info
        
        # Rotation logic
        hsv_w = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
        sil_mask = cv2.inRange(hsv_w, np.array([0, 0, 180]), np.array([180, 80, 255]))
        mom = cv2.moments(sil_mask[off_y:off_y+target_h, :])
        do_rotate = (mom["m00"] > 0 and int(mom["m01"]/mom["m00"]) < (target_h // 2))

        def finalize(img):
            if do_rotate: img = cv2.rotate(img, cv2.ROTATE_180)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            s, e = processor.start_xy, processor.start_xy + processor.crop_dim
            return img[s:e, s:e]

        processed_img = finalize(warped_img)
        input_tensor = transform(image=cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))["image"].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            anomaly_map = output["anomaly_map"] if isinstance(output, dict) else output
            am_np = anomaly_map.cpu().numpy().squeeze()
            am_blurred = cv2.GaussianBlur(am_np, (5, 5), 0)

            # Numerical Masking
            am_masked = am_blurred.copy()
            am_masked[:, :20] = -5.0
            am_masked[:, 226:] = -5.0
            
            # Amplification
            am_amplified = am_masked.copy()
            am_amplified[Y_START:Y_END, X_START:X_END] += PIN_ZONE_BIAS
            amplified_score = float(am_amplified.max())

        # LOGGING FOR HISTOGRAM
        all_results.append({
            "score": amplified_score,
            "label": "Anomaly" if label == 1 else "Good",
            "filename": img_path.name
        })

# Execute Data Collection
run_calibration_loop("good", 0)
run_calibration_loop("anomaly", 1)

# 5. STATISTICAL ANALYSIS & PLOTTING
df = pd.DataFrame(all_results)

# Calculate professional min/max
good_stats = df[df['label'] == 'Good']['score'].describe()
anomaly_stats = df[df['label'] == 'Anomaly']['score'].describe()

print(f"\n{'='*50}\nSCORE DISTRIBUTION REPORT\n{'='*50}")
print(f"GOOD Class    | Min: {good_stats['min']:.4f} | Max: {good_stats['max']:.4f} | Mean: {good_stats['mean']:.4f}")
print(f"ANOMALY Class | Min: {anomaly_stats['min']:.4f} | Max: {anomaly_stats['max']:.4f} | Mean: {anomaly_stats['mean']:.4f}")

# Find the overlap/gap
gap = anomaly_stats['min'] - good_stats['max']
if gap > 0:
    print(f"RESULT: CLEAR GAP FOUND ({gap:.4f}). Threshold can safely move to {good_stats['max'] + gap/2:.4f}")
else:
    print(f"RESULT: CLASS OVERLAP DETECTED ({abs(gap):.4f}). Overlap occurs between {anomaly_stats['min']:.4f} and {good_stats['max']:.4f}")

# Plotting
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x="score", hue="label", kde=True, element="step", palette={"Good": "green", "Anomaly": "red"})
plt.axvline(IMAGE_THRESHOLD, color='blue', linestyle='--', label=f'Current Threshold ({IMAGE_THRESHOLD})')
plt.title("Anomaly Score Distribution: Good vs Anomaly")
plt.xlabel("Amplified Max Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("score_distribution.png")
print(f"\nHistogram saved as 'score_distribution.png'. Analyze the overlap to adjust thresholds.")
