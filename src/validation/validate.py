import torch
import torchvision
import torchmetrics
import numpy as np
import cv2
import os
import time
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from anomalib.models import Fastflow

# 1. SAFETY MEASURES
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

# 2. CONFIGURATION & CALIBRATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "/app/v2/results/R1_flow_8/pcb_fastflow_v2/version_0/checkpoints/pcb_v2_best.ckpt"
RAW_IMG_ROOT = Path("/app/v2/data/val")
RAW_GT_ROOT = Path("/app/v2/data/val/ground_truth")
BASE_SAVE_DIR = Path("/app/v2/debug_segregated_results")

# AMPLIFICATION & MASKING CONFIG
X_START, X_END = 30, 90
Y_START, Y_END = 95, 165 
AMP_TRIGGER_THRESHOLD = -0.4500 # Widened trigger for subtle defects
PIN_ZONE_BIAS = 0.09 

PIXEL_THRESHOLD = -0.2389
IMAGE_THRESHOLD = -0.2800 # Adjusted for cleaner, foam-suppressed signals

for folder in ["TP", "TN", "FP", "FN"]:
    (BASE_SAVE_DIR / folder).mkdir(parents=True, exist_ok=True)

# 3. SYNCHRONIZED PROCESSOR
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
        aspect = h_blue / w_blue
        target_h = int(self.target_w * aspect)
        off_x, off_y = (self.canvas_dim - self.target_w) // 2, (self.canvas_dim - target_h) // 2
        dst_pts = np.float32([[off_x, off_y], [off_x+self.target_w, off_y], 
                             [off_x+self.target_w, off_y+target_h], [off_x, off_y+target_h]])
        matrix = cv2.getPerspectiveTransform(src_pts_h, dst_pts)
        return matrix, (off_y, target_h)

# 4. LOAD TOOLS
model = Fastflow.load_from_checkpoint(checkpoint_path=CKPT_PATH, map_location=DEVICE).to(DEVICE).eval()
processor = SynchronizedProcessor()
transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])

# 5. METRIC INITIALIZATION
pixel_f1 = torchmetrics.classification.BinaryF1Score().to(DEVICE)
image_accuracy = torchmetrics.classification.BinaryAccuracy().to(DEVICE)
image_recall = torchmetrics.classification.BinaryRecall().to(DEVICE)
image_precision = torchmetrics.classification.BinaryPrecision().to(DEVICE)
conf_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(DEVICE)

latencies = []

def run_validation(folder_name, label):
    img_folder = RAW_IMG_ROOT / folder_name
    gt_folder = RAW_GT_ROOT / "anomaly" if label == 1 else RAW_GT_ROOT / "good"
    images = sorted(list(img_folder.glob("*.png")))
    
    for img_path in tqdm(images, desc=f"Processing {folder_name}"):
        t_start = time.time()
        raw_bgr = cv2.imread(str(img_path))
        matrix, board_info = processor.get_warp_info(raw_bgr)
        if matrix is None: continue
        
        warped_img = cv2.warpPerspective(raw_bgr, matrix, (320, 320), flags=cv2.INTER_LANCZOS4)
        off_y, target_h = board_info
        hsv_w = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
        sil_mask = cv2.inRange(hsv_w, np.array([0, 0, 180]), np.array([180, 80, 255]))
        mom = cv2.moments(sil_mask[off_y:off_y+target_h, :])
        do_rotate = (mom["m00"] > 0 and int(mom["m01"]/mom["m00"]) < (target_h // 2))

        def finalize(img, is_mask=False):
            if do_rotate: img = cv2.rotate(img, cv2.ROTATE_180)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            s, e = processor.start_xy, processor.start_xy + processor.crop_dim
            return img[s:e, s:e]

        processed_img = finalize(warped_img)
        image_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            anomaly_map = output["anomaly_map"] if isinstance(output, dict) else output
            raw_score = (output["pred_score"] if (isinstance(output, dict) and "pred_score" in output) else anomaly_map.max()).item()

        am_np = anomaly_map.cpu().numpy().squeeze()
        am_blurred = cv2.GaussianBlur(am_np, (5, 5), 0)

        # 1. ASYMMETRIC DYNAMIC BOUNDARY PROTOCOL
        hsv_final = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
        pcb_blue_mask = cv2.inRange(hsv_final, np.array([80, 40, 40]), np.array([140, 255, 255]))
        coords = np.column_stack(np.where(pcb_blue_mask > 0))
        
        # Adaptive Right Wall
        if coords.size > 0:
            rows, counts = np.unique(coords[:, 0], return_counts=True)
            max_x_per_row = [coords[coords[:, 0] == r][:, 1].max() for r in rows]
            raw_right_x = int(np.median(max_x_per_row))
            # Safety Gate: floor of 185 + 10px buffer
            right_boundary_x = max(raw_right_x + 10, 185)
        else:
            right_boundary_x = 226 # Fallback

        # Adaptive Left Wall
        if coords.size > 0:
            min_x_per_row = [coords[coords[:, 0] == r][:, 1].min() for r in rows]
            raw_left_x = int(np.median(min_x_per_row))
            # Safety Gate: ceiling of 75 - 10px buffer. 
            # Guardrail: If detected edge > 80 (overshoot), fallback to 20.
            if raw_left_x > 80:
                left_boundary_x = 20
            else:
                left_boundary_x = min(raw_left_x - 10, 75)
        else:
            left_boundary_x = 20 # Fallback

        suppression_mask = np.ones((256, 256), dtype=bool) 
        # Apply walls globally
        suppression_mask[:, left_boundary_x:right_boundary_x] = False 
        # Asymmetric Exception: Re-expose pin zone fully for Y in range
        suppression_mask[Y_START:Y_END, 0:left_boundary_x] = False 

        am_masked = am_blurred.copy()
        am_masked[suppression_mask] = -5.0
        am_masked[:, :5] = -5.0 # Hardware crop limits
        am_masked[:, 251:] = -5.0

        # 2. CONDITIONAL AMPLIFICATION
        roi_segment = am_masked[Y_START:Y_END, X_START:X_END]
        am_masked[Y_START:Y_END, X_START:X_END] = np.where(
            roi_segment > AMP_TRIGGER_THRESHOLD, 
            roi_segment + PIN_ZONE_BIAS, 
            roi_segment
        )
        
        amplified_score = am_masked.max()
        anomaly_map_final = torch.from_numpy(am_masked).to(DEVICE).unsqueeze(0).unsqueeze(0)
        latencies.append(time.time() - t_start)

        # 3. DEBUG SYNC GROUND TRUTH
        gt_path = gt_folder / img_path.name
        gt_tensor = torch.zeros((1, 1, 256, 256)).to(DEVICE)
        processed_gt = np.zeros((256, 256), dtype=np.uint8)
        
        if gt_path.exists():
            raw_gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if raw_gt is not None and np.max(raw_gt) > 0:
                warped_gt = cv2.warpPerspective(raw_gt, matrix, (320, 320), flags=cv2.INTER_NEAREST)
                processed_gt = finalize(warped_gt, is_mask=True)
                if np.max(processed_gt) > 0:
                    # NORMALIZATION: Binary 0/1 and int32 cast for metric logic
                    gt_binary = (processed_gt > 127).astype(np.int32)
                    gt_tensor = torch.from_numpy(gt_binary).to(DEVICE).unsqueeze(0).unsqueeze(0)
                else:
                    print(f"[DEBUG] Signal lost during Warp/Rotate: {img_path.name}")
            else:
                print(f"[DEBUG] Empty or invalid GT file: {gt_path}")
        elif label == 1:
            print(f"[DEBUG] GT missing for anomaly: {img_path.name}")

        # CLASSIFICATION
        image_pred = (amplified_score > IMAGE_THRESHOLD)
        label_tensor = torch.tensor([image_pred], device=DEVICE).float()
        target_tensor = torch.tensor([label], device=DEVICE).float()
        
        if label == 1 and image_pred: category = "TP"
        elif label == 0 and not image_pred: category = "TN"
        elif label == 0 and image_pred: category = "FP"
        else: category = "FN"

        pred_mask = (anomaly_map_final > PIXEL_THRESHOLD).float()
        pixel_f1.update(pred_mask.flatten().int(), gt_tensor.flatten().int())
        image_accuracy.update(label_tensor, target_tensor)
        image_recall.update(label_tensor, target_tensor)
        image_precision.update(label_tensor, target_tensor)
        conf_matrix.update(label_tensor, target_tensor)

        # VISUALIZATION
        norm_am = cv2.normalize(am_blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_am, cv2.COLORMAP_JET)
        
        heatmap_gt_overlay = heatmap.copy()
        if gt_path.exists() and np.max(processed_gt) > 0:
            contours, _ = cv2.findContours(processed_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(heatmap_gt_overlay, contours, -1, (255, 0, 0), 2)

        # Verification panel: Visualizes adaptive walls and Pin Protection
        mask_verify = processed_img.copy()
        mask_verify[suppression_mask] = (mask_verify[suppression_mask] * 0.3).astype(np.uint8)
        cv2.line(mask_verify, (left_boundary_x, 0), (left_boundary_x, 256), (0, 255, 0), 1)
        cv2.line(mask_verify, (right_boundary_x, 0), (right_boundary_x, 256), (0, 0, 255), 1)

        vis_img = processed_img.copy()
        cv2.putText(vis_img, f"Raw: {raw_score:.4f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_img, f"Amp: {amplified_score:.4f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis_img, f"Thr: {IMAGE_THRESHOLD}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(vis_img, (X_START, Y_START), (X_END, Y_END), (255, 255, 0), 1)
        
        combined = np.hstack([
            vis_img,
            mask_verify,
            heatmap,
            cv2.cvtColor((pred_mask.cpu().numpy().squeeze()*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        ])
        cv2.imwrite(str(BASE_SAVE_DIR / category / f"res_{img_path.name}"), combined)

run_validation("good", 0)
run_validation("anomaly", 1)

avg_lat = np.mean(latencies)
print(f"\n{'='*50}\nFINAL CALIBRATION REPORT\n{'='*50}")
print(f"Throughput: {1.0/avg_lat:.2f} FPS | Mean Latency: {avg_lat*1000:.2f} ms")
print(f"Recall: {image_recall.compute():.4f} | Precision: {image_precision.compute():.4f}")
print(f"Accuracy: {image_accuracy.compute():.4f} | Pixel-level F1: {pixel_f1.compute():.4f}")
print(f"\nConfusion Matrix (TN, FP / FN, TP):\n{conf_matrix.compute().cpu().numpy().astype(int)}\n{'='*50}")
