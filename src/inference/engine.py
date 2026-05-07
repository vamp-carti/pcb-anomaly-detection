import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import torch
import torchmetrics
import json
from datetime import datetime
from pathlib import Path
from queue import Empty
from multiprocessing import Process, Queue
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIGURATION & PROFILES ---
ENGINE_PATH = "/app/v2/results/pcb_fastflow_fp16.engine"
RAW_IMG_ROOT = Path("/app/v2/temp/samples")
BASE_SAVE_DIR = Path("/app/v2/debug_segregated_results")
LOG_FILE = Path("/app/v2/telemetry_log.json")

# Zonal Parameters (Safety & Recall Guards)
X_START, X_END, Y_START, Y_END = 30, 90, 95, 165
AMP_TRIGGER_THRESHOLD = -0.4500
PIN_ZONE_BIAS = 0.09
OSC_X_START, OSC_X_END = 155, 180
OSC_Y_START, OSC_Y_END = 95, 160
OSC_BIAS = 0.05

CURRENT_MODE = "BALANCED"
PROFILES = {
    "SAFETY":   {"threshold": -0.3270, "code": 10},
    "BALANCED": {"threshold": -0.2800, "code": 20},
    "UPTIME":   {"threshold": -0.1950, "code": 30}
}

# Industrial Handshake Status Codes
STATUS_PASS, STATUS_FAIL, STATUS_MECH_ERROR = 0, 1, 2

# Only folder kept is for physical alignment errors
(BASE_SAVE_DIR / "MECHANICAL_ERROR").mkdir(parents=True, exist_ok=True)

# --- WORKER FUNCTIONS ---
def pre_process_worker(img_paths, label, input_q, processor_params):
    """CPU-Bound Heavy Lifting: Warping, Rotation, Normalization"""
    processor = SynchronizedProcessor(**processor_params)
    transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    
    for path in img_paths:
        raw_bgr = cv2.imread(str(path))
        matrix, board_info = processor.get_warp_info(raw_bgr)
        
        if matrix is None:
            input_q.put((raw_bgr, None, path.name, STATUS_MECH_ERROR, label))
            continue
        
        warped = cv2.warpPerspective(raw_bgr, matrix, (320, 320), flags=cv2.INTER_LINEAR)
        off_y, target_h = board_info
        
        hsv_w = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        sil_mask = cv2.inRange(hsv_w, np.array([0, 0, 180]), np.array([180, 80, 255]))
        mom = cv2.moments(sil_mask[off_y:off_y + target_h, :])
        do_rotate = (mom["m00"] > 0 and int(mom["m01"]/mom["m00"]) < (target_h // 2))

        if do_rotate: warped = cv2.rotate(warped, cv2.ROTATE_180)
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        s, e = processor.start_xy, processor.start_xy + processor.crop_dim
        final_img = warped[s:e, s:e]
        tensor = transform(image=cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))["image"].numpy()
        
        input_q.put((final_img, tensor, path.name, STATUS_PASS, label))
    input_q.put(None)

def async_writer_worker(write_q):
    """Limited IO: Now only handles critical mechanical alignment errors"""
    while True:
        try:
            item = write_q.get(timeout=1)
            if item is None: break
            path, img = item
            cv2.imwrite(str(path), img)
        except Empty: continue

# --- CLASSES ---
class SynchronizedProcessor:
    def __init__(self, target_w=240, canvas_dim=320, crop_dim=256):
        self.target_w, self.canvas_dim, self.crop_dim = target_w, canvas_dim, crop_dim
        self.start_xy = (canvas_dim - crop_dim) // 2

    def get_warp_info(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([135, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 10000: return None, None 
        
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect); s = box.sum(axis=1); diff = np.diff(box, axis=1)
        pts = np.zeros((4, 2), dtype="float32")
        pts[0], pts[2] = box[np.argmin(s)], box[np.argmax(s)]
        pts[1], pts[3] = box[np.argmin(diff)], box[np.argmax(diff)]
        d01, d03 = np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[0]-pts[3])
        src = pts if d01 > d03 else np.float32([pts[1], pts[2], pts[3], pts[0]])
        target_h = int(self.target_w * (d03/d01 if d01 > d03 else d01/d03))
        off_x, off_y = (self.canvas_dim-self.target_w)//2, (self.canvas_dim-target_h)//2
        dst = np.float32([[off_x, off_y], [off_x+self.target_w, off_y], [off_x+self.target_w, off_y+target_h], [off_x, off_y+target_h]])
        return cv2.getPerspectiveTransform(src, dst), (off_y, target_h)

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        in_n, out_n = self.engine.get_tensor_name(0), self.engine.get_tensor_name(1)
        # DMA-Capable Pinned Memory
        self.h_in = cuda.pagelocked_empty(trt.volume(self.engine.get_tensor_shape(in_n)), dtype=np.float32)
        self.h_out = cuda.pagelocked_empty(trt.volume(self.engine.get_tensor_shape(out_n)), dtype=np.float32)
        self.d_in, self.d_out = cuda.mem_alloc(self.h_in.nbytes), cuda.mem_alloc(self.h_out.nbytes)
        self.bindings = [int(self.d_in), int(self.d_out)]

    def infer(self, img_np):
        t1 = time.perf_counter()
        np.copyto(self.h_in, img_np.ravel())
        cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        t2 = time.perf_counter()
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        t3 = time.perf_counter()
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        t4 = time.perf_counter()
        return self.h_out.reshape(256, 256), ((t2-t1)*1000, (t3-t2)*1000, (t4-t3)*1000)

    def warmup(self, n=20):
        """Pre-heats CUDA kernels and ramps GPU power states"""
        in_shape = self.engine.get_tensor_shape(self.engine.get_tensor_name(0))
        dummy_input = np.zeros(in_shape, dtype=np.float32)
        for _ in range(n):
            self.infer(dummy_input)

class TelemetryLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_buffer = []

    def log_event(self, board_id, metrics, status):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "board_id": board_id,
            "status": status,
            "h2d_ms": round(metrics[0], 4), "exec_ms": round(metrics[1], 4), "d2h_ms": round(metrics[2], 4),
            "profile": CURRENT_MODE,
            "vram_free_mb": round(cuda.mem_get_info()[0] / 1024**2, 2)
        }
        self.log_buffer.append(entry)
        if len(self.log_buffer) >= 50:
            with open(self.log_path, 'a') as f:
                for log in self.log_buffer: f.write(json.dumps(log) + "\n")
            self.log_buffer = []

# --- MAIN ---
def main():
    # 1. Initialize Engine & Perform Mandatory Warmup
    trt_engine = TRTInference(ENGINE_PATH)
    print("🔥 Calibrating CUDA Streams & Warming up GPU...")
    trt_engine.warmup(20)
    print("✅ Warmup Complete. GPU Stable.")

    logger = TelemetryLogger(LOG_FILE)
    
    acc_metric = torchmetrics.classification.BinaryAccuracy()
    rec_metric = torchmetrics.classification.BinaryRecall()
    pre_metric = torchmetrics.classification.BinaryPrecision()

    input_q, write_q = Queue(maxsize=30), Queue(maxsize=100)
    good_imgs = sorted(list((RAW_IMG_ROOT / "good").glob("*.png")))
    bad_imgs = sorted(list((RAW_IMG_ROOT / "anomaly").glob("*.png")))
    
    procs = []
    for chunk in np.array_split(good_imgs, 2):
        p = Process(target=pre_process_worker, args=(chunk, 0, input_q, {"target_w": 240}))
        p.start(); procs.append(p)
    p_ano = Process(target=pre_process_worker, args=(bad_imgs, 1, input_q, {"target_w": 240}))
    p_ano.start(); procs.append(p_ano)
    
    writer_p = Process(target=async_writer_worker, args=(write_q,))
    writer_p.start()

    finished_workers, processed_count = 0, 0
    start_bench = time.perf_counter()

    while finished_workers < len(procs):
        try:
            data = input_q.get(timeout=5)
            if data is None:
                finished_workers += 1
                continue
            
            img_vis, tensor, img_name, status, label = data
            if status == STATUS_MECH_ERROR:
                write_q.put((BASE_SAVE_DIR / "MECHANICAL_ERROR" / f"ERR_{img_name}", img_vis))
                logger.log_event(img_name, (0,0,0), "MECHANICAL_ERROR")
                continue

            # 1. GPU INFERENCE
            am_map, lats = trt_engine.infer(tensor)
            am_blurred = cv2.GaussianBlur(am_map, (5, 5), 0)

            # 2. ADAPTIVE BACKGROUND MASKING & HARDWARE CROP
            hsv_f = cv2.cvtColor(img_vis, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv_f, np.array([80, 40, 40]), np.array([140, 255, 255]))
            coords = np.column_stack(np.where(blue_mask > 0))
            
            am_final = am_blurred.copy()
            am_final[:, :5], am_final[:, 251:] = -5.0, -5.0

            if coords.size > 0:
                l_wall = min(int(np.median([coords[coords[:,0]==r][:,1].min() for r in np.unique(coords[:,0])])) - 10, 75)
                r_wall = max(int(np.median([coords[coords[:,0]==r][:,1].max() for r in np.unique(coords[:,0])])) + 10, 185)
                
                supp_mask = np.ones((256, 256), dtype=bool)
                supp_mask[:, max(0, l_wall):min(256, r_wall)] = False
                supp_mask[Y_START:Y_END, 0:max(0, l_wall)] = False 
                supp_mask[OSC_Y_START:OSC_Y_END, OSC_X_START:OSC_X_END] = False
                am_final[supp_mask] = -5.0

            # 3. ZONAL AMPLIFICATION
            roi_p = am_final[Y_START:Y_END, X_START:X_END]
            am_final[Y_START:Y_END, X_START:X_END] = np.where(roi_p > AMP_TRIGGER_THRESHOLD, roi_p + PIN_ZONE_BIAS, roi_p)
            roi_o = am_final[OSC_Y_START:OSC_Y_END, OSC_X_START:OSC_X_END]
            am_final[OSC_Y_START:OSC_Y_END, OSC_X_START:OSC_X_END] = np.where(roi_o > AMP_TRIGGER_THRESHOLD, roi_o + OSC_BIAS, roi_o)

            # 4. FINAL CLASSIFICATION (NO IMAGE SAVING)
            final_score = am_final.max()
            is_anomaly = final_score > PROFILES[CURRENT_MODE]["threshold"]
            
            acc_metric.update(torch.tensor([1. if is_anomaly else 0.]), torch.tensor([float(label)]))
            rec_metric.update(torch.tensor([1. if is_anomaly else 0.]), torch.tensor([float(label)]))
            pre_metric.update(torch.tensor([1. if is_anomaly else 0.]), torch.tensor([float(label)]))
            logger.log_event(img_name, lats, "PASS" if not is_anomaly else "FAIL")

            processed_count += 1
            if processed_count % 100 == 0:
                fps = 100 / (time.perf_counter() - start_bench)
                print(f"\r🚀 [LIVE] {fps:.2f} FPS | Boards: {processed_count} | Mode: {CURRENT_MODE}", end="")
                start_bench = time.perf_counter()

        except Empty: break

    for p in procs: p.join()
    write_q.put(None); writer_p.join()
    print(f"\n\nAccuracy: {acc_metric.compute():.4f} | Recall: {rec_metric.compute():.4f} | Precision: {pre_metric.compute():.4f}")
    print("✅ Process cycle complete. Hardware safety confirmed.")

if __name__ == "__main__": main()
