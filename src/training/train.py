import os
from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.engine import Engine
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch

# 1. DATA CONFIGURATION
datamodule = Folder(
    name="pcb_fastflow",
    root="/app/v2/data/preprocessed_data",
    normal_dir="train/good",
    abnormal_dir="test/anomaly",
    normal_test_dir="test/good",
    mask_dir="ground_truth/anomaly", 
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=4,
    image_size=(256, 256)
)

# 2. MODEL CONFIGURATION
# Removed input_size to fix TypeError; model inherits from datamodule
model = Fastflow(
    backbone="resnet18", 
    flow_steps=16
)

# 3. ENGINE & CALLBACKS
engine = Engine(
    max_epochs=30,
    devices=1,
    accelerator="gpu",
    default_root_dir="/app/v2/results",
    logger=CSVLogger(save_dir="/app/v2/results", name="pcb_fastflow_v2"),
    # Define metrics for threshold optimization
    pixel_metrics=["AUROC", "F1Score"],
    image_metrics=["AUROC", "F1Score"],
    callbacks=[
        ModelCheckpoint(
            monitor="pixel_AUROC",
            mode="max",
            filename="pcb_v2_best",
            save_top_k=1
        )
    ]
)

# 4. EXECUTE & EXTRACT THRESHOLD
if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("--- Starting Training on Preprocessed Data ---")
    engine.fit(model=model, datamodule=datamodule)
    
    print("--- Starting Testing/Evaluation ---")
    # engine.test triggers the threshold optimization logic
    engine.test(model=model, datamodule=datamodule)
    
    # In Anomalib 1.1.0, thresholds are stored in the model's metrics
    print("\n" + "="*40)
    print("FINAL CALIBRATED THRESHOLDS")
    try:
        p_thresh = model.pixel_threshold.value.item()
        i_thresh = model.image_threshold.value.item()
        print(f"Optimal Pixel Threshold: {p_thresh:.4f}")
        print(f"Optimal Image Threshold: {i_thresh:.4f}")
    except AttributeError:
        # Fallback for specific v1.1.0 sub-builds
        print(f"Pixel Threshold: {model.trainer.model.pixel_threshold.value.item():.4f}")
        print(f"Image Threshold: {model.trainer.model.image_threshold.value.item():.4f}")
    print("="*40)
    print("Use the Pixel Threshold above for your local Zair inference engine.")
