import torch
import torchvision
import torchmetrics
import os
from pathlib import Path
from anomalib.models.image.fastflow.torch_model import FastflowModel

# 1. SAFETY MEASURES (Required for PyTorch 2.6+ Checkpoints)
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

def export_pcb_to_onnx():
    # --- CONFIGURATION ---
    CKPT_PATH = "/app/v2/results/R1_flow_8/pcb_fastflow_v2/version_0/checkpoints/pcb_v2_best.ckpt"
    OUTPUT_PATH = "/app/v2/pcb_fastflow_production.onnx"
    
    BACKBONE = "resnet18"
    FLOW_STEPS = 8
    INPUT_SIZE = (256, 256)

    print(f"🚀 Initializing FastFlow ({BACKBONE}) for export...")

    # 1. Instantiate the raw PyTorch model
    model = FastflowModel(
        backbone=BACKBONE,
        flow_steps=FLOW_STEPS,
        input_size=INPUT_SIZE
    )

    # 2. Load Checkpoint with Security Bypass
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Error: Checkpoint not found at {CKPT_PATH}")
        return

    # Explicitly setting weights_only=False to handle the custom torchvision/torchmetrics metadata
    checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    
    # Surgical State-Dict Mapping
    new_state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model."):
            new_state_dict[k.replace("model.", "")] = v
    
    try:
        msg = model.load_state_dict(new_state_dict, strict=True)
        print(f"✅ Weights loaded: {msg}")
    except RuntimeError as e:
        print(f"⚠️ Strict load failed. Attempting non-strict. Error: {e}")
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    # 3. Dummy Input (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, *INPUT_SIZE)

    # 4. Nuclear Legacy Export
    from torch.onnx.utils import _export
    
    print("🛠️ Executing Trace-based Export...")
    
    try:
        _export(
            model,
            (dummy_input,),
            OUTPUT_PATH,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )
        
        if os.path.exists(OUTPUT_PATH):
            file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
            print(f"🎊 EXPORT SUCCESS: {OUTPUT_PATH}")
            print(f"📊 Model Size: {file_size:.2f} MB")
        else:
            print("❌ Export failed: File not written.")
            
    except Exception as e:
        print(f"💥 Export failed: {str(e)}")

if __name__ == "__main__":
    export_pcb_to_onnx()
