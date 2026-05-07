# Industrial PCB Anomaly Detection Engine

A high-throughput, production-grade computer vision pipeline designed for real-time defect localization on PCB substrates.

## Key Performance Metrics
| Metric | Achievement | Industrial Significance |
| :--- | :--- | :--- |
| **Peak Model Throughput** | 749.87 FPS | High-speed multi-camera synchronization |
| **System Throughput** | 53.20 FPS | Real-time rejection for high-speed lines |
| **Mean Model Latency** | 1.49 ms | Near-instantaneous density estimation |
| **VRAM Footprint** | ~0.43 GB | Edge-ready (NVIDIA Jetson / RTX 3050 Ti) |

## Project Architecture
The repository is structured into four distinct engineering phases:
- **Phase 1: Preprocessing:** Coordinate-invariant alignment using moment-based orientation correction.
- **Phase 2: AI Core:** FastFlow unsupervised learning with a ResNet-18 feature extractor.
- **Phase 3: Validation:** Industrial hardening with 3-tier operational profiles (Safety/Balanced/Uptime).
- **Phase 4: Inference:** TensorRT FP16 quantization with CUDA stream double-buffering.

## Industrial Handshake (OT Integration)
The system outputs standardized status codes for IT/OT synchronization:
- \`STATUS_PASS (0)\`: No defect detected.
- \`STATUS_FAIL (1)\`: Anomaly detected; trigger rejection.
- \`STATUS_MECH_ERROR (2)\`: Mechanical/Alignment failure.

## Dataset
The model was trained and validated on the **[VisA Dataset; PCB1](https://huggingface.co/datasets/BrachioLab/visa)**.
*Note: A custom symmetric preprocessing wrapper was implemented to resolve a 91% F1-score bottleneck caused by spatial label-to-image misalignment.*

##  Roadmap: C++ Migration
Currently in development:
**High-Performance C++ Inference Engine**.
- Transitioning from Python/PyTorch to **LibTorch** and **TensorRT C++ API**.
- Implementing zero-copy memory mapping for sub-millisecond H2D transfers.
- Aiming for total system throughput parity with raw model throughput (>500 FPS).
**OPC-UA Handshake**

