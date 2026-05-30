# Industrial PCB Anomaly Detection Engine

A high-throughput, production-grade computer vision pipeline for real-time PCB defect localization using FastFlow, TensorRT, CUDA, and native C++.

The system was fully migrated from Python to a native C++ inference stack, achieving identical validation metrics while doubling throughput and reducing VRAM usage.

---

# Key Performance Metrics

| Metric | Result |
|---|---|
| Accuracy | 97.59% |
| Precision | 83.33% |
| Recall | 95.00% |
| Throughput | 111.15 FPS |
| VRAM Footprint | ~134 MB |
| Backend | TensorRT FP16 |
| Language | Native C++ |

---

# Architecture Overview

The pipeline is divided into four execution stages:

## 1. Preprocessing
- Perspective normalization using contour alignment
- Coordinate-invariant orientation correction
- Morphological PCB segmentation
- Symmetry-aware preprocessing wrapper for dataset correction

## 2. AI Core
- FastFlow unsupervised anomaly detection
- ResNet-18 feature extractor
- TensorRT FP16 inference engine
- GPU-native anomaly map generation

## 3. Validation Layer
- Production-calibrated threshold profiles:
  - SAFETY
  - BALANCED
  - UPTIME
- Mechanical rejection gating
- Threshold tuning for industrial deployment

## 4. High-Performance C++ Inference Engine
- Fully native C++/CUDA/TensorRT pipeline
- CUDA stream double-buffering
- Asynchronous GPU execution
- Multi-threaded producer-consumer architecture
- Pinned host memory allocation
- CUDA-accelerated preprocessing and postprocessing

---

# Major Engineering Optimizations

## Python → Native C++ Migration
The original PyTorch pipeline was fully migrated into:
- TensorRT C++ Runtime API
- CUDA kernels
- OpenCV CUDA
- Multi-threaded C++ execution

while maintaining identical validation metrics.

## Concurrent GPU Pipeline
Implemented using:
- Double-buffered CUDA streams
- Overlapped execution stages
- Asynchronous H2D/D2H transfers
- Ping-pong memory scheduling

## Industrial Early-Reject Gateway
High-speed CPU rejection gates detect:
- Mechanical alignment failures
- Invalid board geometry
- Positioning errors

Rejected boards are automatically logged into telemetry records for audit review.

## Telemetry Logging
Structured JSON telemetry logs include:
- Inference latency
- VRAM usage
- Transfer timings
- Operational profile state
- Mechanical error signatures

---

# Industrial Handshake (OT Integration)

The engine exposes standardized operational states:

```cpp
STATUS_PASS        = 0
STATUS_FAIL        = 1
STATUS_MECH_ERROR  = 2
```

Operational deployment profiles:

```cpp
SAFETY    = 10
BALANCED  = 20
UPTIME    = 30
```

Designed for future:
- OPC-UA integration
- PLC synchronization
- Smart conveyor rejection systems
- MES/SCADA interoperability

---

# Throughput Improvement

Native C++ migration improved end-to-end throughput from:

```text
53.20 FPS  →  111.15 FPS
```

while preserving validation parity with the original Python implementation.

---

# Profiling & Verification

The asynchronous inference pipeline was profiled and validated using NVIDIA Nsight Systems.

Verified behaviors:
- CUDA stream overlap
- Concurrent H2D/D2H transfers
- GPU kernel concurrency
- Double-buffered execution scheduling
- Pipeline stage overlap efficiency

Nsight traces are included in the repository for performance verification and latency analysis.
```
---

# Dataset

The model was trained and validated on the VisA Dataset — PCB1 subset.

A custom preprocessing wrapper was implemented to resolve spatial label-to-image misalignment that previously bottlenecked model performance.

---

# Tech Stack

- C++
- CUDA
- TensorRT
- OpenCV CUDA
- LibTorch
- Multithreading (std::thread)
- NVIDIA GPU Runtime

---

# Repository Structure

```text
src/
├── preprocessing/
├── training/
├── validation/
├── inference/
└── cpp_inference/

configs/
assets/
results/
data/
```

---

# Future Roadmap

- OPC-UA industrial handshake
- Real-time telemetry dashboard

---

# Objective

This project focuses on bridging:
- AI research pipelines
- real-time GPU systems engineering
- and industrial automation workflows

with an emphasis on:
- deterministic latency
- edge deployment
- and production-grade reliability.

---

# Development Environment

The project is currently developed inside a Docker-based CUDA environment while the industrial deployment stack is still under active development.

Current workflow:
- Model training and validation in Python
- Native C++ TensorRT inference pipeline
- CUDA/OpenCV GPU acceleration
- Dockerized dual-environment setup for reproducible experimentation

Production deployment and OPC-UA integration are planned in the next phase.

## Python Dependencies

```bash
pip install -r requirements.txt
```

## Current Stack

### AI / Deep Learning
- PyTorch
- Torchvision
- Anomalib
- FastFlow
- ONNX

### GPU / Optimization
- CUDA
- TensorRT
- CuPy
- OpenCV CUDA

### Data / CV
- NumPy
- Albumentations
- Kornia
- OpenCV
- Scikit-learn

### Systems Engineering
- Native C++
- Multithreading
- CUDA Streams
- TensorRT Runtime API
```
