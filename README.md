# Industrial PCB Anomaly Detection Engine

A high-throughput, production-hardened computer vision pipeline for real-time PCB defect localization using FastFlow, TensorRT, CUDA, and a native OPC UA industrial network stack.

The system features a pure C++ inference engine that completely decouples from Python dependencies, utilizing asynchronous GPU execution streams to process inspection triggers under deterministic latencies.

---

# 📊 Key Performance Metrics

| Metric | Result |
|---|---|
| **Accuracy** | 97.59% |
| **Precision** | 83.33% |
| **Recall** | 95.00% |
| **Throughput** | 111.15 FPS |
| **VRAM Footprint** | ~134 MB |
| **Inference Backend** | TensorRT FP16 (Ampere Compute SM 86 Optimized) |
| **Industrial Protocol** | Native OPC UA (open62541 Amalgamated Server) |
| **Host Target Language** | Pure ISO C++17 & Native CUDA |

---

# 🏗️ Architecture Overview

The production engine executes as a synchronized, state-driven loop tied directly to the industrial network layer.

```text
  [PLC / Client Network]
           │
           ▼ (Trigger Node: True)
┌──────────────────────────────────────┐
│  Native OPC UA Server Layer (4840)   │
└──────────────────┬───────────────────┘
                   │
                   ▼ (Asynchronous Grab)
┌──────────────────────────────────────┐
│  Host File Capture: /app/capture/    │
└──────────────────┬───────────────────┘
                   │
                   ▼ (Stream Overlap Allocator)
┌──────────────────────────────────────┐
│  CUDA Stream Dual-Buffer (Ping-Pong) │
├──────────────────┴───────────────────┤
│ 🔹 Stream [0]: Async Preprocess/H2D  │
│ 🔸 Stream [1]: TensorRT FP16 Kernel  │
└──────────────────┬───────────────────┘
                   │
                   ▼ (Edge Evaluation)
┌──────────────────────────────────────┐
│   Dynamic Threshold Profile Policy   │
├──────────────────────────────────────┤
│ 🛑 SAFETY   (Value: 10 -> -0.3200f)  │
│ ⚖️ BALANCED (Value: 20 -> -0.2500f)  │
│ ⚡ UPTIME   (Value: 30 -> -0.1500f)  │
└──────────────────┬───────────────────┘
                   │
                   ▼ (Atomic Registers Flush)
┌──────────────────────────────────────┐
│   Rejection Gate + Async Telemetry   │
└──────────────────────────────────────┘

1. Edge-Inbound Networking
- **Reactive Idle Processing:** The engine spins down to 0% active CPU usage, sleeping on an atomic network latch waiting for an industrial client event.
- **Native OPC UA Variable Mapping:** Directly exposes system registers (`InspectionTrigger`, `InspectionResult`, `AnomalyScore`, `Heartbeat`, `IndustrialProfile`, `SystemFaultCode`) over standard loopback port `4840`.

2. High-Performance GPU Core
- **FastFlow Unsupervised Anomaly Detection:** Leverages an FP16 ResNet-18 backbone inside an optimized TensorRT execution context.
- **Double-Buffered CUDA Streams:** Implements strict asynchronous memory copies overlapped with concurrent GPU kernel processing to hide host-to-device ($H2D$) and device-to-host ($D2H$) data transfer overheads.
- **Asymmetric Pipeline Drain:** A dedicated barrier clean-up loop flushes execution pipelines sequentially during graceful engine shutdowns to prevent edge memory leaks.

3. Dynamic Operational Profiles
- **On-The-Fly Policy Switching:** Allows real-time modification of anomaly sensitivity tolerances without stopping the container or incurring production line downtime.
- **Profile Enforcements:** Supports **SAFETY** (high sensitivity for critical components), **BALANCED** (factory defaults), and **UPTIME** (lenient tolerances to prevent line bottlenecks).

4. Telemetry Logging
- **Thread-Isolated Performance Audits:** High-speed asynchronous worker batches and structures metrics directly into system paths (`/var/log/pcb_engine/telemetry_production.json`). Tracks inference speeds, VRAM capacities ($MB$), transfer latencies, and physical trigger stamps.

---

# 🚀 Major Engineering Optimizations

## 💻 Python to Native C++ Execution Path
The original PyTorch and Python development layers were systematically extracted and rewritten using the native TensorRT C++ Runtime API. By stripping away Global Interpreter Lock (GIL) runtime constraints and heavy Python packages, edge throughput leaped from **53.20 FPS to 111.15 FPS**, while shrinking the system's operational memory baseline down to a hyper-lean **~134 MB of VRAM**.

## 🔄 Dual Stream Ping-Pong Scheduling
Memory management maps out two independent runtime allocations tied to specific `cudaStream_t` slots. While Stream `N` is processing data on an active TensorRT inference pass, Stream `N+1` is simultaneously completing its host-to-device image upload over pinned memory boundaries, allowing continuous hardware utilization.

## 🛡️ Resilient Industrial Watchdog
An independent, high-priority safety thread acts as a system sentinel. It tracks stuck hardware handshakes, processing timeouts, network connection drops, and memory leaks, writing diagnostic registers to prevent equipment pile-ups or physical conveyor line faults.

---

# 🌐 Operational Integration (OT Interoperability)

The system enforces standardized industrial status codes across the network namespace:

```cpp
STATUS_PASS        = 0  // Board within nominal validation tolerance
STATUS_FAIL        = 1  // Defect anomaly localized by engine
STATUS_MECH_ERROR  = 2  // Alignment/geometry failure caught by gateway


Operational deployment profiles:

```cpp
PROFILE_SAFETY     = 10 // Threshold: -0.3200f
PROFILE_BALANCED   = 20 // Threshold: -0.2500f
PROFILE_UPTIME     = 30 // Threshold: -0.1500f

```markdown
# 🐳 Deployment & Multi-Stage Orchestration

The system uses a hardened, multi-stage Docker production pattern to ensure reproducible, zero-dependency deployments on edge hardware.

### Production Environment Topology
- **Isolated System Paths:** Mounts runtime assets directly from host OS standard directories to completely safeguard internal storage volumes:
  - Input Frames Folder: `/opt/factory/capture/live_pcb.png`
  - Output Telemetry Logs: `/var/log/pcb_engine/telemetry_production.json`
  - Compiled TensorRT Weights: `./models/pcb_fastflow_fp16.engine`

### Running the Production Engine Natively
Ensure your model weights are dropped inside your `./models/` workspace directory, then execute the orchestration stack:

```bash
# Clean up stale container interfaces
docker compose down
docker container prune -f

# Build and run the multi-stage C++ engine
docker compose build cv-production
docker compose up cv-production

```markdown
# 🛠️ Tech Stack

- **Languages:** ISO C++17, Native CUDA (NVIDIA NVCC)
- **Frameworks:** TensorRT Runtime C++ API, open62541 (OPC UA Stack), OpenCV CUDA
- **Concurrency:** C++ Multithreading (`std::thread`, `std::atomic`), CUDA Asynchronous Streams
- **Infrastructure:** Multi-Stage Docker, Docker Compose Volume Mapping
