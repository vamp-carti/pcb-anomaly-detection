# syntax=docker/dockerfile:1

# --- BASE STAGE ---
# Using the official TensorRT image which has CUDA/TRT pre-configured
FROM nvcr.io/nvidia/tensorrt:24.03-py3 AS base
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install essential runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- DEVELOPMENT STAGE ---
# Used for coding, debugging, and Phase 1 data acquisition
FROM base AS development
RUN apt-get update && apt-get install -y git build-essential
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# No 'COPY .' here because docker-compose.yml handles the live volume mapping
CMD ["/bin/bash"]

# --- PRODUCTION STAGE ---
# Used for deployment on the edge device/conveyor belt
FROM base AS production
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
# Copy the actual project files into the image so it's self-contained
COPY . . 
# Optimize: Remove the venv folder if it exists to keep image small
RUN rm -rf venv
# Start the main inference application
CMD ["python3", "src/inference/engine.py"]
