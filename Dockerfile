# syntax=docker/dockerfile:1

# ==========================================
# STAGE 1: RUNTIME BASE
# ==========================================
FROM nvcr.io/nvidia/tensorrt:24.03-py3 AS base
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install the minimal system runtimes AND the core shared object image codec formats
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg8 \
    libpng16-16 \
    libtiff5 \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# STAGE 2: BUILDER (Compiles Everything)
# ==========================================
FROM base AS builder

# Install build tools and standard packaging tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    unzip \
    pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev \
    libatlas-base-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

# Build OpenCV with CUDA from Source
WORKDIR /opt
RUN git clone --depth 1 https://github.com/opencv/opencv.git && \
    git clone --depth 1 https://github.com/opencv/opencv_contrib.git

WORKDIR /opt/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D CUDA_ARCH_BIN=8.6 \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_python3=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D WITH_NVCUVID=OFF \
    -D WITH_NVCUVENC=OFF \
    -D BUILD_opencv_cudacodec=OFF \
    .. && \
    make -j$(nproc) && \
    make install

# Build open62541 (OPC UA C/C++ Stack) from Source
WORKDIR /opt
RUN git clone --depth 1 -b v1.3.11 https://github.com/open62541/open62541.git

WORKDIR /opt/open62541/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D UA_ENABLE_AMALGAMATION=ON \
          -D UA_ENABLE_MULTITHREADING=ON \
          .. && \
    make -j$(nproc) && \
    make install

# Copy C++ source files to build the production binary
WORKDIR /app
COPY src/ ./src/

# Navigate to your isolated production workspace and compile
WORKDIR /app/src/production/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE .. && \
    make -j$(nproc)

# ==========================================
# STAGE 3: RUNTIME PRODUCTION IMAGE
# ==========================================
FROM base AS production

# Copy compiled shared libraries directly from the builder stage
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/include /usr/local/include
RUN ldconfig

WORKDIR /app

# Copy ONLY the compiled native production binary executable
COPY --from=builder /app/src/production/build/pcb_engine ./pcb_engine

# Set environment variable configuration defaults (Overridden by docker-compose)
ENV MODEL_ENGINE_PATH="/app/models/pcb_fastflow_fp16.engine"
ENV TELEMETRY_LOG_PATH="/app/logs/telemetry_production.json"
ENV DEPLOYMENT_PROFILE="20"
ENV OPC_UA_PORT="4840"
ENV LIVE_CAPTURE_PATH="/app/capture/live_pcb.png"
ENV ERROR_DUMP_DIR="/app/logs/mechanical_errors/"

# Launch the compiled binary directly
ENTRYPOINT ["./pcb_engine"]
