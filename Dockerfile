# syntax=docker/dockerfile:1

# --- BASE STAGE ---
FROM nvcr.io/nvidia/tensorrt:24.03-py3 AS base
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install essential runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- DEVELOPMENT STAGE ---
FROM base AS development

# 1. Install Build Tools and OpenCV Dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    unzip \
    pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libatlas-base-dev gfortran \
    libnvinfer-dev \
    libnvonnxparsers-dev \
    libnvparsers-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Build OpenCV with CUDA from Source
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
    -D BUILD_opencv_python3=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    # Professional Fix for CUDA_CUDA_LIBRARY NOTFOUND:
    # Disabling Video Codec SDK dependency which is not needed for image-based PCB anomaly detection
    -D WITH_NVCUVID=OFF \
    -D WITH_NVCUVENC=OFF \
    -D BUILD_opencv_cudacodec=OFF \
    .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# 3. Build open62541 (OPC UA C/C++ Stack) from Source
WORKDIR /opt
RUN git clone --depth 1 -b v1.3.11 https://github.com/open62541/open62541.git

WORKDIR /opt/open62541/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D UA_ENABLE_AMALGAMATION=ON \
          -D UA_ENABLE_MULTITHREADING=ON \
          .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
CMD ["/bin/bash"]

# --- PRODUCTION STAGE ---
FROM base AS production
# Copy compiled OpenCV libraries from development stage to keep production slim
COPY --from=development /usr/local /usr/local
RUN ldconfig
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY . . 
RUN rm -rf venv
CMD ["python3", "src/inference/engine.py"]
