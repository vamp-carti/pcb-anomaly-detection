#include <cuda_runtime.h>
#include <cstdint>

// The exact same kernel logic
__global__ void hwc_to_chw_normalize_kernel(const uint8_t* __restrict__ src, float* __restrict__ dst, 
                                            int width, int height, int src_step,
                                            float m0, float m1, float m2,
                                            float s0, float s1, float s2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int spatial_idx = y * width + x;
        // Use the explicit hardware step (stride) from OpenCV rather than a hardcoded width*3
        int src_idx = y * src_step + x * 3;
        int plane_area = width * height;

        float b = static_cast<float>(src[src_idx])     / 255.0f;
        float g = static_cast<float>(src[src_idx + 1]) / 255.0f;
        float r = static_cast<float>(src[src_idx + 2]) / 255.0f;

        dst[spatial_idx]                  = (r - m0) / s0; 
        dst[spatial_idx + plane_area]      = (g - m1) / s1; 
        dst[spatial_idx + plane_area * 2]  = (b - m2) / s2; 
    }
}

// C-compatible wrapper function that debug.cpp will call
extern "C" void launch_hwc_to_chw_interface(const uint8_t* src_data, float* dst_device, 
                                            int width, int height, int src_step, 
                                            cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    hwc_to_chw_normalize_kernel<<<grid, block, 0, stream>>>(
        src_data, dst_device, width, height, src_step,
        0.485f, 0.456f, 0.406f,
        0.229f, 0.224f, 0.225f
    );
}
