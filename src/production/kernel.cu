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
        int src_idx = y * src_step + x * 3;
        int plane_area = width * height;

        // Extract raw pixel intensity bytes directly
        float b_val = static_cast<float>(src[src_idx]);
        float g_val = static_cast<float>(src[src_idx + 1]);
        float r_val = static_cast<float>(src[src_idx + 2]);

        // Normalize exactly matching Albumentations scaling logic to kill score drift
        dst[spatial_idx]                  = (r_val - (m0 * 255.0f)) / (s0 * 255.0f); // Red
        dst[spatial_idx + plane_area]      = (g_val - (m1 * 255.0f)) / (s1 * 255.0f); // Green
        dst[spatial_idx + plane_area * 2]  = (b_val - (m2 * 255.0f)) / (s2 * 255.0f); // Blue
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
// --- UNIFIED MASKING AND ZONAL AMPLIFICATION KERNEL ---
__global__ void postprocess_mask_amp_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                            int width, int height,
                                            int final_left, int final_right,
                                            int y_start, int y_end,
                                            int x_start, int x_end,
                                            int osc_y_start, int osc_y_end,
                                            int osc_x_start, int osc_x_end,
                                            float amp_trigger_threshold,
                                            float pin_zone_bias, float osc_bias) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float score = src[idx];

        // 1. Resolve Dynamic Left Wall vs Unconditional 20px Pin Zone Mask Limit
        int target_left = (y >= y_start && y <= y_end) ? 20 : final_left;

        // 2. Unconditional Spatial Masking Pass
        if (x < target_left || x > final_right) {
            score = -5.0f;
        } else {
            // 3. In-Line Volumetric Zonal Amplification
            if (score > amp_trigger_threshold) {
                // Pin Zone Region Condition
                if (y >= y_start && y <= y_end && x >= x_start && x <= x_end) {
                    score += pin_zone_bias;
                }
                // Oscillator Zone Region Condition
                else if (y >= osc_y_start && y <= osc_y_end && x >= osc_x_start && x <= osc_x_end) {
                    score += osc_bias;
                }
            }
        }

        dst[idx] = score;
    }
}

extern "C" void launch_postprocess_mask_amp(const float* src_device, float* dst_device,
                                            int width, int height,
                                            int final_left, int final_right,
                                            int y_start, int y_end,
                                            int x_start, int x_end,
                                            int osc_y_start, int osc_y_end,
                                            int osc_x_start, int osc_x_end,
                                            float amp_trigger_threshold,
                                            float pin_zone_bias, float osc_bias,
                                            cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    postprocess_mask_amp_kernel<<<grid, block, 0, stream>>>(
        src_device, dst_device, width, height,
        final_left, final_right,
        y_start, y_end, x_start, x_end,
        osc_y_start, osc_y_end, osc_x_start, osc_x_end,
        amp_trigger_threshold, pin_zone_bias, osc_bias
    );
}
