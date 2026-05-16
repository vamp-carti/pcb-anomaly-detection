#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <filesystem>

namespace fs = std::filesystem;

// --- CONFIGURATION CONSTANTS ---
constexpr int INPUT_H = 256;
constexpr int INPUT_W = 256;
constexpr int INPUT_C = 3;
constexpr size_t INPUT_SIZE = INPUT_H * INPUT_W * INPUT_C * sizeof(float);
constexpr size_t OUTPUT_SIZE = INPUT_H * INPUT_W * sizeof(float);
constexpr int RING_BUFFER_SIZE = 10;

// --- FIXED REGIONAL AND ALIGNMENT TUNING CONSTANTS (ZONAL GUARDS) ---
constexpr int X_START = 30;
constexpr int X_END = 90;
constexpr int Y_START = 95;
constexpr int Y_END = 165;
constexpr float AMP_TRIGGER_THRESHOLD = -0.4500f;
constexpr float PIN_ZONE_BIAS = 0.1f;

constexpr int OSC_X_START = 155;
constexpr int OSC_X_END = 180;
constexpr int OSC_Y_START = 95;
constexpr int OSC_Y_END = 160;
constexpr float OSC_BIAS = 0.05f;

// Pre-inference structural width validation limit (Early Reject Threshold)
constexpr int VALID_MIN_WIDTH_THRESHOLD = 95; 

class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

// --- THREAD-SAFE RING BUFFER FOR ASYNC PRODUCER-CONSUMER FLOW ---
struct FrameData {
    cv::Mat frame;
    std::string filename;
};

class FrameRingBuffer {
private:
    std::queue<FrameData> queue_;
    std::mutex mutex_;
    std::condition_variable cv_not_full_;
    std::condition_variable cv_not_empty_;
    bool finished_ = false;

public:
    void push(FrameData&& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [this]() { return queue_.size() < RING_BUFFER_SIZE; });
        queue_.push(std::move(data));
        cv_not_empty_.notify_one();
    }

    bool pop(FrameData& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [this]() { return !queue_.empty() || finished_; });
        if (queue_.empty() && finished_) return false;
        data = std::move(queue_.front());
        queue_.pop();
        cv_not_full_.notify_one();
        return true;
    }

    void set_finished() {
        std::unique_lock<std::mutex> lock(mutex_);
        finished_ = true;
        cv_not_empty_.notify_all();
    }
};

// --- EXTERNAL SEPARATED CUDA COMPILATION UNIT INTERFACES ---
extern "C" void launch_hwc_to_chw_interface(const uint8_t* src_data, float* dst_device, 
                                            int width, int height, int src_step, 
                                            cudaStream_t stream);

extern "C" void launch_postprocess_mask_amp(const float* src_device, float* dst_device,
                                            int width, int height,
                                            int final_left, int final_right,
                                            int y_start, int y_end,
                                            int x_start, int x_end,
                                            int osc_y_start, int osc_y_end,
                                            int osc_x_start, int osc_x_end,
                                            float amp_trigger_threshold,
                                            float pin_zone_bias, float osc_bias,
                                            cudaStream_t stream);

// --- FILE DISK PRODUCER THREAD WORKER ---
void producer_file_reader(const std::vector<std::string>& file_names, FrameRingBuffer& ring_buffer) {
    for (const auto& file_path : file_names) {
        cv::Mat frame = cv::imread(file_path);
        if (frame.empty()) continue;
        
        FrameData data;
        data.frame = frame;
        data.filename = fs::path(file_path).filename().string();
        
        ring_buffer.push(std::move(data));
    }
    ring_buffer.set_finished();
}

int main() {
    std::string input_path = "/app/temp/images/*.png";
    std::string engine_path = "pcb.engine";
    float threshold = -0.2800f;

    // --- TENSORRT ENGINE INITIALIZATION ---
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) { std::cerr << "Engine file verification failed!" << std::endl; return -1; }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), size);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // --- ALLOCATE PERSISTENT ENGINES & STAGING BUFFERS (EXACTLY ONCE) ---
    void* buffers[2];
    cudaMalloc(&buffers[0], INPUT_SIZE);  // Device Input Tensor Pointer (Planar Float)
    cudaMalloc(&buffers[1], OUTPUT_SIZE); // Device Output Tensor Pointer (Raw Map Float - d_anomaly_map_raw)

    // Allocate Host Pinned (Page-Locked) output memory area for zero-allocation DMA extraction
    float* h_output_pinned = nullptr;
    cudaHostAlloc(&h_output_pinned, OUTPUT_SIZE, cudaHostAllocPortable);
    cv::Mat h_anomaly(INPUT_H, INPUT_W, CV_32FC1, h_output_pinned);

    // --- PRE-ALLOCATED STORAGE ARENA ON VRAM ---
    cv::cuda::GpuMat d_frame; 
    cv::cuda::GpuMat d_warped_320(320, 320, CV_8UC3);
    cv::cuda::GpuMat d_rotated_320(320, 320, CV_8UC3);
    cv::cuda::GpuMat d_final_256; 

    // --- POST-PROCESSING SPECIFIC PERSISTENT VRAM BUFFERS ---
    cv::cuda::GpuMat d_anomaly_map_blurred(INPUT_H, INPUT_W, CV_32FC1);
    cv::cuda::GpuMat d_anomaly_map_final(INPUT_H, INPUT_W, CV_32FC1);

    // Initialize Asynchronous Execution Stream
    cv::cuda::Stream stream;
    cudaStream_t raw_stream = cv::cuda::StreamAccessor::getStream(stream);

    // Pre-allocate and cache the GPU Gaussian Blur execution filter
    cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(CV_32FC1, CV_32FC1, cv::Size(5, 5), 0);

    // Cache structural rotation specifications once
    cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point2f(160.0f, 160.0f), 90.0, 1.0);

    // Gather file lists
    std::vector<std::string> file_names;
    cv::glob(input_path, file_names);
    if(file_names.empty()) { std::cerr << "No verification images found in input target directory." << std::endl; return -1; }

    // Spin up Asynchronous Pipeline Framework
    FrameRingBuffer ring_buffer;
    std::thread reader_thread(producer_file_reader, std::cref(file_names), std::ref(ring_buffer));

    FrameData job;
    // --- MAIN ENGINE PIPELINE EXECUTION LOOP ---
    while (ring_buffer.pop(job)) {
        
        // --- ASYNC CPU PRE-PROCESSING CORNER EXTRACTIONS ---
        cv::Mat hsv, mask;
        cv::cvtColor(job.frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(90, 50, 50), cv::Scalar(135, 255, 255), mask);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) continue;

        auto largest = *std::max_element(contours.begin(), contours.end(), 
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) { 
                return cv::contourArea(a) < cv::contourArea(b); 
            });

        cv::RotatedRect rect = cv::minAreaRect(largest);
        cv::Point2f pts[4];
        rect.points(pts);

        std::vector<cv::Point2f> src_pts(4);
        std::vector<cv::Point2f> box(pts, pts + 4);
        auto sum_c = [](cv::Point2f a, cv::Point2f b) { return (a.x + a.y) < (b.x + b.y); };
        auto diff_c = [](cv::Point2f a, cv::Point2f b) { return (a.y - a.x) < (b.y - b.x); };
        
        src_pts[0] = *std::min_element(box.begin(), box.end(), sum_c); 
        src_pts[1] = *std::min_element(box.begin(), box.end(), diff_c);
        src_pts[2] = *std::max_element(box.begin(), box.end(), sum_c);
        src_pts[3] = *std::max_element(box.begin(), box.end(), diff_c);

        float d01 = cv::norm(src_pts[0] - src_pts[1]);
        float d03 = cv::norm(src_pts[0] - src_pts[3]);
        float board_w, board_h;

        if (d03 > d01) {
            std::rotate(src_pts.begin(), src_pts.begin() + 1, src_pts.end());
            board_w = d03; board_h = d01;
        } else {
            board_w = d01; board_h = d03;
        }

        cv::Mat mom_mask = ~mask; 
        cv::Moments mom = cv::moments(mom_mask, true);
        if (mom.m00 > 0 && (mom.m01 / mom.m00) < (job.frame.rows / 2.0f)) {
            std::rotate(src_pts.begin(), src_pts.begin() + 2, src_pts.end());
        }

        float target_w = 240.0f; 
        float aspect = board_h / board_w; 
        float target_h = target_w * aspect; 
        float ox = (320.0f - target_w) / 2.0f;
        float oy = (320.0f - target_h) / 2.0f;

        std::vector<cv::Point2f> dst_pts(4);
        dst_pts[0] = {ox,            oy};
        dst_pts[1] = {ox + target_w, oy};
        dst_pts[2] = {ox + target_w, oy + target_h};
        dst_pts[3] = {ox,            oy + target_h};

        cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);

        // --- ASYNC OVERLAPPED GPU PIPELINE ---
        d_frame.upload(job.frame, stream);

        // STAGE 1: Perspective Warp 
        cv::cuda::warpPerspective(d_frame, d_warped_320, M, cv::Size(320, 320), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0), stream);

        // STAGE 2: Affine Clockwise Rotation Matrix Processing
        cv::cuda::warpAffine(d_warped_320, d_rotated_320, rot_mat, cv::Size(320, 320), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0), stream);

        // STAGE 3: Structural Crop via Shallow Sub-Matrix Selection (Zero Allocation overhead)
        cv::Rect roi_box(32, 32, INPUT_W, INPUT_H);
        d_final_256 = d_rotated_320(roi_box);

        // ========================================================================
        // STAGE 3.5: PRE-INFERENCE 1D EDGE EXTRACTION & EARLY REJECT (ME3)
        // ========================================================================
        // Re-extract mask from aligned BGR view to find exact board edge bounds
        cv::cuda::GpuMat d_hsv, d_mask;
        cv::cuda::cvtColor(d_final_256, d_hsv, cv::COLOR_BGR2HSV, 0, stream);
        
        // Isolate the physical board body color profile layout
        cv::cuda::inRange(d_hsv, cv::Scalar(90, 50, 50), cv::Scalar(135, 255, 255), d_mask, stream);
        
        // 1D Column-wise projection reduction mapping
        cv::cuda::GpuMat d_proj_cols;
        cv::cuda::reduce(d_mask, d_proj_cols, 0, cv::REDUCE_SUM, CV_32S, stream);
        
        // Download small 1D projection array back to host synchronously for layout sanity check
        cv::Mat h_proj_cols;
        d_proj_cols.download(h_proj_cols, stream);
        stream.waitForCompletion(); // Sync stream to process edge analysis integers

        int detected_l_wall = -1;
        int detected_r_wall = -1;
        const int* proj_ptr = h_proj_cols.ptr<int>();

        // Scan columns left-to-right for edge entries
        for (int c = 0; c < INPUT_W; ++c) {
            if (proj_ptr[c] > 0) {
                if (detected_l_wall == -1) detected_l_wall = c;
                detected_r_wall = c;
            }
        }

        // Run Early Reject Width Guard Checks
        int current_board_width = detected_r_wall - detected_l_wall;
        if (detected_l_wall == -1 || current_board_width < VALID_MIN_WIDTH_THRESHOLD) {
            std::cout << "⚠️ [" << job.filename << "] EARLY REJECT: (ME2)" << std::endl;
            // Asynchronous logging/offloading queue would handle disk save here
            continue; 
        }

        // --- CPU PARAMETER RESOLUTION LAYER ---
        // Pass dynamic padded coordinate boundaries straight to GPU kernel registers
        int final_left_wall  = detected_l_wall - 5;
        int final_right_wall = detected_r_wall + 5;

        // --- INFERENCE PIPELINE ---
        // STAGE 4: Interleaved-to-Planar Conversion and Normalization via Custom Unified Kernel
        launch_hwc_to_chw_interface(d_final_256.data, static_cast<float*>(buffers[0]), 
                                    d_final_256.cols, d_final_256.rows, d_final_256.step, 
                                    raw_stream);

        // STAGE 5: Execute Engine Model Asynchronous Inference
        context->enqueueV2(buffers, raw_stream, nullptr);

        // ========================================================================
        // DEVICE-CENTRIC ANOMALY POST-PROCESSING ENGINE
        // ========================================================================
        // STAGE 1: GPU Spatial Smoothing
        // cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(CV_32FC1, CV_32FC1, cv::Size(5, 5), 0);
        gaussian_filter->apply(cv::cuda::GpuMat(INPUT_H, INPUT_W, CV_32FC1, buffers[1]), d_anomaly_map_blurred, stream);
        // STAGE 2 & 3: Unified Structural Mask Pass + In-Line Regional Gain Overrides
        launch_postprocess_mask_amp(
            reinterpret_cast<float*>(d_anomaly_map_blurred.data),
            reinterpret_cast<float*>(d_anomaly_map_final.data),
            INPUT_W, INPUT_H,
            final_left_wall, final_right_wall,
            Y_START, Y_END, X_START, X_END,
            OSC_Y_START, OSC_Y_END, OSC_X_START, OSC_X_END,
            AMP_TRIGGER_THRESHOLD, PIN_ZONE_BIAS, OSC_BIAS,
            raw_stream
        );

        // STAGE 4: Parallel Tree Min/Max Value Reduction Call on VRAM Layout
        cudaMemcpyAsync(h_anomaly.data, d_anomaly_map_final.data, OUTPUT_SIZE, cudaMemcpyDeviceToHost, raw_stream);

        // Synchronize stream execution context to execute evaluation logic checks
        stream.waitForCompletion();

        // Global Max Extraction
        double max_val;
        cv::minMaxLoc(h_anomaly, nullptr, &max_val);
        std::cout << "[" << job.filename << "] Score: " << max_val << " | Status: " << (max_val > threshold ? "FAIL" : "PASS") << std::endl;
    }

    // Join ingestion tracking thread context
    if (reader_thread.joinable()) reader_thread.join();

    // Clean up allocated assets
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFreeHost(h_output_pinned);
    delete context; 
    delete engine; 
    delete runtime;
    return 0;
}
