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
#include <chrono>
#include <sstream>
#include <iomanip>

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
constexpr float PIN_ZONE_BIAS = 0.09f; 

constexpr int OSC_X_START = 155; 
constexpr int OSC_X_END = 180; 
constexpr int OSC_Y_START = 95; 
constexpr int OSC_Y_END = 160; 
constexpr float OSC_BIAS = 0.05f; 

constexpr int VALID_MIN_WIDTH_THRESHOLD = 95;  

// Industrial Handshake / OPC Status Profiles
enum class IndustrialProfile {
    SAFETY = 10,
    BALANCED = 20,
    UPTIME = 30
};

class TRTLogger : public nvinfer1::ILogger { 
    void log(Severity severity, const char* msg) noexcept override { 
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl; 
    } 
} gLogger; 

// --- SEPARATED FILE INGESTION STRUCT WITH GROUND TRUTH LABELS --- 
struct FrameData { 
    cv::Mat frame; 
    std::string filename; 
    int label; // 0 = Good, 1 = Anomaly
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

// --- SEGREGATED INPUT MULTI-CLASS PRODUCER WORKER --- 
// ─── CHANGER 1: MODIFIED MULTI-THREADED PRODUCER WORKER ───
void producer_file_reader(std::vector<std::pair<std::string, int>>::const_iterator start, 
                          std::vector<std::pair<std::string, int>>::const_iterator end, 
                          FrameRingBuffer& ring_buffer) { 
    for (auto it = start; it != end; ++it) { 
        cv::Mat frame = cv::imread(it->first); 
        if (frame.empty()) continue; 
         
        FrameData data; 
        data.frame = frame; 
        data.filename = fs::path(it->first).filename().string(); 
        data.label = it->second; // Fixed: Label is safely packed
         
        ring_buffer.push(std::move(data)); 
    } 
}

// Helper function to return system ISO timestamp strings
std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

int main() { 
    // Target Segregated Image Paths
    std::string good_dir = "/app/data/images/good/*.png"; 
    std::string anomaly_dir = "/app/data/images/anomaly/*.png"; 
    std::string engine_path = "/app/results/pcb_fastflow_fp16.engine"; 
    std::string telemetry_log_path = "/app/temp/telemetry_log.json";

    // Dynamic Industrial Profile Configuration Layer
    IndustrialProfile current_profile = IndustrialProfile::BALANCED;
    float threshold = -0.2600f; // Default (BALANCED)
    std::string profile_str = "BALANCED";

    if (current_profile == IndustrialProfile::SAFETY) {
        threshold = -0.3200f;
        profile_str = "SAFETY";
    } else if (current_profile == IndustrialProfile::UPTIME) {
        threshold = -0.1500f;
        profile_str = "UPTIME";
    }

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
    cudaMalloc(&buffers[0], INPUT_SIZE);  
    cudaMalloc(&buffers[1], OUTPUT_SIZE); 

    float* h_output_pinned = nullptr; 
    cudaHostAlloc(&h_output_pinned, OUTPUT_SIZE, cudaHostAllocPortable); 
    cv::Mat h_anomaly(INPUT_H, INPUT_W, CV_32FC1, h_output_pinned); 

    // --- PRE-ALLOCATED STORAGE ARENA ON VRAM (ZERO RUNTIME CHURN) --- 
    cv::cuda::GpuMat d_frame;  
    cv::cuda::GpuMat d_frame_rgba;
    cv::cuda::GpuMat d_warped_320(320, 320, CV_8UC4);   
    cv::cuda::GpuMat d_rotated_320(320, 320, CV_8UC4);  
    cv::cuda::GpuMat d_final_padded;
    cv::cuda::GpuMat d_final_256(INPUT_H, INPUT_W, CV_8UC3); 

    std::vector<cv::cuda::GpuMat> bgra_channels; 
    std::vector<cv::cuda::GpuMat> bgr_channels(3);

    cv::cuda::GpuMat d_anomaly_map_blurred(INPUT_H, INPUT_W, CV_32FC1); 
    cv::cuda::GpuMat d_anomaly_map_final(INPUT_H, INPUT_W, CV_32FC1); 

    cv::cuda::GpuMat d_hsv, d_mask, d_proj_cols;
    cv::Mat h_proj_cols;

    cv::cuda::Stream stream; 
    cudaStream_t raw_stream = cv::cuda::StreamAccessor::getStream(stream); 

    cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(CV_32FC1, CV_32FC1, cv::Size(5, 5), 0); 

    // ======================================================================== 
    // MANDATORY GPU WARMUP ENGINE TENSORS
    // ======================================================================== 
    std::cout << "🔥 Calibrating CUDA Streams & Warming up GPU (20 Tensors)..." << std::endl;
    cudaMemsetAsync(buffers[0], 0, INPUT_SIZE, raw_stream);
    for (int w = 0; w < 20; ++w) {
        context->enqueueV2(buffers, raw_stream, nullptr);
    }
    cudaStreamSynchronize(raw_stream);
    std::cout << "✅ Warmup Complete. GPU Stable. Handshake Code: " << static_cast<int>(current_profile) << std::endl;

    // Open Telemetry Appender Log File
    std::ofstream telemetry_log(telemetry_log_path, std::ios::app);

    // Gather Segregated File Arrays
    std::vector<std::string> good_paths, anomaly_paths; 
    cv::glob(good_dir, good_paths); 
    cv::glob(anomaly_dir, anomaly_paths); 

    if(good_paths.empty() && anomaly_paths.empty()) { 
        std::cerr << "No verification images found in segregated target directories." << std::endl; 
        return -1; 
    } 

    // Statistical Accuracy/Confusion Matrix Trackers
    int true_positives = 0, false_positives = 0;
    int true_negatives = 0, false_negatives = 0;
    int total_processed = 0;

    // Latency Telemetry Metrics Buffers (Accumulators for reporting)
    double total_preprocess_ms = 0.0, total_h2d_ms = 0.0;
    double total_inference_ms = 0.0, total_postprocess_ms = 0.0;
    double total_d2h_ms = 0.0;

    // Spin up Asynchronous Pipeline Framework 
    // ─── CHANGER 2: DEPLOY 10 PARALLEL I/O WORKER THREADS ───
    // ─── FIXED PARALLEL PRODUCER POOL INITIALIZATION ───
    FrameRingBuffer ring_buffer; 
    constexpr int NUM_WORKERS = 10;
    std::vector<std::thread> producer_pool;
    producer_pool.reserve(NUM_WORKERS);

    size_t total_files = good_paths.size() + anomaly_paths.size();
    // Pair each image string path explicitly with its correct label (0 = Good, 1 = Anomaly)
    std::vector<std::pair<std::string, int>> all_paths;
    all_paths.reserve(total_files);
    
    for (const auto& p : good_paths)    all_paths.emplace_back(p, 0);
    for (const auto& p : anomaly_paths) all_paths.emplace_back(p, 1);

    size_t chunk_size = total_files / NUM_WORKERS;
    size_t remainder = total_files % NUM_WORKERS;

    auto current_start = all_paths.begin();

    for (int i = 0; i < NUM_WORKERS; ++i) {
        size_t current_chunk = chunk_size + (i < remainder ? 1 : 0);
        auto current_end = current_start + current_chunk;

        if (current_start != current_end) {
            producer_pool.emplace_back(producer_file_reader, current_start, current_end, std::ref(ring_buffer));
        }
        current_start = current_end;
    }

    FrameData job; 
    auto pipeline_start_time = std::chrono::high_resolution_clock::now();

    // --- MAIN ENGINE PIPELINE EXECUTION LOOP --- 
    while (ring_buffer.pop(job)) { 
         
        auto t_pre_start = std::chrono::high_resolution_clock::now();

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
        bool do_rotate = (mom.m00 > 0 && (mom.m01 / mom.m00) < (job.frame.rows / 2.0f)); 

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

        cv::cuda::cvtColor(d_frame, d_frame_rgba, cv::COLOR_BGR2BGRA, 0, stream); 

        // STAGE 1: Perspective Warp 
        cv::cuda::warpPerspective(d_frame_rgba, d_warped_320, M, cv::Size(320, 320), 
                                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0,0), stream); 

        // STAGE 2: Lossless 180-Degree Flip Check
        if (do_rotate) { 
            cv::cuda::flip(d_warped_320, d_warped_320, -1, stream); 
        } 

        // STAGE 3: Lossless 90-Degree CCW Transpose and Flip
        cv::cuda::transpose(d_warped_320, d_rotated_320, stream); 
        cv::cuda::flip(d_rotated_320, d_rotated_320, 0, stream);  

        // STAGE 4: Window Crop 
        cv::Rect roi_box(32, 32, INPUT_W, INPUT_H); 
        d_final_padded = d_rotated_320(roi_box);  

        // STAGE 5: Channel Split & Remerge back into CV_8UC3
        cv::cuda::split(d_final_padded, bgra_channels, stream);  
        bgr_channels[0] = bgra_channels[0]; 
        bgr_channels[1] = bgra_channels[1]; 
        bgr_channels[2] = bgra_channels[2]; 
        cv::cuda::merge(bgr_channels, d_final_256, stream); 

        // STAGE 5.5: PRE-INFERENCE 1D EDGE EXTRACTION
        cv::cuda::cvtColor(d_final_256, d_hsv, cv::COLOR_BGR2HSV, 0, stream); 
        cv::cuda::inRange(d_hsv, cv::Scalar(90, 50, 50), cv::Scalar(135, 255, 255), d_mask, stream); 
        cv::cuda::reduce(d_mask, d_proj_cols, 0, cv::REDUCE_SUM, CV_32S, stream); 
         
        d_proj_cols.download(h_proj_cols, stream); 
        stream.waitForCompletion(); // Sync point 1

        int detected_l_wall = -1; 
        int detected_r_wall = -1; 
        const int* proj_ptr = h_proj_cols.ptr<int>(); 

        for (int c = 0; c < INPUT_W; ++c) { 
            if (proj_ptr[c] > 0) { 
                if (detected_l_wall == -1) detected_l_wall = c; 
                detected_r_wall = c; 
            } 
        } 

        int current_board_width = detected_r_wall - detected_l_wall; 
        if (detected_l_wall == -1 || current_board_width < VALID_MIN_WIDTH_THRESHOLD) { 
            std::cout << "⚠️ [" << job.filename << "] EARLY REJECT" << std::endl; 
            continue;  
        } 

        int final_left_wall  = detected_l_wall - 5; 
        int final_right_wall = detected_r_wall + 5; 

        auto t_pre_end = std::chrono::high_resolution_clock::now();
        double preprocess_ms = std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();

        // --- STAGED LATENCY BENCHMARKING (INFERENCE & TRANSFERS) ---
        auto t_h2d_start = std::chrono::high_resolution_clock::now();
        // STAGE 6: Interleaved-to-Planar Conversion (Custom HWC->CHW Kernel) 
        launch_hwc_to_chw_interface(d_final_256.data, static_cast<float*>(buffers[0]),  
                                    d_final_256.cols, d_final_256.rows, d_final_256.step,  
                                    raw_stream); 
        cudaStreamSynchronize(raw_stream);
        auto t_h2d_end = std::chrono::high_resolution_clock::now();
        double h2d_ms = std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_start).count();

        auto t_infer_start = std::chrono::high_resolution_clock::now();
        // STAGE 7: Execute Engine Model Asynchronous Inference 
        context->enqueueV2(buffers, raw_stream, nullptr); 
        cudaStreamSynchronize(raw_stream);
        auto t_infer_end = std::chrono::high_resolution_clock::now();
        double inference_ms = std::chrono::duration<double, std::milli>(t_infer_end - t_infer_start).count();

        auto t_post_start = std::chrono::high_resolution_clock::now();
        // STAGE 8: GPU Spatial Smoothing 
        gaussian_filter->apply(cv::cuda::GpuMat(INPUT_H, INPUT_W, CV_32FC1, buffers[1]), d_anomaly_map_blurred, stream); 
        
        // STAGE 9: Unified Structural Mask Pass + Regional Overrides 
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
        cudaStreamSynchronize(raw_stream);
        auto t_post_end = std::chrono::high_resolution_clock::now();
        double postprocess_ms = std::chrono::duration<double, std::milli>(t_post_end - t_post_start).count();

        auto t_d2h_start = std::chrono::high_resolution_clock::now();
        // STAGE 10: Host Pinned Extract DMA copy
        cudaMemcpyAsync(h_anomaly.data, d_anomaly_map_final.data, OUTPUT_SIZE, cudaMemcpyDeviceToHost, raw_stream); 
        stream.waitForCompletion(); // Sync point 2
        auto t_d2h_end = std::chrono::high_resolution_clock::now();
        double d2h_ms = std::chrono::duration<double, std::milli>(t_d2h_end - t_d2h_start).count();

        // Accumulate timing matrix metrics
        total_preprocess_ms += preprocess_ms;
        total_h2d_ms += h2d_ms;
        total_inference_ms += inference_ms;
        total_postprocess_ms += postprocess_ms;
        total_d2h_ms += d2h_ms;

        // Global Max Score Evaluation Pass
        double max_val; 
        cv::minMaxLoc(h_anomaly, nullptr, &max_val); 

        bool predicted_anomaly = (max_val > threshold);
        std::string status_str = predicted_anomaly ? "FAIL" : "PASS";

        // Performance Statistics Grid Matching
        if (job.label == 1) { // Ground Truth Anomaly
            if (predicted_anomaly) true_positives++;
            else false_negatives++;
        } else { // Ground Truth Good
            if (predicted_anomaly) false_positives++;
            else true_negatives++;
        }
        total_processed++;

        // Structured Inline JSON Telemetry Stream Logger
        if (telemetry_log.is_open()) {
            size_t free_mem = 0, total_mem = 0;
            cudaMemGetInfo(&free_mem, &total_mem);
            double vram_free_mb = static_cast<double>(free_mem) / (1024.0 * 1024.0);

            telemetry_log << "{\"timestamp\":\"" << get_iso_timestamp() 
                          << "\",\"board_id\":\"" << job.filename 
                          << "\",\"status\":\"" << status_str 
                          << "\",\"preprocess_ms\":" << preprocess_ms
                          << ",\"h2d_ms\":" << h2d_ms 
                          << ",\"exec_ms\":" << inference_ms 
                          << ",\"postprocess_ms\":" << postprocess_ms
                          << ",\"d2h_ms\":" << d2h_ms 
                          << ",\"profile\":\"" << profile_str 
                          << "\",\"vram_free_mb\":" << vram_free_mb << "}\n";
        }

        if (total_processed % 100 == 0) {
            std::cout << "🚀 [LIVE] Boards Processed: " << total_processed << " | Mode: " << profile_str << std::endl;
        }
    } 

    // ─── CHANGER 3: WAIT FOR ALL WORKERS AND SIGNAL FINISHED ───
    for (auto& worker : producer_pool) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    // ─── FIXED CLEAN-UP TIMELINE ───
   
    ring_buffer.set_finished();

    // 2. Safely join each worker thread exactly once
    for (auto& worker : producer_pool) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    if (telemetry_log.is_open()) telemetry_log.close();

    auto pipeline_end_time = std::chrono::high_resolution_clock::now();
    double total_wall_time_s = std::chrono::duration<double>(pipeline_end_time - pipeline_start_time).count();

    // --- CONSOLIDATED PERFORMANCE STATS EVALUATION ---
    double accuracy = total_processed > 0 ? static_cast<double>(true_positives + true_negatives) / total_processed : 0.0;
    double precision = (true_positives + false_positives) > 0 ? static_cast<double>(true_positives) / (true_positives + false_positives) : 0.0;
    double recall = (true_positives + false_negatives) > 0 ? static_cast<double>(true_positives) / (true_positives + false_negatives) : 0.0;
    double throughput = total_processed > 0 ? static_cast<double>(total_processed) / total_wall_time_s : 0.0;
    
    double avg_latency_ms = total_processed > 0 ? 
        (total_preprocess_ms + total_h2d_ms + total_inference_ms + total_postprocess_ms + total_d2h_ms) / total_processed : 0.0;

    std::cout << "\n" << std::string(65, '=') << "\n";
    std::cout << "               STAGEWISE LATENCY EVALUATION REPORT\n";
    std::cout << std::string(65, '-') << "\n";
    if (total_processed > 0) {
        std::cout << std::left << std::setw(25) << "CPU_PREPROCESS"   << " | " << std::setw(20) << (total_preprocess_ms / total_processed)  << " ms\n";
        std::cout << std::left << std::setw(25) << "GPU_H2D_TRANSFER"  << " | " << std::setw(20) << (total_h2d_ms / total_processed)         << " ms\n";
        std::cout << std::left << std::setw(25) << "GPU_EXECUTION"     << " | " << std::setw(20) << (total_inference_ms / total_processed)   << " ms\n";
        std::cout << std::left << std::setw(25) << "GPU_POSTPROCESS"   << " | " << std::setw(20) << (total_postprocess_ms / total_processed) << " ms\n";
        std::cout << std::left << std::setw(25) << "GPU_D2H_TRANSFER"  << " | " << std::setw(20) << (total_d2h_ms / total_processed)         << " ms\n";
    }
    std::cout << std::string(65, '=') << "\n\n";

    std::cout << "========================================\n";
    std::cout << "METRIC SUMMARY REPORT (Mode: " << profile_str << ")\n";
    std::cout << "========================================\n";
    std::cout << "Accuracy:   " << std::fixed << std::setprecision(4) << accuracy << "\n";
    std::cout << "Precision:  " << std::fixed << std::setprecision(4) << precision << "\n";
    std::cout << "Recall:     " << std::fixed << std::setprecision(4) << recall << "\n";
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughput << " FPS\n";
    std::cout << "Avg Latency:" << std::fixed << std::setprecision(2) << avg_latency_ms << " ms\n";
    std::cout << "========================================\n";

    // Clean up allocated assets 
    cudaFree(buffers[0]); 
    cudaFree(buffers[1]); 
    cudaFreeHost(h_output_pinned); 
    delete context;  
    delete engine;  
    delete runtime; 
    return 0; 
}
