#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <NvInfer.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

int main() {
    std::string input_path = "/app/temp/images/*.png";
    std::string engine_path = "pcb.engine";
    std::string base_output = "debug_output";
    float threshold = -0.2800f;

    fs::create_directories(base_output + "/stage1_warp_320");
    fs::create_directories(base_output + "/stage2_rotated_320");
    fs::create_directories(base_output + "/stage3_final_crop_256");
    fs::create_directories(base_output + "/anomaly_maps");

    // --- TENSORRT INITIALIZATION ---
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) { std::cerr << "Engine not found!" << std::endl; return -1; }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    char* engine_data = new char[size];
    file.read(engine_data, size);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data, size);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    delete[] engine_data;

    void* buffers[2];
    cudaMalloc(&buffers[0], 1 * 3 * 256 * 256 * sizeof(float));
    cudaMalloc(&buffers[1], 1 * 1 * 256 * 256 * sizeof(float));

    std::vector<std::string> file_names;
    cv::glob(input_path, file_names);

    // GPU PERSISTENCE
    cv::cuda::GpuMat d_frame, d_warped_320, d_rotated_320, d_float, d_rgb, d_planar;
    cv::cuda::Stream stream;
    cudaStream_t raw_stream = cv::cuda::StreamAccessor::getStream(stream);

    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std_dev(0.229, 0.224, 0.225);

    for (const auto& file_path : file_names) {
        std::string filename = fs::path(file_path).filename().string();
        cv::Mat frame = cv::imread(file_path);
        if (frame.empty()) continue;

        // --- PRE-PROCESSING (CPU) ---
        cv::Mat hsv, mask;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(90, 50, 50), cv::Scalar(135, 255, 255), mask);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
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
        if (mom.m00 > 0 && (mom.m01 / mom.m00) < (frame.rows / 2.0f)) {
            std::rotate(src_pts.begin(), src_pts.begin() + 2, src_pts.end());
        }

        // --- ISOTROPIC MATRIX BAKING ---
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

        // --- GPU PIPELINE ---
        d_frame.upload(frame, stream);

        // STAGE 1: Warp
        cv::cuda::warpPerspective(d_frame, d_warped_320, M, cv::Size(320, 320), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0), stream);
        cv::Mat h_s1; d_warped_320.download(h_s1, stream);
        stream.waitForCompletion();
        cv::imwrite(base_output + "/stage1_warp_320/" + filename, h_s1);

        // STAGE 2: Rotate
        cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point2f(160.0f, 160.0f), 90.0, 1.0);
        cv::cuda::warpAffine(d_warped_320, d_rotated_320, rot_mat, cv::Size(320, 320), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0), stream);
        cv::Mat h_s2; d_rotated_320.download(h_s2, stream);
        stream.waitForCompletion();
        cv::imwrite(base_output + "/stage2_rotated_320/" + filename, h_s2);

        // STAGE 3: Crop
        cv::Rect roi(32, 32, 256, 256);
        cv::cuda::GpuMat d_final = d_rotated_320(roi);
        cv::Mat h_s3; d_final.download(h_s3, stream);
        stream.waitForCompletion();
        cv::imwrite(base_output + "/stage3_final_crop_256/" + filename, h_s3);

        // STAGE 4: Inference Pre-processing (RGB + Norm + Planar)
        cv::cuda::cvtColor(d_final, d_rgb, cv::COLOR_BGR2RGB, 0, stream);
        d_rgb.convertTo(d_float, CV_32FC3, 1.0/255.0, stream);
        cv::cuda::subtract(d_float, mean, d_float, cv::noArray(), -1, stream);
        cv::cuda::divide(d_float, std_dev, d_float, 1.0, -1, stream);

        std::vector<cv::cuda::GpuMat> planes(3);
        cv::cuda::split(d_float, planes, stream);
        d_planar.create(1, 256 * 256 * 3, CV_32F);
        for(int i=0; i<3; ++i) {
            planes[i].reshape(1, 1).copyTo(d_planar.colRange(i*256*256, (i+1)*256*256), stream);
        }

        // STAGE 5: Inference
        cudaMemcpyAsync(buffers[0], d_planar.data, 3 * 256 * 256 * sizeof(float), cudaMemcpyDeviceToDevice, raw_stream);
        context->enqueueV2(buffers, raw_stream, nullptr);

        cv::Mat h_anomaly(256, 256, CV_32F);
        cudaMemcpyAsync(h_anomaly.data, buffers[1], 256 * 256 * sizeof(float), cudaMemcpyDeviceToHost, raw_stream);
        stream.waitForCompletion();

        double max_val;
        cv::minMaxLoc(h_anomaly, nullptr, &max_val);
        std::cout << "[" << filename << "] Score: " << max_val << " | Status: " << (max_val > threshold ? "FAIL" : "PASS") << std::endl;

        // STAGE 6: Anomaly Map Save
        cv::Mat adj_map;
        cv::normalize(h_anomaly, adj_map, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(adj_map, adj_map, cv::COLORMAP_JET);
        cv::imwrite(base_output + "/anomaly_maps/" + filename, adj_map);
    }

    cudaFree(buffers[0]); cudaFree(buffers[1]);
    delete context; delete engine; delete runtime;
    return 0;
}
