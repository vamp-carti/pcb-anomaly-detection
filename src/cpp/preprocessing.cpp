#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    std::string input_path = "/app/temp/images/*.png";
    std::string base_output = "debug_output";
    
    fs::create_directories(base_output + "/stage1_warp_320");
    fs::create_directories(base_output + "/stage2_rotated_320");
    fs::create_directories(base_output + "/stage3_final_crop_256");

    std::vector<std::string> file_names;
    cv::glob(input_path, file_names);

    // PERSISTENT GPU MEMORY
    cv::cuda::GpuMat d_frame, d_warped_320, d_rotated_320, d_float, d_rgb;
    cv::cuda::Stream stream;

    // Normalization Constants (ImageNet)
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);

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

        // --- MATRIX BAKING (Targeting 240px within 320x320) ---
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

        // STAGE 1: Warp to 320x320
        cv::cuda::warpPerspective(d_frame, d_warped_320, M, cv::Size(320, 320), 
                                 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0), stream);

        // STAGE 2: 90 CCW Rotation
        cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point2f(160.0f, 160.0f), 90.0, 1.0);
        cv::cuda::warpAffine(d_warped_320, d_rotated_320, rot_mat, cv::Size(320, 320), 
                            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0), stream);

        // STAGE 3: Final 256x256 Crop
        cv::Rect roi(32, 32, 256, 256);
        cv::cuda::GpuMat d_final = d_rotated_320(roi);

        // STAGE 4: RGB + Normalization (Parity with Python)
        cv::cuda::cvtColor(d_final, d_rgb, cv::COLOR_BGR2RGB, 0, stream);
        d_rgb.convertTo(d_float, CV_32FC3, 1.0/255.0, stream); // Scale to [0,1]
        
        // Pixel-wise: (x - mean) / std
        cv::cuda::subtract(d_float, mean, d_float, cv::noArray(), -1, stream);
        cv::cuda::divide(d_float, std, d_float, 1.0, -1, stream);

        // STAGE 5: Planar Conversion (HWC -> CHW)
        std::vector<cv::cuda::GpuMat> planes(3);
        cv::cuda::split(d_float, planes, stream);
        
        // Allocate contiguous memory for RRR...GGG...BBB...
        cv::cuda::GpuMat d_planar(1, 256 * 256 * 3, CV_32F);
        for(int i=0; i<3; ++i) {
            planes[i].reshape(1, 1).copyTo(d_planar.colRange(i*256*256, (i+1)*256*256), stream);
        }

        // --- VALIDATION PROBE (Download for Checksum) ---
        cv::Mat h_final_float;
        d_float.download(h_final_float, stream); // Interleaved for pixel check
        stream.waitForCompletion();

        // LOG PIXEL (128, 128) for Python comparison
        cv::Vec3f p = h_final_float.at<cv::Vec3f>(128, 128);
        std::cout << "[" << filename << "] Pixel (128,128) -> R:" << p[0] << " G:" << p[1] << " B:" << p[2] << std::endl;

        // Save Stage 2 for visual geometry confirmation
        cv::Mat h_stage2;
        d_rotated_320.download(h_stage2, stream);
        stream.waitForCompletion();
        cv::imwrite(base_output + "/stage2_rotated_320/" + filename, h_stage2);
    }

    return 0;
}
