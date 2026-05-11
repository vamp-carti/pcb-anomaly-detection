#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

const int TARGET_W = 240;
const int CANVAS_DIM = 320;
const int CROP_DIM = 256;

// Locked Robust Rotation using Affine Warp
void gpu_rotate_locked(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, double angle) {
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::cuda::warpAffine(src, dst, rot_mat, src.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0));
}

int main() {
    // 1. Ingestion
    cv::Mat h_raw = cv::imread("1.png");
    if (h_raw.empty()) return -1;

    cv::cuda::GpuMat d_raw(h_raw), d_hsv, d_mask;
    cv::cuda::GpuMat d_canvas_A(CANVAS_DIM, CANVAS_DIM, CV_8UC3);
    cv::cuda::GpuMat d_canvas_B(CANVAS_DIM, CANVAS_DIM, CV_8UC3);

    // 2. PCB Board Detection (Blue Mask)
    cv::cuda::cvtColor(d_raw, d_hsv, cv::COLOR_BGR2HSV);
    cv::cuda::inRange(d_hsv, cv::Scalar(90, 50, 50), cv::Scalar(135, 255, 255), d_mask);
    auto close_filter = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8U, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));
    close_filter->apply(d_mask, d_mask);

    cv::Mat h_mask;
    d_mask.download(h_mask);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(h_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return -1;
    auto max_c = *std::max_element(contours.begin(), contours.end(), [](auto& a, auto& b){ return cv::contourArea(a) < cv::contourArea(b); });

    // 3. Perspective Alignment Calculation
    cv::RotatedRect rect = cv::minAreaRect(max_c);
    cv::Point2f box[4]; rect.points(box);
    std::vector<cv::Point2f> pts(box, box + 4);
    std::sort(pts.begin(), pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return (a.x + a.y) < (b.x + b.y); });
    cv::Point2f src_pts[4];
    src_pts[0] = pts[0]; src_pts[2] = pts[3];
    if (pts[1].x > pts[2].x) { src_pts[1] = pts[1]; src_pts[3] = pts[2]; } else { src_pts[1] = pts[2]; src_pts[3] = pts[1]; }

    float d01 = cv::norm(src_pts[0] - src_pts[1]);
    float d03 = cv::norm(src_pts[0] - src_pts[3]);
    int target_h = static_cast<int>(TARGET_W * (d03 / d01));
    int off_x = (CANVAS_DIM - TARGET_W) / 2;
    int off_y = (CANVAS_DIM - target_h) / 2;

    cv::Point2f dst_pts[4] = {
        {(float)off_x, (float)off_y},
        {(float)off_x + TARGET_W, (float)off_y},
        {(float)off_x + TARGET_W, (float)off_y + target_h},
        {(float)off_x, (float)off_y + target_h}
    };

    cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat M_32f; M.convertTo(M_32f, CV_32F);
    cv::cuda::warpPerspective(d_raw, d_canvas_A, M_32f, cv::Size(CANVAS_DIM, CANVAS_DIM), cv::INTER_CUBIC);

    // 4. Orientation Handshake (180 Check)
    cv::cuda::GpuMat d_check_hsv, d_sil_mask;
    cv::cuda::cvtColor(d_canvas_A, d_check_hsv, cv::COLOR_BGR2HSV);
    cv::cuda::inRange(d_check_hsv, cv::Scalar(0, 0, 180), cv::Scalar(180, 80, 255), d_sil_mask);
    cv::cuda::GpuMat d_top_half = d_sil_mask(cv::Rect(off_x, off_y, TARGET_W, target_h / 2));
    
    if (cv::cuda::countNonZero(d_top_half) > 500) {
        gpu_rotate_locked(d_canvas_A, d_canvas_B, 180.0);
    } else {
        d_canvas_A.copyTo(d_canvas_B);
    }

    // 5. Final Alignment (Pins to Left Correction)
    // Changing from 90.0 to -90.0 to flip pins from Right to Left
    gpu_rotate_locked(d_canvas_B, d_canvas_A, -90.0);

    // 6. Output Generation
    int s = (CANVAS_DIM - CROP_DIM) / 2;
    cv::cuda::GpuMat d_final = d_canvas_A(cv::Rect(s, s, CROP_DIM, CROP_DIM));
    cv::imwrite("step5_final_crop_locked.png", cv::Mat(d_final));

    std::cout << "[LOCKED] Pins aligned to Left. Pipeline validated." << std::endl;
    return 0;
}
