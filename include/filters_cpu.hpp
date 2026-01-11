#pragma once
#include <opencv2/opencv.hpp>
#include "workspace.hpp"

//Converts a color image (BGR) into grayscale
// Input: bgr (CV_8UC3)
// Output: gray (CV_8UC1)
void grayscale_cpu(const cv::Mat& bgr, cv::Mat& gray, int threads);

// Box blur on grayscale image
// radius = 1 -> 3x3, radius = 2 -> 5x5 etc.

void box_blur_cpu_fast(const cv::Mat& gray, cv::Mat& blurred, int radius, int threads);

//Sobel edge detection on grayscale image
void sobel_cpu(const cv::Mat& gray, cv::Mat& edges, int threads);

// Multi-threaded versions
void grayscale_cpu_mt(const cv::Mat& bgr, cv::Mat& gray, int threads);
void sobel_cpu_mt(const cv::Mat& gray, cv::Mat& edges, int threads);

/*
#pragma once prvents the header from being included twice

const cv... means: i promise bot to modify bgr
                    dont copy the whole image
cv::Mat& gray means: I will fill this output image
                        passed by reference so we can write into it


For multihtreading we are splitting these functions by rows
-- they tell the compiler: these functions exist somewhere


*/
// Fast blur using workspace (no allocations inside)
void box_blur_cpu_fast_mt_ws(
    const cv::Mat &gray,
    cv::Mat &blurred,
    int radius,
    int threads,
    CpuWorkspace& ws
);