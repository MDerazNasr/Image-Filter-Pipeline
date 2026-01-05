#pragma once
#include <opencv2/opencv.hpp>

//Converts a color image (BGR) into grayscale
// Input: bgr (CV_8UC3)
// Output: gray (CV_8UC1)
void grayscale_cpu(const cv::Mat& bgr, cv::Mat& gray);

// Box blur on grayscale image
// radius = 1 -> 3x3, radius = 2 -> 5x5 etc.

void box_blur_cpu_fast(const cv::Mat& gray, cv::Mat& blurred, int radius);

//Sobel edge detection on grayscale image
void sobel_cpu(const cv::Mat& gray, cv::Mat& edges);

/*
#pragma once prvents the header from being included twice

const cv... means: i promise bot to modify bgr
                    dont copy the whole image
cv::Mat& gray means: I will fill this output image
                        passed by reference so we can write into it



*/