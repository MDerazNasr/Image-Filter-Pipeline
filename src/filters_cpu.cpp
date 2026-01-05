#include "filters_cpu.hpp"
#include <cstdint>
#include <stdexcept>

// Helper: clamp an integer into [0, 255]
static inline uint8_t clamp_u8(int v){
    if (v < 0) return 0;
    if (v > 255) return 255;
    return static_cast<uint8_t>(v);
}

void grayscale_cpu(const cv::Mat& bgr, cv::Mat& gray) {
    // Validate input so we fail loudly instead of silently
    if (bgr.empty()) {
        throw std::runtime_error("Input image is empty (failed to load?)");

    }
    if (bgr.type() != CV_8UC3) {
        throw std::runtime_error("Expected input");
    }

    // 2) Allocate output memory (same width/height), but 1 channel)
    gray.create(bgr.rows, bgr.cols, CV_8UC1);
    
    // 3) loop through rows (y) and columns (x)
    for (int y = 0; y < bgr.rows; y++) {
        // ptr<uint8_t>(y) gives us a pointer to the FIRST byte of row y.
        // for CV_8UC3, each pixel is 3 bytes (B,G,R)
        const uint8_t* inRow = bgr.ptr<uint8_t>(y);

        // For CV_8UC1, each pixel is 1 byte (gray)
        uint8_t* outRow = gray.ptr<uint8_t>(y);

        for (int x = 0; x < bgr.cols; x++) {
            int idx = x * 3; //3 bytes per pixel in the input row

            uint8_t B = inRow[idx + 0];
            uint8_t G = inRow[idx + 1];
            uint8_t R = inRow[idx + 2];

            // Weighted grayscale (brightness perception)
            int g = static_cast<int>(0.114 * B + 0.587 * G + 0.299 * R);

            outRow[x] = clamp_u8(g);
        }

    }

}