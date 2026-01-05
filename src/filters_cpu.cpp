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
        throw std::runtime_error("Expected input")
    }
}