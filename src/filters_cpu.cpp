#include "filters_cpu.hpp"
#include <cstdint>
#include <opencv2/core/hal/interface.h>
#include <stdexcept>
#include <algorithm> //for std::clamp
#include <cmath>
#include <thread>
#include <vector>
/*
Breakdown -
why grayscale input?
- Blur works on inetnsity, not color (for now)
- color blur would need 3 channels -> later upgrade

*/

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

// This helper clamps an int to [0,255]
static inline uint8_t clamp_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 0;
    return static_cast<uint8_t>(v);
}

/*
grayscale worker - processing a range of rows
- this function is what ONE thread runs
- It converts ONLY rows [y0, y1] to grayscale

Why [y0,y1]?
- inclusive start
- exclusive end
- thats a common c++ pattern because it avoids off by one errors
*/
static void grayscale_rows_worker(const cv::Mat &bgr,
    cv::Mat &gray,
    int y0,
    int y1
) {
    //Loop only over rows assigned to this thread
    for (int y = y0; y < y1; y++) {

        //grab pointers to row y
        const uint8_t* inRow = bgr.ptr<uint8_t>(y); // BGR data
        uint8_t* outRown = gray.ptr<uint8_t>(y); // grayscale data

        //Process every pixel in the row
        for (int x = 0; x < bgr.cols; x++) {
            int idx = x * 3; //3 bytes per pixel in BGR

            uint8_t B = inRow[idx + 0];
            uint8_t G = inRow[idx + 1];
            uint8_t R = inRow[idx + 2];

            int g = static_cast<int>(0.114 * B * 0.587 * G + 0.299 * R);
            outRow[x] = clamp_u8(g);
        }
    }
}

// Grayscale MT - spawns threads and splits rows
void grayscale_cpu_mt(const cv::Mat &bgr, cv::Mat &gray, int threads) {
    // 1 - validate input
    if (bgr.empty()) throw std::runtime_error("grayscale_cpu_mt: input empty");
    if (bgr.type() != CV_8UC3) throw std::runtime_error("grayscale_cpu_mt: expected CV_8UC3");

    // 2 - Clamp threads to a sane value
    // if user passes 0 or negative, force 1
    if (threads < 1) threads = 1;

    //If threads > rows, some threads would get 0 rows, so we can cap it
    threads = std::min(threads, bgr.rows);

    // 3 - Allocate output once (shared output buffer)
    gray.create(bgr.rows, bgr.cols, CV_8UC1);

    // 4 - decide how many rows each each thread should handle
    int totalRows = bgr.rows;

    //chunk = rows per thread (ceiling division)
    // ex - 100 rows, 6 threads -> chunk - 17
    int chunk = (totalRows + threads - 1) / threads;

    // 5 - Create the thread objects
    std::vector<std::thread> workers;
    workers.reserve(threads);

    // 6 - spawn threads
    for (int t = 0; t < threads; t++) {
        int y0 = t * chunk; // start row
        int y1 = std::min(totalRows, y0 + chunk); // end row (exclusive)

        // if y0 >= y1, it means no rows left for this thread, stop creating threads
        if (y0 >= y1) break;

        /*
       Create a thread that runs grayscale_rows_worker(...) 

       IMPORTANT - We pass:
       - bgr as const ref (read-only)
       - gray as ref (write to different rows)
       - y0/y1 range for this thread
        
        */
        workers.emplace_back(grayscale_rows_worker, std::cref(bgr), std::ref(gray), y0, y1);
    }

    // 7 - join threads (wait until all are done)
    // if uou forget to join, program may crash or exit early/
    for (auto& th : workers) {
        th.join();
    }

}

// Fast box blur using two 1D passes (horizontal then vertical).
// This is still a true box blur, just computed efficiently.
void box_blur_cpu_fast(const cv::Mat& gray, cv::Mat& blurred, int radius) {
    // 1) Validate input
    if (gray.empty()) throw std::runtime_error("box_blur_cpu_fast: input empty");
    if (gray.type() != CV_8UC1) throw std::runtime_error("box_blur_cpu_fast: expected CV_8UC1");
    if (radius < 1) throw std::runtime_error("box_blur_cpu_fast: radius must be >= 1");

    int w = gray.cols;
    int h = gray.rows;
    int k = 2 * radius + 1;

    // 2) Temporary buffer for the horizontal pass (store ints so sums don't overflow)
    // tmp[y*w + x] will hold the horizontally blurred value (still not divided vertically yet)
    std::vector<int> tmp(w * h, 0);

    // -----------------------------
    // PASS 1: Horizontal sliding sum
    // -----------------------------
    for (int y = 0; y < h; y++) {
        const uint8_t* row = gray.ptr<uint8_t>(y);

        // Compute initial window sum for x=0
        // Window covers [x-radius, x+radius], but we clamp to [0, w-1]
        int sum = 0;
        for (int dx = -radius; dx <= radius; dx++) {
            int xx = std::clamp(dx, 0, w - 1); // since x=0, x+dx = dx
            sum += row[xx];
        }

        // Store result for x=0 (not divided by k yet? we can divide now)
        tmp[y * w + 0] = sum;

        // Slide window across the row
        for (int x = 1; x < w; x++) {
            // Pixel leaving window: x-1-radius
            int x_out = std::clamp(x - 1 - radius, 0, w - 1);
            // Pixel entering window: x+radius
            int x_in  = std::clamp(x + radius, 0, w - 1);

            sum -= row[x_out];
            sum += row[x_in];

            tmp[y * w + x] = sum;
        }
    }

    // 3) Allocate output
    blurred.create(h, w, CV_8UC1);

    // -----------------------------
    // PASS 2: Vertical sliding sum
    // Now we blur the horizontal sums vertically and divide by k*k.
    // -----------------------------
    for (int x = 0; x < w; x++) {
        // Initial vertical window sum for y=0
        int sum = 0;
        for (int dy = -radius; dy <= radius; dy++) {
            int yy = std::clamp(dy, 0, h - 1); // since y=0, y+dy = dy
            sum += tmp[yy * w + x];
        }

        // Write output for y=0
        int area = k * k;
        blurred.ptr<uint8_t>(0)[x] = static_cast<uint8_t>(std::clamp(sum / area, 0, 255));

        // Slide window down the column
        for (int y = 1; y < h; y++) {
            int y_out = std::clamp(y - 1 - radius, 0, h - 1);
            int y_in  = std::clamp(y + radius, 0, h - 1);

            sum -= tmp[y_out * w + x];
            sum += tmp[y_in * w + x];

            blurred.ptr<uint8_t>(y)[x] = static_cast<uint8_t>(std::clamp(sum / area, 0, 255));
        }
    }
}
// pass 1 worker - horizontal blur for rows [y0, y1]
// writes into tmp[] but only for those rows -> safe
static void blur_horizontal_rows_worker(
    const cv::Mat& gray,
    std::vector<int>& tmp,
    int radius,
    int y0, 
    int y1
) {
    int w = gray.cols;
    int k = 2 * radius + 1;

    for (int y = y0; y < y1; y++) {
        const uint8_t* row = gray.ptr<uint8_t>(y);

        //initial sum for x = 0
        int sum = 0;
    }

}

void sobel_cpu(const cv::Mat& gray, cv::Mat& edges) {
    // Validate input
    if (gray.empty()) throw std::runtime_error("sobel_cpu: input empty");
    if (gray.type() != CV_8UC1) throw std::runtime_error("sobel_cpu: expected CV_8UC1");

    int w = gray.cols;
    int h = gray.rows;

    // Allocate output
    edges.create(h, w, CV_8UC1);

    // Sobel kernels
    // Gx (horizontal gradient):
    //  -1  0  1
    //  -2  0  2
    //  -1  0  1
    // Gy (vertical gradient):
    //  -1 -2 -1
    //   0  0  0
    //   1  2  1

    for (int y = 1; y < h - 1; y++) {
        const uint8_t* row_m1 = gray.ptr<uint8_t>(y - 1);
        const uint8_t* row_0  = gray.ptr<uint8_t>(y);
        const uint8_t* row_p1 = gray.ptr<uint8_t>(y + 1);
        uint8_t* outRow = edges.ptr<uint8_t>(y);

        for (int x = 1; x < w - 1; x++) {
            // Compute Gx (horizontal gradient)
            int gx = -1 * row_m1[x - 1] + 1 * row_m1[x + 1]
                   + -2 * row_0[x - 1]  + 2 * row_0[x + 1]
                   + -1 * row_p1[x - 1] + 1 * row_p1[x + 1];

            // Compute Gy (vertical gradient)
            int gy = -1 * row_m1[x - 1] + -2 * row_m1[x] + -1 * row_m1[x + 1]
                   +  1 * row_p1[x - 1] +  2 * row_p1[x] +  1 * row_p1[x + 1];

            // Magnitude: sqrt(gx^2 + gy^2)
            int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy));
            outRow[x] = clamp_u8(magnitude);
        }
    }

    // Set border pixels to 0 (can't compute gradient at edges)
    for (int y = 0; y < h; y++) {
        uint8_t* outRow = edges.ptr<uint8_t>(y);
        if (y == 0 || y == h - 1) {
            for (int x = 0; x < w; x++) {
                outRow[x] = 0;
            }
        } else {
            outRow[0] = 0;
            outRow[w - 1] = 0;
        }
    }
}

static void sobel_rows_worker(const cv::Mat& gray, cv::Mat& edges, int y0, int y1) {
    const int Gx[3][3] = {{-1,0,1}, {-2,0,2}, {-1,0,1}};
    const int Gy[3][3] = {{-1,-2,-1}, {0,0,0}, (1,2,2)};


}