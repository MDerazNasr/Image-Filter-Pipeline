// src/main.cpp
//
// Day 4 version (works for Day 3 image mode + Day 4 video mode).
// - Image: grayscale -> blur -> sobel -> write output image
// - Video: per-frame grayscale -> blur -> sobel -> write output video
//
// IMPORTANT:
// - We only use OpenCV for reading/writing and for cvtColor (1ch->3ch for video writer).
// - All filtering logic (grayscale/blur/sobel) is our own code in filters_cpu.cpp.

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <stdexcept>

#include "filters_cpu.hpp"
#include "utils.hpp"

// Print help if the user runs it wrong.
static void printUsage() {
    std::cout <<
    "Usage:\n"
    "  Image:\n"
    "    ./pipeline --image <path> --out <path>\n"
    "    Example: ./pipeline --image data/input.jpg --out output/out_edges.png\n"
    "\n"
    "  Video:\n"
    "    ./pipeline --video <path> --out <path>\n"
    "    Example: ./pipeline --video data/input.mp4 --out output/out_edges.mp4\n";
}

// Very small helper: convert a flag's next argument into a value.
// Example: --image data/input.jpg  (value is "data/input.jpg")
static std::string needValue(int& i, int argc, char** argv, const std::string& flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error("Missing value after " + flag);
    }
    return std::string(argv[++i]); // increment i and return argv[i]
}

int main(int argc, char** argv) {
    // argc = number of command-line arguments
    // argv = array of strings (C-strings) containing those arguments
    if (argc < 5) {
        printUsage();
        return 1;
    }

    // These are filled in by parsing command line flags.
    std::string imagePath;
    std::string videoPath;
    std::string outPath;

    // -------------------------
    // 1) Parse CLI args
    // -------------------------
    try {
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];

            if (a == "--image") {
                imagePath = needValue(i, argc, argv, a);
            } else if (a == "--video") {
                videoPath = needValue(i, argc, argv, a);
            } else if (a == "--out") {
                outPath = needValue(i, argc, argv, a);
            } else {
                std::cerr << "Unknown flag: " << a << "\n";
                printUsage();
                return 1;
            }
        }

        // Basic validation:
        if (outPath.empty()) {
            throw std::runtime_error("Missing --out <path>");
        }
        if (imagePath.empty() && videoPath.empty()) {
            throw std::runtime_error("You must provide either --image or --video");
        }
        if (!imagePath.empty() && !videoPath.empty()) {
            throw std::runtime_error("Provide only one of --image or --video (not both)");
        }

        // -------------------------
        // 2) IMAGE MODE
        // -------------------------
        if (!imagePath.empty()) {
            // Load a color image (OpenCV loads color as BGR)
            cv::Mat bgr = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (bgr.empty()) {
                throw std::runtime_error("imread failed (check path): " + imagePath);
            }

            // Output buffers (OpenCV matrices)
            cv::Mat gray;
            cv::Mat blurred;
            cv::Mat edges;

            // --- grayscale ---
            Timer tGray;
            int T = 8; // number of threads
            grayscale_cpu(bgr, gray, T);
            double msGray = tGray.ms();

            // --- blur ---
            // Use the FAST blur if you implemented it; otherwise change this to box_blur_cpu(...)
            Timer tBlur;
            box_blur_cpu_fast(gray, blurred, 1, T); // radius=1 => 3x3 box blur
            double msBlur = tBlur.ms();

            // --- sobel edges ---
            Timer tSobel;
            sobel_cpu(blurred, edges, T);
            double msSobel = tSobel.ms();

            // Save result
            if (!cv::imwrite(outPath, edges)) {
                throw std::runtime_error("imwrite failed (check permissions/extension): " + outPath);
            }

            std::cout << "Saved: " << outPath << "\n";
            std::cout << "Grayscale: " << msGray << " ms\n";
            std::cout << "Blur:      " << msBlur << " ms\n";
            std::cout << "Sobel:     " << msSobel << " ms\n";
        }

        // -------------------------
        // 3) VIDEO MODE
        // -------------------------
        else {
            cv::VideoCapture cap(videoPath);
            if (!cap.isOpened()) {
                throw std::runtime_error("Failed to open video: " + videoPath);
            }

            // Read video properties
            int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            double fpsIn = cap.get(cv::CAP_PROP_FPS);
            if (fpsIn <= 0) fpsIn = 30.0; // fallback if metadata is missing

            // VideoWriter often expects 3-channel BGR frames.
            // We'll compute edges as 1-channel, then convert to BGR for writing.
            cv::VideoWriter writer;
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // common mp4 codec

            writer.open(outPath, fourcc, fpsIn, cv::Size(w, h), true);
            if (!writer.isOpened()) {
                throw std::runtime_error("Failed to open VideoWriter: " + outPath);
            }

            // Reusable buffers (IMPORTANT: don't allocate each frame)
            cv::Mat frame;                  // input BGR frame (3-channel)
            cv::Mat gray(h, w, CV_8UC1);     // grayscale (1-channel)
            cv::Mat blurred(h, w, CV_8UC1);  // blurred (1-channel)
            cv::Mat edges(h, w, CV_8UC1);    // sobel edges (1-channel)
            cv::Mat edgesBgr(h, w, CV_8UC3); // edges converted to BGR for VideoWriter

            Timer total;
            int frames = 0;

            while (true) {
                if (!cap.read(frame)) break; // end of video
                frames++;

                Timer perFrame;
                int T = 8; // start with 8, change later or make CLI flag

                // Our custom CPU pipeline
                grayscale_cpu(frame, gray, T);
                box_blur_cpu_fast(gray, blurred, 1, T);
                sobel_cpu(blurred, edges, T);

                // Convert 1-channel -> 3-channel so the writer can encode it
                cv::cvtColor(edges, edgesBgr, cv::COLOR_GRAY2BGR);

                // Write frame to output video
                writer.write(edgesBgr);

                // Print every ~30 frames so console isn't spammed
                if (frames % 30 == 0) {
                    std::cout << "frame " << frames << " time=" << perFrame.ms() << " ms\n";
                }
            }

            double totalMs = total.ms();
            double fpsOut = (totalMs > 0) ? (frames / (totalMs / 1000.0)) : 0.0;

            std::cout << "Saved: " << outPath << "\n";
            std::cout << "Video done. Frames=" << frames
                      << " total=" << totalMs << " ms"
                      << " avgFPS=" << fpsOut << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
