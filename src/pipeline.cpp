#include "pipeline.hpp"
#include "filters_cpu.hpp"
#include "workspace.hpp"
#include "utils.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>

// Helper: convert Mode to string for printing
static const char* modeName(Mode m) {
    switch (m) {
        case Mode::CPU_SINGLE: return "cpu-single";
        case Mode::CPU_MT:     return "cpu-mt";
        case Mode::GPU:        return "gpu";
    }
    return "unknown";
}

void Pipeline::run(const Args& args) {
    // Decide which path is used
    if (!args.imagePath.empty()) {
        runImage(args);
        return;
    }
    if (!args.videoPath.empty()) {
        runVideo(args);
        return;
    }
    throw std::runtime_error("You must provide --image or --video");
}

void Pipeline::runImage(const Args& args) {
    // 1) Load image (OpenCV only for IO)
    cv::Mat bgr = cv::imread(args.imagePath, cv::IMREAD_COLOR);
    if (bgr.empty()) throw std::runtime_error("Failed to load image: " + args.imagePath);

    if (args.mode == Mode::GPU) {
        // Mac note: CUDA unavailable. Keep placeholder.
        throw std::runtime_error("GPU mode not available on this machine (CUDA requires NVIDIA).");
    }

    cv::Mat gray, blurred, edges;
    CpuWorkspace ws;
    ws.ensureSize(bgr.cols, bgr.rows);

    Timer total;

    // --- Stage 1: Grayscale ---
    Timer t1;
    if (args.mode == Mode::CPU_MT) {
        grayscale_cpu_mt(bgr, gray, args.threads);
    } else {
        grayscale_cpu(bgr, gray, 1);
    }
    double msGray = t1.ms();

    // --- Stage 2: Blur (fast + reusable workspace) ---
    Timer t2;
    if (args.mode == Mode::CPU_MT) {
        box_blur_cpu_fast_mt_ws(gray, blurred, args.radius, args.threads, ws);
    } else {
        box_blur_cpu_fast(gray, blurred, args.radius, 1);
    }
    double msBlur = t2.ms();

    // --- Stage 3: Sobel ---
    Timer t3;
    if (args.mode == Mode::CPU_MT) {
        sobel_cpu_mt(blurred, edges, args.threads);
    } else {
        sobel_cpu(blurred, edges, 1);
    }
    double msSobel = t3.ms();

    // 2) Save output (OpenCV only for IO)
    if (!cv::imwrite(args.outPath, edges)) {
        throw std::runtime_error("Failed to write output: " + args.outPath);
    }

    // 3) Print timing summary
    std::cout << "[IMAGE] mode=" << modeName(args.mode)
              << " size=" << bgr.cols << "x" << bgr.rows
              << " radius=" << args.radius
              << " threads=" << args.threads << "\n";
    std::cout << "  grayscale: " << msGray  << " ms\n";
    std::cout << "  blur:      " << msBlur  << " ms\n";
    std::cout << "  sobel:     " << msSobel << " ms\n";
    std::cout << "  total:     " << total.ms() << " ms\n";
}

void Pipeline::runVideo(const Args& args) {
    cv::VideoCapture cap(args.videoPath);
    if (!cap.isOpened()) throw std::runtime_error("Failed to open video: " + args.videoPath);

    if (args.mode == Mode::GPU) {
        throw std::runtime_error("GPU mode not available on this machine (CUDA requires NVIDIA).");
    }

    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fpsIn = cap.get(cv::CAP_PROP_FPS);
    if (fpsIn <= 0) fpsIn = 30.0;

    // Output writer (expects BGR frames)
    cv::VideoWriter writer;
    int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    writer.open(args.outPath, fourcc, fpsIn, cv::Size(w, h), true);
    if (!writer.isOpened()) throw std::runtime_error("Failed to open VideoWriter: " + args.outPath);

    // Pre-allocate reusable buffers (VERY IMPORTANT)
    cv::Mat frame;
    cv::Mat gray(h, w, CV_8UC1);
    cv::Mat blurred(h, w, CV_8UC1);
    cv::Mat edges(h, w, CV_8UC1);
    cv::Mat edgesBgr(h, w, CV_8UC3);

    CpuWorkspace ws;
    ws.ensureSize(w, h);

    // We will compute average stage times across all frames
    double sumGray = 0.0, sumBlur = 0.0, sumSobel = 0.0;
    int frames = 0;

    Timer total;

    while (true) {
        if (!cap.read(frame)) break;
        frames++;

        // Stage 1: grayscale
        Timer t1;
        if (args.mode == Mode::CPU_MT) grayscale_cpu_mt(frame, gray, args.threads);
        else grayscale_cpu(frame, gray, 1);
        sumGray += t1.ms();

        // Stage 2: blur
        Timer t2;
        if (args.mode == Mode::CPU_MT) box_blur_cpu_fast_mt_ws(gray, blurred, args.radius, args.threads, ws);
        else box_blur_cpu_fast(gray, blurred, args.radius, 1);
        sumBlur += t2.ms();

        // Stage 3: sobel
        Timer t3;
        if (args.mode == Mode::CPU_MT) sobel_cpu_mt(blurred, edges, args.threads);
        else sobel_cpu(blurred, edges, 1);
        sumSobel += t3.ms();

        // Convert edges (1 channel) -> BGR so writer accepts it
        cv::cvtColor(edges, edgesBgr, cv::COLOR_GRAY2BGR);
        writer.write(edgesBgr);

        // Print occasional progress
        if (frames % 60 == 0) {
            std::cout << "frame " << frames << " processed\n";
        }
    }

    double totalMs = total.ms();
    double fpsOut = (totalMs > 0) ? (frames / (totalMs / 1000.0)) : 0.0;

    std::cout << "[VIDEO] mode=" << modeName(args.mode)
              << " size=" << w << "x" << h
              << " radius=" << args.radius
              << " threads=" << args.threads << "\n";
    std::cout << "  frames:    " << frames << "\n";
    std::cout << "  avg gray:  " << (frames ? sumGray / frames : 0.0) << " ms\n";
    std::cout << "  avg blur:  " << (frames ? sumBlur / frames : 0.0) << " ms\n";
    std::cout << "  avg sobel: " << (frames ? sumSobel / frames : 0.0) << " ms\n";
    std::cout << "  total:     " << totalMs << " ms\n";
    std::cout << "  avg FPS:   " << fpsOut << "\n";
}
