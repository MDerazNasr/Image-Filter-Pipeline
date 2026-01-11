#include "pipeline.hpp"
#include <iostream>
#include <string>
#include <stdexcept>

// Print usage instructions
static void usage() {
    std::cout <<
    "Usage:\n"
    "  Image:\n"
    "    ./pipeline --image <path> --mode <cpu-single|cpu-mt> --out <path> [--threads N] [--radius R]\n"
    "  Video:\n"
    "    ./pipeline --video <path> --mode <cpu-single|cpu-mt> --out <path> [--threads N] [--radius R]\n"
    "\nExamples:\n"
    "  ./pipeline --image data/input.jpg --mode cpu-single --radius 1 --out output/out_edges.png\n"
    "  ./pipeline --image data/input.jpg --mode cpu-mt --threads 8 --radius 2 --out output/out_edges_mt.png\n"
    "  ./pipeline --video data/input.mp4 --mode cpu-mt --threads 8 --radius 1 --out output/out_edges_mt.mp4\n";
}

// Convert string -> Mode enum
static Mode parseMode(const std::string& s) {
    if (s == "cpu-single") return Mode::CPU_SINGLE;
    if (s == "cpu-mt")     return Mode::CPU_MT;
    if (s == "gpu")        return Mode::GPU; // placeholder
    throw std::runtime_error("Unknown mode: " + s);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    Args args;
    std::string modeStr;

    // Simple flag parsing
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];

        // Helper: read the next argument as a value
        auto needValue = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value after " + flag);
            return argv[++i];
        };

        if (a == "--image")   args.imagePath = needValue(a);
        else if (a == "--video")  args.videoPath = needValue(a);
        else if (a == "--out")    args.outPath = needValue(a);
        else if (a == "--mode")   modeStr = needValue(a);
        else if (a == "--threads") args.threads = std::stoi(needValue(a));
        else if (a == "--radius")  args.radius = std::stoi(needValue(a));
        else {
            std::cerr << "Unknown flag: " << a << "\n";
            usage();
            return 1;
        }
    }

    // Validate required inputs
    if (args.outPath.empty()) {
        std::cerr << "Missing --out\n";
        usage();
        return 1;
    }
    if (args.imagePath.empty() && args.videoPath.empty()) {
        std::cerr << "Missing --image or --video\n";
        usage();
        return 1;
    }
    if (modeStr.empty()) {
        std::cerr << "Missing --mode\n";
        usage();
        return 1;
    }

    // Parse mode
    args.mode = parseMode(modeStr);

    // Validate numeric flags
    if (args.radius < 1) {
        std::cerr << "--radius must be >= 1\n";
        return 1;
    }
    if (args.threads < 1) args.threads = 1;

    try {
        Pipeline p;
        p.run(args);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
