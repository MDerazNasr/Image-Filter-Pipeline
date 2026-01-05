//CLI + load + run + save + timing

#include <exception>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

// for printing current working directory (Mac/Linux)
#include <sys/syslimits.h>
#include <unistd.h>
#include <limits.h>

//For checking if output folder exists
#include <sys/stat.h>

#include "filters_cpu.hpp"
#include "utils.hpp"

//Print help for if it runs wrong
static void printUsage() {
    std::cout <<
    "Usage:\n"
    " ./pipeline --image <path> --out <path>\n"
    "Example:\n"
    "  ./pipeline --image data/input.jpg --out output/out_gray.png\n";
}
// Returns true if a directory exists at path
static bool dirExists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) return false;
    return (info.st_mode & S_IFDIR) != 0;
}

static std::string cwd() {
    char buf[PATH_MAX];
    if (getcwd(buf, sizeof(buf)) == nullptr) return "unknown";
    return std::string(buf);
}


int main(int argc, char** argv) {
    // argc = number of command-line arguments
    // argv = array of C-strings (char*) containing each argument

    if (argc < 5) {
        printUsage();
        return 1;
    }

    std::string imagePath;
    std::string outPath;

    // Very small CLI parser:
    // We scan flags like --image and read the next argument as its value

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];

        auto needValue = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc){
                throw std::runtime_error("Missin value after " + flag);
            }
            return argv[++i]; // move i forward and return that vaue
        };

        if (a == "--image") {
            imagePath = needValue(a);
        } else if (a == "--out") {
            outPath = needValue(a);
        } else {
            std::cerr << "Unknown flag: " << a << "\n";
            printUsage();
            return 1;
        }
    }

    try {
        std::cout << "Current working directory: " << cwd() << "\n";
        std::cout << "Input path: " << imagePath << "\n";
        std::cout << "Output path: " << outPath << "\n";

        // Quick check: does "output/" directory exist?
        // (If user passed output/out_gray.png, directory is output)
        auto slashPos = outPath.find_last_of('/');
        if (slashPos != std::string::npos) {
            std::string outDir = outPath.substr(0, slashPos);
            std::cout << "Output directory: " << outDir
                      << " (exists=" << (dirExists(outDir) ? "yes" : "no") << ")\n";
        }

        cv::Mat bgr = cv::imread(imagePath, cv::IMREAD_COLOR);

        // Print what OpenCV actually loaded
        std::cout << "Loaded image: "
                  << "empty=" << (bgr.empty() ? "yes" : "no")
                  << " size=" << bgr.cols << "x" << bgr.rows
                  << " type=" << bgr.type()
                  << "\n";

        if (bgr.empty()) {
            throw std::runtime_error("imread failed. Check the input path and file.");
        }

        cv::Mat gray;
        cv::Mat blurred;

        Timer tGray;
        grayscale_cpu(bgr, gray);
        double msGray = tGray.ms();

        Timer tBlur;
        box_blur_cpu(gray, blurred, 1); //radius = 1 -> 3x3
        double msBlur = tBlur.ms();

        // Save blurred image
        bool ok = cv::imwrite(outPath, blurred);
        if (!ok) {
            throw std::runtime_error("Failed to write output image");
        }
        
        std::cout << "Grayscale time: " << msGray << " ms\n";
        std::cout << "Blur time: " << msBlur << " ms\n";
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
/*
    what is the std:: thing for
    what is a parser
    i dont understand includes
    i dont get the function of hpp files
    what is the & at the end of variable names mean 
    what is start_
    what is cerr
    */

