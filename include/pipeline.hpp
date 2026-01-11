#pragma once
#include <string>

enum class Mode {
    CPU_SINGLE,
    CPU_MT,
    GPU // placeholder for future
};

struct Args {
    std::string imagePath;
    std::string videoPath;
    std::string outPath;
    Mode mode = Mode::CPU_SINGLE;
    int threads = 4;
    int radius = 1;
};

class Pipeline {
public:
    void run(const Args& args);

private:
    void runImage(const Args& args);
    void runVideo(const Args& args);
};
