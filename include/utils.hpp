#pragma once
#include <chrono>

/*
class Timer defines a new type
Timer() is a constructor: runs when you do Timer t;
start_ is private so only Timer controls it (good design)
ms() is const meaning "this function won't change the object"

*/

// Timer = a stopwatch object.
// We create one, then ask "how many ms passed?"

class Timer {
public:
    // Constructor runs automatically when you create the object
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    // reset() restarts the stopwatch
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    // ms() returns elapsed milliseconds since start/reset
    double ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = now - start_;
        return diff.count();

    }
private:
    // start_ stores the time when the timer began.
    std::chrono::high_resolution_clock::time_point start_;
};