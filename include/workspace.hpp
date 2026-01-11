#pragma once
#include <cstddef>
#include <vector>

/*
this struct stores reusable memory buffers for processing

Why do we want this?

Because allocating memory every frame is wasteful
we allocate once and keep reusing it
*/

struct CpuWorkspace {
    int w = 0;
    int h = 0;
    std::vector<int> tmp; // used by blur (stores horizontal sums)

    //Ensure tmp is big enough for an image of size (w x h)
    //If size changed, resize once; otherwise do nothing
    void ensureSize(int width, int height) {
        if (width == w && height == h) {
            return; // no change -> keep using exisiting memory
        }
        w = width;
        h = height;

        //Allocate exactly w*h ints
        tmp.assign((size_t)w * (size_t)h, 0);
    }
};