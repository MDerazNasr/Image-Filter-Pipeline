cmake_minimum_required(VERSION 3.18)
project(image_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(pipeline
    src/main.cpp
    src/pipeline.cpp
    src/filters_cpu.cpp
)

target_include_directories(pipeline PRIVATE include)
target_link_libraries(pipeline PRIVATE ${OpenCV_LIBS})
