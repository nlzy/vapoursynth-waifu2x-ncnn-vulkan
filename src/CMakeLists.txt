project(vapoursynth-waifu2x-ncnn-vulkan)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_BUILD_TYPE Release)

find_package(ncnn REQUIRED)
find_package(w2xnvk REQUIRED)

add_library(vsw2xnvk SHARED vsw2xnvk.cpp)
target_include_directories(vsw2xnvk PRIVATE "/usr/include/vapoursynth")
target_link_libraries(vsw2xnvk w2xnvk)
