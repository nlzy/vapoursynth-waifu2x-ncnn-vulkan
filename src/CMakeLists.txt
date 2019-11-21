project(vapoursynth-waifu2x-ncnn-vulkan)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_BUILD_TYPE Release)

# change these path to yours
set(ncnn_DIR "/home/nlzy/OpenSource/ncnn/build-host-gcc-linux/ncnn/lib/cmake/ncnn")
set(w2xnvk_DIR "/home/nlzy/OpenSource/waifu2x-ncnn-vulkan/src/build/lib/w2xnvk")
find_package(ncnn REQUIRED)
find_package(w2xnvk REQUIRED)

add_library(vsw2xnvk SHARED vsw2xnvk.cpp)
target_include_directories(vsw2xnvk PRIVATE "/usr/include/vapoursynth")
target_link_libraries(vsw2xnvk w2xnvk)
