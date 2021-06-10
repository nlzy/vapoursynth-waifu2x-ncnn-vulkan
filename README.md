# VapourSynth Waifu2x NCNN Vulkan Plugin

Waifu2x filter for VapourSynth, based on [waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan).

## Install

Download pre-built binaries and model files from [releases](https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan/releases). Uncompress and put into VapourSynth plugin folder.

## Usage

```
core.w2xnvk.Waifu2x(clip[, noise, scale, model, tile_size, gpu_id, gpu_thread, precision, tile_size_w, tile_size_h])
```

* clip: Input clip. Only 32-bit float RGB is supported.

* noise: Denoise level. (int -1/0/1/2/3, defualt=0)
  * -1 = none
  * 0 = low
  * 1 = medium
  * 2 = high
  * 3 = highest

* scale: Upscale ratio. (int 1/2, default=2)
  * 1 = no scaling, denoise only. upconv_7 doesn't support this mode.
  * 2 = upscale 2x.

* model: Model to use. (int 0/1/2, default=0)
  * 0 = upconv_7_anime_style_art_rgb
  * 1 = upconv_7_photo
  * 2 = cunet (For 2D artwork. Slow, but better quality.)

* tile_size: Tile size. Must be divisible by 4. Increasing this value may improve performance and take more VRAM. (int >=32, default=0 for auto choose)

* gpu_id: GPU device to use. (int >=0, default=0)

* gpu_thread: Number of threads that can simultaneously access GPU. (int >=1, default=0 for auto detect)

* precision: Floating-point precision. Single-precision (fp32) is slow but more precise in color. Default is half-precision (fp16). (int 16/32, default=16)

* tile_size_w / tile_size_h: Override width and height of tile_size.

## Performance Comparison

### AMD graphics card

* Ryzen 5 1600X + Radeon RX 580 2048SP
* vapoursynth-waifu2x-ncnn-vulkan: `core.w2xnvk.Waifu2x(last, noise=0, scale=2)`
* vapoursynth-waifu2x-w2xc: `core.w2xc.Waifu2x(last, noise=0, scale=2)`

|                                 |  540p -> 1080p |  720p -> 2K | 1080p -> 4K |
|---------------------------------|----------------|-------------|-------------|
| vapoursynth-waifu2x-w2xc        |      0.744 fps |   0.435 fps |   0.199 fps |
| vapoursynth-waifu2x-ncnn-vulkan |      3.203 fps |   1.788 fps |   0.795 fps |

### NVIDIA graphics card

* Ryzen 3 1400 + GeForce GTX 1050 Ti
* vapoursynth-waifu2x-ncnn-vulkan: `core.w2xnvk.Waifu2x(last, noise=0, scale=2, model=0)`
* vapoursynth-waifu2x-caffe: `core.caffe.Waifu2x(last, noise=0, scale=2, model=3)`

|                                 |  540p -> 1080p |  720p -> 2K | 1080p -> 4K |
|---------------------------------|----------------|-------------|-------------|
| vapoursynth-waifu2x-caffe       |      1.674 fps |   1.085 fps |   0.477 fps |
| vapoursynth-waifu2x-ncnn-vulkan |      1.880 fps |   1.034 fps |   0.471 fps |

## Build

### Arch Linux

```bash
# Install Vulkan SDK and something else:
sudo pacman -S vapoursynth glslang vulkan-icd-loader vulkan-headers

# Clone repository and submodule
git clone https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan.git
cd vapoursynth-waifu2x-ncnn-vulkan
git submodule update --init --recursive
mkdir build
cd build

# build
cmake ..
cmake --build . -j 4
```

### Windows

Install [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).

Open `Git Bash`, clone repository and submodule:

```bash
git clone https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan.git
cd vapoursynth-waifu2x-ncnn-vulkan
git submodule update --init --recursive
mkdir build
```

Open `Start Menu` -> `Visual Studio 2019` -> `x64 Native Tools Command Prompt for VS 2019`, then build:

```
cd X:\path_to_vapoursynth-waifu2x-ncnn-vulkan\build
cmake -G "NMake Makefiles" -DVAPOURSYNTH_HEADER_DIR=X:\path_to_vapoursynth\sdk\include\vapoursynth ..
cmake --build .
```

Note: If you are using VapourSynth "Portable" version, use `-DVAPOURSYNTH_HEADER_DIR=X:\path_to_vapoursynth\sdk\include` instead.
