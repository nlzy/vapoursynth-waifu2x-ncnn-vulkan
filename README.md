# VapourSynth Waifu2x NCNN Vulkan Plugin

Waifu2x filter for VapourSynth, based on [waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan).

## DEPRECATED

**This plugin has been deprecated.**

I recommend using [vs-mlrt](https://github.com/AmusementClub/vs-mlrt) instead, which offers broader model support, provides more inference framework options (including NCNN Vulkan), and is actively maintained.

## Install

Download pre-built binaries and model files from [releases](https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan/releases). Uncompress and put into VapourSynth plugin folder.

## Usage

```
core.w2xnvk.Waifu2x(clip[, noise, scale, model, tile_size, gpu_id, gpu_thread, precision, tile_size_w, tile_size_h, tta])
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

* tta: TTA (test-time augmentation) mode. (bool True/False, default=False)

## Build

### Linux

Install dependencies:

```bash
# Arch Linux
sudo pacman -S vapoursynth glslang vulkan-icd-loader vulkan-headers
# Fedora
sudo dnf install vapoursynth-devel glslang vulkan-loader-devel vulkan-headers
```

Get source code and build:

```bash
# clone repository and submodule
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
