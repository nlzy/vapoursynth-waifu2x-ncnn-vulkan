# VapourSynth Waifu2x NCNN Vulkan Plugin

Waifu2x filter for VapourSynth, based on [waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan).

## Install

Download pre-built binaries and model files from [release page](https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan/releases). Uncompress and put into VapourSynth plugin folder.

## Usage

```
core.w2xnvk.Waifu2x(clip[, noise, scale, model, tile_size, gpu_id, gpu_thread])
```

* clip: input clip. Only 8-bit RGB is supported.

* noise: denoise level (int -1/0/1/2/3, defualt=0)
  * -1 = none
  * 0 = low
  * 1 = medium
  * 2 = high
  * 3 = highest

* scale: upscale ratio (int 1/2, default=2)

* model: model to use (int 0/1/2, default=0)
  * 0 = upconv_7_anime_style_art_rgb
  * 1 = upconv_7_photo
  * 2 = cunet (slow, but better quality)

* tile_size: tile size (int >=32, default=180)

* gpu_id: gpu device to use (int >=0, default=0)

* gpu_thread: Number of threads that can simultaneously access GPU (int >=1, default=0 for auto detect)

## Performance Comparison

Hardware: Ryzen 5 1600X + Radeon RX 580 2048SP

Parameters:
* vapoursynth-waifu2x-ncnn-vulkan: `core.w2xnvk.Waifu2x(last, noise=0, scale=2, gpu_thread=3)`
* vapoursynth-waifu2x-w2xc: `core.w2xc.Waifu2x(last, noise=0, scale=2)`

|                                 |  540p -> 1080p |  720p -> 2K | 1080p -> 4K |
|---------------------------------|----------------|-------------|-------------|
| vapoursynth-waifu2x-ncnn-vulkan |      3.282 fps |   1.848 fps |   0.822 fps |
| vapoursynth-waifu2x-w2xc        |      0.744 fps |   0.435 fps |   0.199 fps |
