#ifndef WAIFU2X_HPP
#define WAIFU2X_HPP

#define RGB_CHANNELS 3

#include <string>
#include "net.h"
#include "gpu.h"

class Waifu2x
{
public:
    Waifu2x(int gpuid);
    ~Waifu2x();
    int load(const std::string& parampath, const std::string& modelpath);
    int process(const float *srcR, const float *srcG, const float *srcB, float *dstR, float *dstG, float *dstB, int srcStride, int dstStride) const;
public:
    int w;
    int h;
    int noise;
    int scale;
    int tilesize;
    int prepadding;
private:
    ncnn::Net net;
    ncnn::Pipeline* waifu2x_preproc;
    ncnn::Pipeline* waifu2x_postproc;
};

#endif
