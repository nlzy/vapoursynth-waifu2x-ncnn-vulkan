#include <mutex>
#include <fstream>

#include "gpu.h"
#include "waifu2x.h"

#include "VSHelper.h"

typedef struct {
    VSNodeRef *node;
    VSVideoInfo vi;
    uint8_t *srcInterleaved, *dstInterleaved;
    Waifu2x *waifu2x;
} FilterData;

static std::mutex g_lock{};
static int g_filter_instance_count = 0;

static bool filter(const VSFrameRef *src, VSFrameRef *dst, FilterData * const VS_RESTRICT d, const VSAPI *vsapi) noexcept {
    const int width = vsapi->getFrameWidth(src, 0);
    const int height = vsapi->getFrameHeight(src, 0);
    const int srcStride = vsapi->getStride(src, 0) / static_cast<int>(sizeof(uint8_t));
    const int dstStride = vsapi->getStride(dst, 0) / static_cast<int>(sizeof(uint8_t));
    auto *srcR = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(src, 0));
    auto *srcG = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(src, 1));
    auto *srcB = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(src, 2));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int pos = (width * y + x) * 3;
            d->srcInterleaved[pos + 0] = srcR[x];
            d->srcInterleaved[pos + 1] = srcG[x];
            d->srcInterleaved[pos + 2] = srcB[x];
        }
        srcR += srcStride;
        srcG += srcStride;
        srcB += srcStride;
    }

    {
        std::lock_guard<std::mutex> guard(g_lock);

        ncnn::Mat inImage = ncnn::Mat(width, height, static_cast<void *>(d->srcInterleaved), static_cast<size_t>(3), 3);
        ncnn::Mat outImage = ncnn::Mat(d->vi.width, d->vi.height, static_cast<void *>(d->dstInterleaved),static_cast<size_t>(3), 3);
        d->waifu2x->process(inImage, outImage);
    }

    auto * VS_RESTRICT dstR = reinterpret_cast<uint8_t *>(vsapi->getWritePtr(dst, 0));
    auto * VS_RESTRICT dstG = reinterpret_cast<uint8_t *>(vsapi->getWritePtr(dst, 1));
    auto * VS_RESTRICT dstB = reinterpret_cast<uint8_t *>(vsapi->getWritePtr(dst, 2));

    for (int y = 0; y < d->vi.height; y++) {
        for (int x = 0; x < d->vi.width; x++) {
            const int pos = (d->vi.width * y + x) * 3;
            dstR[x] = d->dstInterleaved[pos + 0];
            dstG[x] = d->dstInterleaved[pos + 1];
            dstB[x] = d->dstInterleaved[pos + 2];
        }
        dstR += dstStride;
        dstG += dstStride;
        dstB += dstStride;
    }

    return true;
}

static void VS_CC filterInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    auto *d = static_cast<FilterData *>(*instanceData);
    vsapi->setVideoInfo(&d->vi, 1, node);
}

static const VSFrameRef *VS_CC filterGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    auto *d = static_cast<FilterData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->node, frameCtx);

        VSFrameRef * dst = vsapi->newVideoFrame(d->vi.format, d->vi.width, d->vi.height, src, core);
        filter(src, dst, d, vsapi);

        vsapi->freeFrame(src);
        return dst;
    }

    return 0;
}

static void VS_CC filterFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    auto *d = static_cast<FilterData *>(instanceData);
    vsapi->freeNode(d->node);
    delete[] d->srcInterleaved;
    delete[] d->dstInterleaved;
    delete d->waifu2x;
    delete d;

    std::lock_guard<std::mutex> guard(g_lock);
    g_filter_instance_count--;
    if (g_filter_instance_count == 0) {
        ncnn::destroy_gpu_instance();
    }
}

static void VS_CC filterCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    FilterData d{};

    d.node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d.vi = *vsapi->getVideoInfo(d.node);

    {
        std::lock_guard<std::mutex> guard(g_lock);

        if (g_filter_instance_count == 0) {
            ncnn::create_gpu_instance();
        }
        g_filter_instance_count++;
    }

    try {
        if (!isConstantFormat(&d.vi) || d.vi.format->colorFamily != cmRGB || d.vi.format->sampleType != stInteger || d.vi.format->bitsPerSample != 8)
            throw std::string{ "only constant RGB format and 8 bit integer input supported" };

        // init waifu2x instance
        int err;
        const int gpuId = int64ToIntS(vsapi->propGetInt(in, "gpu_id", 0, &err));
        auto waifu2x = new Waifu2x(gpuId);
        d.waifu2x = waifu2x;

        // get and set filter default parameters
        waifu2x->noise = int64ToIntS(vsapi->propGetInt(in, "noise", 0, &err));
        if (err)
            waifu2x->noise = 0;

        waifu2x->scale = int64ToIntS(vsapi->propGetInt(in, "scale", 0, &err));
        if (err)
            waifu2x->scale = 2;

        waifu2x->tilesize = int64ToIntS(vsapi->propGetInt(in, "tile_size", 0, &err));
        if (err)
            waifu2x->tilesize = 240;

        int prePadding = 7;
        waifu2x->prepadding = prePadding;

        d.vi.width *= waifu2x->scale;
        d.vi.height *= waifu2x->scale;

        // check parameters value
        if (waifu2x->noise < -1 || waifu2x->noise > 3)
            throw std::string{ "noise must be -1, 0, 1, 2, or 3" };

        if (waifu2x->scale != 1 && waifu2x->scale != 2)
            throw std::string{ "scale must be 1 or 2" };

        if (waifu2x->tilesize < 32)
            throw std::string{ "tile size must be greater than or equal to 32" };

        // alloc buffer
        d.srcInterleaved = new (std::nothrow) uint8_t[d.vi.width * d.vi.height * 3 / waifu2x->scale / waifu2x->scale];
        d.dstInterleaved = new (std::nothrow) uint8_t[d.vi.width * d.vi.height * 3];
        if (!d.srcInterleaved || !d.dstInterleaved)
            throw std::string{ "malloc failure (srcInterleaved/dstInterleaved)" };

        // set model path
        const std::string pluginDir{ vsapi->getPluginPath(vsapi->getPluginById("net.nlzy.vsw2xnvk", core)) };
        const std::string modelsDir{ pluginDir.substr(0, pluginDir.find_last_of('/')) + "/models/" };
        std::string modelName;
        if (waifu2x->noise == -1) {
            modelName = "scale2.0x_model";
        } else if (waifu2x->scale == 1) {
            modelName = "noise" + std::to_string(waifu2x->noise) + "_model";
        } else{
            modelName = "noise" + std::to_string(waifu2x->noise) + "_scale2.0x_model";
        }
        const std::string paramPath = modelsDir + modelName + ".param";
        const std::string modelPath = modelsDir + modelName + ".bin";

        // check model file readable
        std::ifstream pf(paramPath);
        std::ifstream mf(modelPath);
        if (!pf.good() || !mf.good())
            throw std::string{ "can't open model file" };

        std::lock_guard<std::mutex> guard(g_lock);
        waifu2x->load(paramPath, modelPath);
    } catch (std::string &err) {
        vsapi->setError(out, ("Waifu2x-NcnnVulkan: " + err).c_str());
        vsapi->freeNode(d.node);
        return;
    }

    auto *data = new FilterData{ d };

    vsapi->createFilter(in, out, "Filter", filterInit, filterGetFrame, filterFree, fmParallel, 0, data, core);
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("net.nlzy.vsw2xnvk", "w2xnvk", "VapourSynth Waifu2x NCNN Vulkan Plugin", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Waifu2x", "clip:clip;"
                            "noise:int:opt;"
                            "scale:int:opt;"
                            "tile_size:int:opt;"
                            "gpu_id:int:opt;"
                            , filterCreate, 0, plugin);
}
