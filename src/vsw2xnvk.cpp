#include <mutex>

#include "gpu.h"
#include "waifu2x.h"

#include "VSHelper.h"

static std::mutex g_lock{};
static int g_filter_instance_count = 0;

static int conv_rgb24(uint8_t *src, uint8_t *dst, int src_w, int src_h) {
    std::lock_guard<std::mutex> guard(g_lock);
    int noise = 0;
    int scale = 2;
    int tileSize = 120;
    int gpuId = 0;

    int prePadding = 7;

    std::string paramPath = "noise0_scale2.0x_model.param";
    std::string modelPath = "noise0_scale2.0x_model.bin";

    Waifu2x waifu2x(gpuId);
    waifu2x.load(paramPath, modelPath);

    waifu2x.noise = noise;
    waifu2x.scale = scale;
    waifu2x.tilesize = tileSize;
    waifu2x.prepadding = prePadding;

    // load
    ncnn::Mat inImage = ncnn::Mat(src_w, src_h, (void*)src, (size_t)3, 3);
    ncnn::Mat outImage = ncnn::Mat(src_w * scale, src_h * scale, (void *)dst,(size_t)3u, 3);

    // proc
    waifu2x.process(inImage, outImage);

    return 0;
}

typedef struct {
    VSNodeRef *node;
    VSVideoInfo vi;
    uint8_t * srcInterleaved, * dstInterleaved;
} FilterData;

static bool filter(const VSFrameRef * src, VSFrameRef * dst, FilterData * const VS_RESTRICT d, const VSAPI * vsapi) noexcept {
    const int width = vsapi->getFrameWidth(src, 0);
    const int height = vsapi->getFrameHeight(src, 0);
    const int srcStride = vsapi->getStride(src, 0) / (int)sizeof(uint8_t);
    const int dstStride = vsapi->getStride(dst, 0) / (int)sizeof(uint8_t);
    const uint8_t * srcpR = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(src, 0));
    const uint8_t * srcpG = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(src, 1));
    const uint8_t * srcpB = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(src, 2));
    uint8_t * VS_RESTRICT dstpR = reinterpret_cast<uint8_t *>(vsapi->getWritePtr(dst, 0));
    uint8_t * VS_RESTRICT dstpG = reinterpret_cast<uint8_t *>(vsapi->getWritePtr(dst, 1));
    uint8_t * VS_RESTRICT dstpB = reinterpret_cast<uint8_t *>(vsapi->getWritePtr(dst, 2));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int pos = (width * y + x) * 3;
            d->srcInterleaved[pos + 0] = srcpR[x];
            d->srcInterleaved[pos + 1] = srcpG[x];
            d->srcInterleaved[pos + 2] = srcpB[x];
        }

        srcpR += srcStride;
        srcpG += srcStride;
        srcpB += srcStride;
    }

    conv_rgb24(d->srcInterleaved, d->dstInterleaved, width, height);

    for (int y = 0; y < d->vi.height; y++) {
        for (int x = 0; x < d->vi.width; x++) {
            const int pos = (d->vi.width * y + x) * 3;
            dstpR[x] = d->dstInterleaved[pos + 0];
            dstpG[x] = d->dstInterleaved[pos + 1];
            dstpB[x] = d->dstInterleaved[pos + 2];
        }

        dstpR += dstStride;
        dstpG += dstStride;
        dstpB += dstStride;
    }

    return true;
}

static void VS_CC filterInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    FilterData *d = static_cast<FilterData *>(*instanceData);
    vsapi->setVideoInfo(&d->vi, 1, node);
}

static const VSFrameRef *VS_CC filterGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    FilterData *d = static_cast<FilterData *>(*instanceData);

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
    FilterData *d = static_cast<FilterData *>(instanceData);
    vsapi->freeNode(d->node);
    delete[] d->srcInterleaved;
    delete[] d->dstInterleaved;
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

    d.vi.width *= 2;
    d.vi.height *= 2;

    d.srcInterleaved = new (std::nothrow) uint8_t[vsapi->getVideoInfo(d.node)->width * vsapi->getVideoInfo(d.node)->height * 3];
    d.dstInterleaved = new (std::nothrow) uint8_t[d.vi.width * d.vi.height * 3];

    FilterData *data = new FilterData{ d };

    std::lock_guard<std::mutex> guard(g_lock);
    if (g_filter_instance_count == 0) {
        ncnn::create_gpu_instance();
    }
    g_filter_instance_count++;

    vsapi->createFilter(in, out, "Filter", filterInit, filterGetFrame, filterFree, fmParallel, 0, data, core);
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.example.filter", "filter", "VapourSynth Filter Skeleton", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Filter", "clip:clip;", filterCreate, 0, plugin);
}
