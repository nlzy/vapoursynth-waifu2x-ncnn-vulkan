/*
  MIT License

  Copyright (c) 2018-2019 HolyWu
  Copyright (c) 2019 NaLan ZeYu

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#include <map>
#include <mutex>
#include <fstream>
#include <condition_variable>
#include <algorithm>

#include "gpu.h"
#include "waifu2x.h"

#include "VSHelper.h"

class Semaphore {
private:
    int val;
    std::mutex mtx;
    std::condition_variable cv;
public:
    explicit Semaphore(int init_value) : val(init_value) {
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        while (val <= 0) {
            cv.wait(lock);
        }
        val--;
    }
    void signal() {
        std::lock_guard<std::mutex> guard(mtx);
        val++;
        cv.notify_one();
    }
};

typedef struct {
    VSNodeRef *node;
    VSVideoInfo vi;
    Waifu2x *waifu2x;
    Semaphore *gpuSemaphore;
} FilterData;

static std::mutex g_lock{};
static int g_filter_instance_count = 0;
static std::map<int, Semaphore *> g_gpu_semaphore;

static bool filter(const VSFrameRef *src, VSFrameRef *dst, FilterData * const VS_RESTRICT d, const VSAPI *vsapi) noexcept {
    const int width = vsapi->getFrameWidth(src, 0);
    const int height = vsapi->getFrameHeight(src, 0);
    const int srcStride = vsapi->getStride(src, 0) / static_cast<int>(sizeof(uint8_t));
    const int dstStride = vsapi->getStride(dst, 0) / static_cast<int>(sizeof(uint8_t));
    auto *srcR = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(src, 0));
    auto *srcG = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(src, 1));
    auto *srcB = reinterpret_cast<const uint8_t *>(vsapi->getReadPtr(src, 2));

    auto *srcInterleaved = new uint8_t[width * height * 3];
    auto *dstInterleaved = new uint8_t[d->vi.width * d->vi.height * 3];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int pos = (width * y + x) * 3;
            srcInterleaved[pos + 0] = srcR[x];
            srcInterleaved[pos + 1] = srcG[x];
            srcInterleaved[pos + 2] = srcB[x];
        }
        srcR += srcStride;
        srcG += srcStride;
        srcB += srcStride;
    }

    d->gpuSemaphore->wait();
    ncnn::Mat inImage = ncnn::Mat(width, height, static_cast<void *>(srcInterleaved), static_cast<size_t>(3), 3);
    ncnn::Mat outImage = ncnn::Mat(d->vi.width, d->vi.height, static_cast<void *>(dstInterleaved),static_cast<size_t>(3), 3);
    if (d->waifu2x->process(inImage, outImage)) {
        delete[] srcInterleaved;
        delete[] dstInterleaved;
        return false;
    }
    d->gpuSemaphore->signal();

    auto * VS_RESTRICT dstR = reinterpret_cast<uint8_t *>(vsapi->getWritePtr(dst, 0));
    auto * VS_RESTRICT dstG = reinterpret_cast<uint8_t *>(vsapi->getWritePtr(dst, 1));
    auto * VS_RESTRICT dstB = reinterpret_cast<uint8_t *>(vsapi->getWritePtr(dst, 2));

    for (int y = 0; y < d->vi.height; y++) {
        for (int x = 0; x < d->vi.width; x++) {
            const int pos = (d->vi.width * y + x) * 3;
            dstR[x] = dstInterleaved[pos + 0];
            dstG[x] = dstInterleaved[pos + 1];
            dstB[x] = dstInterleaved[pos + 2];
        }
        dstR += dstStride;
        dstG += dstStride;
        dstB += dstStride;
    }

    delete[] srcInterleaved;
    delete[] dstInterleaved;

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

        if (!filter(src, dst, d, vsapi)) {
            vsapi->setFilterError("Waifu2x-NCNN-Vulkan: Waifu2x::process error. Do you have enough GPU memory?", frameCtx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            return nullptr;
        }

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC filterFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    auto *d = static_cast<FilterData *>(instanceData);
    vsapi->freeNode(d->node);
    delete d->waifu2x;
    delete d;

    std::lock_guard<std::mutex> guard(g_lock);
    g_filter_instance_count--;
    if (g_filter_instance_count == 0) {
        ncnn::destroy_gpu_instance();
        for (auto pair : g_gpu_semaphore) {
            delete pair.second;
        }
        g_gpu_semaphore.clear();
    }
}

static void VS_CC filterCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    FilterData d{};

    d.node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d.vi = *vsapi->getVideoInfo(d.node);

    {
        std::lock_guard<std::mutex> guard(g_lock);

        if (g_filter_instance_count == 0)
            ncnn::create_gpu_instance();

        g_filter_instance_count++;
    }

    int gpuId, noise, scale, model, tileSize, gpuThread;
    std::string paramPath, modelPath;
    try {
        int err;

        if (!isConstantFormat(&d.vi) || d.vi.format->colorFamily != cmRGB || d.vi.format->sampleType != stInteger || d.vi.format->bitsPerSample != 8)
            throw std::string{ "only constant RGB format and 8 bit integer input supported" };

        gpuId = int64ToIntS(vsapi->propGetInt(in, "gpu_id", 0, &err));
        if (gpuId < 0 || gpuId >= ncnn::get_gpu_count())
            throw std::string{"invaild gpu_id"};

        noise = int64ToIntS(vsapi->propGetInt(in, "noise", 0, &err));
        if (noise < -1 || noise > 3)
            throw std::string{ "noise must be -1, 0, 1, 2, or 3" };

        scale = int64ToIntS(vsapi->propGetInt(in, "scale", 0, &err));
        if (err)
            scale = 2;
        if (scale != 1 && scale != 2)
            throw std::string{ "scale must be 1 or 2" };

        model = int64ToIntS(vsapi->propGetInt(in, "model", 0, &err));
        if (model < 0 || model > 2)
            throw std::string{ "model must be 0, 1, 2" };

        tileSize = int64ToIntS(vsapi->propGetInt(in, "tile_size", 0, &err));
        if (err)
            tileSize = 180;
        if (tileSize < 32)
            throw std::string{ "tile size must be greater than or equal to 32" };

        int customGpuThread = int64ToIntS(vsapi->propGetInt(in, "gpu_thread", 0, &err));
        if (customGpuThread > 0) {
            gpuThread = customGpuThread;
        } else {
            gpuThread = int64ToIntS(ncnn::get_gpu_info(gpuId).transfer_queue_count);
        }
        gpuThread = std::min(gpuThread, int64ToIntS(ncnn::get_gpu_info(gpuId).compute_queue_count));

        if (scale == 1 && noise == -1)
            throw std::string{ "noise can't be -1 when scale=1" };

        if (scale == 1 && model != 2)
            throw std::string{ "only cunet model support scale=1" };

        // set model path
        const std::string pluginFilePath{ vsapi->getPluginPath(vsapi->getPluginById("net.nlzy.vsw2xnvk", core)) };
        const std::string pluginDir = pluginFilePath.substr(0, pluginFilePath.find_last_of('/'));

        std::string modelsDir;
        if (model == 0)
            modelsDir += pluginDir + "/models-upconv_7_anime_style_art_rgb/";
        else if (model == 1)
            modelsDir += pluginDir + "/models-upconv_7_photo/";
        else
            modelsDir += pluginDir + "/models-cunet/";

        std::string modelName;
        if (noise == -1)
            modelName = "scale2.0x_model";
        else if (scale == 1)
            modelName = "noise" + std::to_string(noise) + "_model";
        else
            modelName = "noise" + std::to_string(noise) + "_scale2.0x_model";

        paramPath = modelsDir + modelName + ".param";
        modelPath = modelsDir + modelName + ".bin";

        // check model file readable
        std::ifstream pf(paramPath);
        std::ifstream mf(modelPath);
        if (!pf.good() || !mf.good())
            throw std::string{ "can't open model file" };

    } catch (std::string &err) {
        {
            std::lock_guard<std::mutex> guard(g_lock);

            g_filter_instance_count--;
            if (g_filter_instance_count == 0)
                ncnn::destroy_gpu_instance();
        }
        vsapi->setError(out, ("Waifu2x-NCNN-Vulkan: " + err).c_str());
        vsapi->freeNode(d.node);
        return;
    }

    d.vi.width *= scale;
    d.vi.height *= scale;

    {
        std::lock_guard<std::mutex> guard(g_lock);

        if (!g_gpu_semaphore.count(gpuId))
            g_gpu_semaphore.insert(std::pair<int, Semaphore *>(0, new Semaphore(gpuThread)));

        d.gpuSemaphore = g_gpu_semaphore.at(gpuId);

        d.waifu2x = new Waifu2x(gpuId);
        d.waifu2x->scale = scale;
        d.waifu2x->noise = noise;
        d.waifu2x->tilesize = tileSize;
        if (model == 2 && scale == 1)
            d.waifu2x->prepadding = 28;
        else if (model == 2)
            d.waifu2x->prepadding = 18;
        else
            d.waifu2x->prepadding = 7;

#if _WIN32
        std::wstring pp(paramPath.begin(), paramPath.end());
        std::wstring mp(modelPath.begin(), modelPath.end());
        d.waifu2x->load(pp, mp);
#else
        d.waifu2x->load(paramPath, modelPath);
#endif
    }

    auto *data = new FilterData{ d };

    vsapi->createFilter(in, out, "Waifu2x", filterInit, filterGetFrame, filterFree, fmParallel, 0, data, core);
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("net.nlzy.vsw2xnvk", "w2xnvk", "VapourSynth Waifu2x NCNN Vulkan Plugin", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Waifu2x", "clip:clip;"
                            "noise:int:opt;"
                            "scale:int:opt;"
                            "model:int:opt;"
                            "tile_size:int:opt;"
                            "gpu_id:int:opt;"
                            "gpu_thread:int:opt;"
                            , filterCreate, nullptr, plugin);
}
