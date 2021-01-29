/*
  MIT License

  Copyright (c) 2018-2019 HolyWu
  Copyright (c) 2019-2020 NaLan ZeYu

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
#include "waifu2x.hpp"

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
    const int srcStride = vsapi->getStride(src, 0) / static_cast<int>(sizeof(float));
    const int dstStride = vsapi->getStride(dst, 0) / static_cast<int>(sizeof(float));
    auto *             srcR = reinterpret_cast<const float *>(vsapi->getReadPtr(src, 0));
    auto *             srcG = reinterpret_cast<const float *>(vsapi->getReadPtr(src, 1));
    auto *             srcB = reinterpret_cast<const float *>(vsapi->getReadPtr(src, 2));
    auto * VS_RESTRICT dstR = reinterpret_cast<float *>(vsapi->getWritePtr(dst, 0));
    auto * VS_RESTRICT dstG = reinterpret_cast<float *>(vsapi->getWritePtr(dst, 1));
    auto * VS_RESTRICT dstB = reinterpret_cast<float *>(vsapi->getWritePtr(dst, 2));

    d->gpuSemaphore->wait();
    if (d->waifu2x->process(srcR, srcG, srcB, dstR, dstG, dstB, srcStride, dstStride)) {
        return false;
    }
    d->gpuSemaphore->signal();

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
            vsapi->setFilterError("Waifu2x-NCNN-Vulkan: Waifu2x::process error. Do you have enough VRAM?", frameCtx);
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

    int gpuId, noise, scale, model, tileSize, gpuThread, precision;
    std::string paramPath, modelPath;
    char const * err_prompt = nullptr;

    while (true) {
        int err;

        if (!isConstantFormat(&d.vi) || d.vi.format->colorFamily != cmRGB || d.vi.format->sampleType != stFloat || d.vi.format->bitsPerSample != 32) {
            err_prompt = "only constant RGB format and 32 bit float input supported";
            break;
        }

        gpuId = int64ToIntS(vsapi->propGetInt(in, "gpu_id", 0, &err));
        if (gpuId < 0 || gpuId >= ncnn::get_gpu_count()) {
            err_prompt = "invalid 'gpu_id'";
            break;
        }

        noise = int64ToIntS(vsapi->propGetInt(in, "noise", 0, &err));
        if (noise < -1 || noise > 3) {
            err_prompt = "'noise' must be -1, 0, 1, 2, or 3";
            break;
        }

        scale = int64ToIntS(vsapi->propGetInt(in, "scale", 0, &err));
        if (err)
            scale = 2;
        if (scale != 1 && scale != 2) {
            err_prompt = "'scale' must be 1 or 2";
            break;
        }

        model = int64ToIntS(vsapi->propGetInt(in, "model", 0, &err));
        if (model < 0 || model > 2) {
            err_prompt = "'model' must be 0, 1 or 2";
            break;
        }

        tileSize = int64ToIntS(vsapi->propGetInt(in, "tile_size", 0, &err));
        if (err)
            tileSize = 180;
        if (tileSize < 32) {
            err_prompt = "'tile_size' must be greater than or equal to 32";
            break;
        }
        if (tileSize % 4) {
            err_prompt = "'tile_size' must be multiple of 4";
            break;
        }

        precision = int64ToIntS(vsapi->propGetInt(in, "precision", 0, &err));
        if (err)
            precision = 16;
        if (precision != 16 && precision != 32) {
            err_prompt = "'precision' must be 16 or 32";
            break;
        }

        int customGpuThread = int64ToIntS(vsapi->propGetInt(in, "gpu_thread", 0, &err));
        if (customGpuThread > 0) {
            gpuThread = customGpuThread;
        } else {
            gpuThread = int64ToIntS(ncnn::get_gpu_info(gpuId).transfer_queue_count());
        }
        gpuThread = std::min(gpuThread, int64ToIntS(ncnn::get_gpu_info(gpuId).compute_queue_count()));

        if (scale == 1 && noise == -1) {
            err_prompt = "use 'noise=-1' and 'scale=1' at same time is useless";
            break;
        }

        if (scale == 1 && model != 2) {
            err_prompt = "only cunet model support 'scale=1'";
            break;
        }

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
        if (!pf.good() || !mf.good()) {
            err_prompt = "can't open model file";
            break;
        }

        break;
    }

    if (err_prompt) {
        {
            std::lock_guard<std::mutex> guard(g_lock);

            g_filter_instance_count--;
            if (g_filter_instance_count == 0)
                ncnn::destroy_gpu_instance();
        }
        vsapi->setError(out, (std::string{"Waifu2x-NCNN-Vulkan: "} + err_prompt).c_str());
        vsapi->freeNode(d.node);
        return;
    }

    {
        std::lock_guard<std::mutex> guard(g_lock);

        if (!g_gpu_semaphore.count(gpuId))
            g_gpu_semaphore.insert(std::pair<int, Semaphore *>(gpuId, new Semaphore(gpuThread)));

        d.gpuSemaphore = g_gpu_semaphore.at(gpuId);

        d.waifu2x = new Waifu2x(gpuId, precision);
        d.waifu2x->w = d.vi.width;
        d.waifu2x->h = d.vi.height;
        d.waifu2x->scale = scale;
        d.waifu2x->noise = noise;
        d.waifu2x->tilesize = tileSize;
        if (model == 2 && scale == 1)
            d.waifu2x->prepadding = 28;
        else if (model == 2)
            d.waifu2x->prepadding = 18;
        else
            d.waifu2x->prepadding = 7;

        d.waifu2x->load(paramPath, modelPath);
    }

    d.vi.width *= scale;
    d.vi.height *= scale;

    auto *data = new FilterData{ d };

    vsapi->createFilter(in, out, "Waifu2x", filterInit, filterGetFrame, filterFree, fmParallel, 0, data, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("net.nlzy.vsw2xnvk", "w2xnvk", "VapourSynth Waifu2x NCNN Vulkan Plugin", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("Waifu2x", "clip:clip;"
                            "noise:int:opt;"
                            "scale:int:opt;"
                            "model:int:opt;"
                            "tile_size:int:opt;"
                            "gpu_id:int:opt;"
                            "gpu_thread:int:opt;"
                            "precision:int:opt;"
                            , filterCreate, nullptr, plugin);
}
