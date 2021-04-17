/*
  MIT License

  Copyright (c) 2019 nihui
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

#include <algorithm>
#include <vector>
#include "waifu2x.hpp"

#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))
#define ALIGN_CEIL(a, b) (((a) + (b) - 1) / (b) * (b))

static const uint32_t waifu2x_preproc_spv_data[] = {
    #include "waifu2x_preproc.spv.hex.h"
};
static const uint32_t waifu2x_preproc_fp16s_spv_data[] = {
    #include "waifu2x_preproc_fp16s.spv.hex.h"
};
static const uint32_t waifu2x_preproc_int8s_spv_data[] = {
    #include "waifu2x_preproc_int8s.spv.hex.h"
};
static const uint32_t waifu2x_postproc_spv_data[] = {
    #include "waifu2x_postproc.spv.hex.h"
};
static const uint32_t waifu2x_postproc_fp16s_spv_data[] = {
    #include "waifu2x_postproc_fp16s.spv.hex.h"
};
static const uint32_t waifu2x_postproc_int8s_spv_data[] = {
    #include "waifu2x_postproc_int8s.spv.hex.h"
};

Waifu2x::Waifu2x(int width, int height, int scale, int tilesize, int gpuid, int gputhread,
    int precision, int prepadding, const std::string& parampath, const std::string& modelpath) :
    width(width), height(height), scale(scale), tilesize(tilesize), prepadding(prepadding), semaphore(gputhread)
{
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_packed = precision == 16;
    net.opt.use_fp16_storage = precision == 16;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_int8_storage = false;
    net.opt.use_int8_arithmetic = false;

    net.set_vulkan_device(gpuid);
    net.load_param(parampath.c_str());
    net.load_model(modelpath.c_str());

    // initialize preprocess and postprocess pipeline
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        waifu2x_preproc = new ncnn::Pipeline(net.vulkan_device());
        waifu2x_preproc->set_optimal_local_size_xyz(32, 32, 3);
        if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
            waifu2x_preproc->create(waifu2x_preproc_int8s_spv_data, sizeof(waifu2x_preproc_int8s_spv_data), specializations);
        else if (net.opt.use_fp16_storage)
            waifu2x_preproc->create(waifu2x_preproc_fp16s_spv_data, sizeof(waifu2x_preproc_fp16s_spv_data), specializations);
        else
            waifu2x_preproc->create(waifu2x_preproc_spv_data, sizeof(waifu2x_preproc_spv_data), specializations);

        waifu2x_postproc = new ncnn::Pipeline(net.vulkan_device());
        waifu2x_postproc->set_optimal_local_size_xyz(32, 32, 3);
        if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
            waifu2x_postproc->create(waifu2x_postproc_int8s_spv_data, sizeof(waifu2x_postproc_int8s_spv_data), specializations);
        else if (net.opt.use_fp16_storage)
            waifu2x_postproc->create(waifu2x_postproc_fp16s_spv_data, sizeof(waifu2x_postproc_fp16s_spv_data), specializations);
        else
            waifu2x_postproc->create(waifu2x_postproc_spv_data, sizeof(waifu2x_postproc_spv_data), specializations);
    }
}

Waifu2x::~Waifu2x() {
    delete waifu2x_preproc;
    delete waifu2x_postproc;
}

int Waifu2x::process(const float *srcR, const float *srcG, const float *srcB,
                     float *dstR, float *dstG, float *dstB,
                     const ptrdiff_t srcStride, const ptrdiff_t dstStride) const {
    semaphore.wait();

    ncnn::VkAllocator* blob_vkallocator = net.vulkan_device()->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = net.vulkan_device()->acquire_staging_allocator();
    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    const int xtiles = DIV_CEIL(width, tilesize);
    const int ytiles = DIV_CEIL(height, tilesize);
    const int prepadding_right = prepadding + ALIGN_CEIL(width, 4 / scale) - width;
    const int prepadding_bottom = prepadding + ALIGN_CEIL(height, 4 / scale) - height;

    for (int yi = 0; yi < ytiles; yi++) {
        ncnn::VkCompute cmd(net.vulkan_device());

        // upload
        ncnn::VkMat in_gpu;
        {
            const int in_tile_y0 = std::max(yi * tilesize - prepadding, 0);
            const int in_tile_y1 = std::min((yi + 1) * tilesize + (yi == ytiles - 1 ? prepadding_bottom : prepadding), height);
            const int in_tile_w = width;
            const int in_tile_h = in_tile_y1 - in_tile_y0;

            ncnn::Mat tmp;
            tmp.create(in_tile_w, in_tile_h, RGB_CHANNELS, sizeof(float));
            float* in_tile_r = tmp.channel(0);
            float* in_tile_g = tmp.channel(1);
            float* in_tile_b = tmp.channel(2);
            const float* sr = srcR + in_tile_y0 * srcStride;
            const float* sg = srcG + in_tile_y0 * srcStride;
            const float* sb = srcB + in_tile_y0 * srcStride;
            for (int y = 0; y < in_tile_h; y++) {
                for (int x = 0; x < in_tile_w; x++) {
                    in_tile_r[in_tile_w * y + x] = sr[srcStride * y + x] * 255.F;
                    in_tile_g[in_tile_w * y + x] = sg[srcStride * y + x] * 255.F;
                    in_tile_b[in_tile_w * y + x] = sb[srcStride * y + x] * 255.F;
                }
            }

            cmd.record_clone(tmp, in_gpu, opt);

            if (xtiles > 1) {
                if (cmd.submit_and_wait()) {
                    return ERROR_UPLOAD;
                }
                cmd.reset();
            }
        }

        const int out_tile_y0 = std::max(yi * tilesize, 0);
        const int out_tile_y1 = std::min((yi + 1) * tilesize, height);
        ncnn::VkMat out_gpu;
        out_gpu.create(width * scale, (out_tile_y1 - out_tile_y0) * scale, RGB_CHANNELS, sizeof(float), blob_vkallocator);
        for (int xi = 0; xi < xtiles; xi++) {
            // preproc
            ncnn::VkMat in_tile_gpu;
            {
                // crop tile
                int tile_x0 = xi * tilesize;
                int tile_x1 = std::min((xi + 1) * tilesize, width) + prepadding + (xi == xtiles - 1 ? prepadding_right : prepadding);
                int tile_y0 = yi * tilesize;
                int tile_y1 = std::min((yi + 1) * tilesize, height) + prepadding + (yi == ytiles - 1 ? prepadding_bottom : prepadding);

                in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, RGB_CHANNELS, sizeof(float), 1, blob_vkallocator);

                std::vector<ncnn::VkMat> bindings(2);
                bindings[0] = in_gpu;
                bindings[1] = in_tile_gpu;

                std::vector<ncnn::vk_constant_type> constants(9);
                constants[0].i = in_gpu.w;
                constants[1].i = in_gpu.h;
                constants[2].i = in_gpu.cstep;
                constants[3].i = in_tile_gpu.w;
                constants[4].i = in_tile_gpu.h;
                constants[5].i = in_tile_gpu.cstep;
                constants[6].i = std::max(prepadding - yi * tilesize, 0);
                constants[7].i = prepadding;
                constants[8].i = xi * tilesize;

                cmd.record_pipeline(waifu2x_preproc, bindings, constants, in_tile_gpu);
            }

            // waifu2x
            ncnn::VkMat out_tile_gpu;
            {
                ncnn::Extractor ex = net.create_extractor();

                ex.set_blob_vkallocator(blob_vkallocator);
                ex.set_workspace_vkallocator(blob_vkallocator);
                ex.set_staging_vkallocator(staging_vkallocator);

                ex.input("Input1", in_tile_gpu);

                if (ex.extract("Eltwise4", out_tile_gpu, cmd)) {
                    return ERROR_EXTRACTOR;
                }
            }

            // postproc
            {
                std::vector<ncnn::VkMat> bindings(2);
                bindings[0] = out_tile_gpu;
                bindings[1] = out_gpu;

                std::vector<ncnn::vk_constant_type> constants(8);
                constants[0].i = out_tile_gpu.w;
                constants[1].i = out_tile_gpu.h;
                constants[2].i = out_tile_gpu.cstep;
                constants[3].i = out_gpu.w;
                constants[4].i = out_gpu.h;
                constants[5].i = out_gpu.cstep;
                constants[6].i = xi * tilesize * scale;
                constants[7].i = out_gpu.w - xi * tilesize * scale;

                ncnn::VkMat dispatcher;
                dispatcher.w = out_gpu.w - xi * tilesize * scale;
                dispatcher.h = out_gpu.h;
                dispatcher.c = 3;

                cmd.record_pipeline(waifu2x_postproc, bindings, constants, dispatcher);
            }

            if (xtiles > 1) {
                if (cmd.submit_and_wait()) {
                    return ERROR_SUBMIT;
                }
                cmd.reset();
            }
        }

        // download
        {
            ncnn::Mat out;
            cmd.record_clone(out_gpu, out, opt);
            if (cmd.submit_and_wait()) {
                return ERROR_DOWNLOAD;
            }

            const float* out_tile_r = out.channel(0);
            const float* out_tile_g = out.channel(1);
            const float* out_tile_b = out.channel(2);
            float* dr = dstR + yi * tilesize * scale * dstStride;
            float* dg = dstG + yi * tilesize * scale * dstStride;
            float* db = dstB + yi * tilesize * scale * dstStride;
            for (int y = 0; y < out.h; y++) {
                for (int x = 0; x < out.w; x++) {
                    dr[dstStride * y + x] = std::min(1.F, std::max(0.F, out_tile_r[out.w * y + x] / 255.F));
                    dg[dstStride * y + x] = std::min(1.F, std::max(0.F, out_tile_g[out.w * y + x] / 255.F));
                    db[dstStride * y + x] = std::min(1.F, std::max(0.F, out_tile_b[out.w * y + x] / 255.F));
                }
            }
        }
    }

    net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
    net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);
    semaphore.signal(); // release only when successful
    return ERROR_OK;
}
