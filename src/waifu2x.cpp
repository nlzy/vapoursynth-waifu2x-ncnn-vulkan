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

#include "waifu2x.hpp"

#include <algorithm>
#include <vector>

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

Waifu2x::Waifu2x(int gpuid, int precision) {
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_packed = precision == 16;
    net.opt.use_fp16_storage = precision == 16;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_int8_storage = false;
    net.opt.use_int8_arithmetic = false;

    net.set_vulkan_device(gpuid);

    waifu2x_preproc = nullptr;
    waifu2x_postproc = nullptr;
}

Waifu2x::~Waifu2x() {
    delete waifu2x_preproc;
    delete waifu2x_postproc;
}


int Waifu2x::load(const std::string& parampath, const std::string& modelpath) {
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
            waifu2x_preproc->create(waifu2x_preproc_int8s_spv_data, sizeof(waifu2x_preproc_int8s_spv_data), "waifu2x_preproc_int8s", specializations, 2, 9);
        else if (net.opt.use_fp16_storage)
            waifu2x_preproc->create(waifu2x_preproc_fp16s_spv_data, sizeof(waifu2x_preproc_fp16s_spv_data), "waifu2x_preproc_fp16s", specializations, 2, 9);
        else
            waifu2x_preproc->create(waifu2x_preproc_spv_data, sizeof(waifu2x_preproc_spv_data), "waifu2x_preproc", specializations, 2, 9);

        waifu2x_postproc = new ncnn::Pipeline(net.vulkan_device());
        waifu2x_postproc->set_optimal_local_size_xyz(32, 32, 3);
        if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
            waifu2x_postproc->create(waifu2x_postproc_int8s_spv_data, sizeof(waifu2x_postproc_int8s_spv_data), "waifu2x_postproc_int8s", specializations, 2, 8);
        else if (net.opt.use_fp16_storage)
            waifu2x_postproc->create(waifu2x_postproc_fp16s_spv_data, sizeof(waifu2x_postproc_fp16s_spv_data), "waifu2x_postproc_fp16s", specializations, 2, 8);
        else
            waifu2x_postproc->create(waifu2x_postproc_spv_data, sizeof(waifu2x_postproc_spv_data), "waifu2x_postproc", specializations, 2, 8);
    }

    return 0;
}

int Waifu2x::process(const float *srcR, const float *srcG, const float *srcB,
                     float *dstR, float *dstG, float *dstB,
                     const int srcStride, const int dstStride) const {
    int TILE_SIZE_X = tilesize;
    int TILE_SIZE_Y = tilesize;

    ncnn::VkAllocator* blob_vkallocator = net.vulkan_device()->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = net.vulkan_device()->acquire_staging_allocator();

    // prepadding
    int prepadding_bottom = prepadding;
    int prepadding_right = prepadding;
    if (scale == 1) {
        prepadding_bottom += (h + 3) / 4 * 4 - h;
        prepadding_right += (w + 3) / 4 * 4 - w;
    }
    if (scale == 2) {
        prepadding_bottom += (h + 1) / 2 * 2 - h;
        prepadding_right += (w + 1) / 2 * 2 - w;
    }

    // each tile 400x400
    int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    for (int yi = 0; yi < ytiles; yi++) {
        const int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        const int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);
        const int in_tile_w = w;
        const int in_tile_h = in_tile_y1 - in_tile_y0;

        ncnn::Mat in;
        in.create(in_tile_w, in_tile_h, RGB_CHANNELS, sizeof(float));

        float *in_tile_r = in.channel(0);
        float *in_tile_g = in.channel(1);
        float *in_tile_b = in.channel(2);
        const float *sr = srcR + in_tile_y0 * srcStride;
        const float *sg = srcG + in_tile_y0 * srcStride;
        const float *sb = srcB + in_tile_y0 * srcStride;
        for (int y = 0; y < in_tile_h; y++) {
            for (int x = 0; x < in_tile_w; x++) {
                in_tile_r[in_tile_w * y + x] = sr[srcStride * y + x] * 255.F;
                in_tile_g[in_tile_w * y + x] = sg[srcStride * y + x] * 255.F;
                in_tile_b[in_tile_w * y + x] = sb[srcStride * y + x] * 255.F;
            }
        }

        ncnn::VkCompute cmd(net.vulkan_device());

        // upload
        ncnn::VkMat in_gpu;
        {
            in_gpu.create_like(in, blob_vkallocator, staging_vkallocator);

            in_gpu.prepare_staging_buffer();
            in_gpu.upload(in);

            cmd.record_upload(in_gpu);

            if (xtiles > 1) {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        const int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        const int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

        ncnn::VkMat out_gpu;
        out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, RGB_CHANNELS, sizeof(float), blob_vkallocator, staging_vkallocator);

        for (int xi = 0; xi < xtiles; xi++) {
            // preproc
            ncnn::VkMat in_tile_gpu;
            {
                // crop tile
                int tile_x0 = xi * TILE_SIZE_X;
                int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding + prepadding_right;
                int tile_y0 = yi * TILE_SIZE_Y;
                int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding + prepadding_bottom;

                in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, RGB_CHANNELS, sizeof(float), 1, blob_vkallocator, staging_vkallocator);

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
                constants[6].i = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                constants[7].i = prepadding;
                constants[8].i = xi * TILE_SIZE_X;

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
                    return -1;
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
                constants[6].i = xi * TILE_SIZE_X * scale;
                constants[7].i = out_gpu.w - xi * TILE_SIZE_X * scale;

                ncnn::VkMat dispatcher;
                dispatcher.w = out_gpu.w - xi * TILE_SIZE_X * scale;
                dispatcher.h = out_gpu.h;
                dispatcher.c = 3;

                cmd.record_pipeline(waifu2x_postproc, bindings, constants, dispatcher);
            }

            if (xtiles > 1) {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        // download
        {
            out_gpu.prepare_staging_buffer();
            cmd.record_download(out_gpu);

            cmd.submit_and_wait();
        }

        ncnn::Mat out;
        out.create_like(out_gpu, net.opt.blob_allocator);
        out_gpu.download(out);

        const float *out_tile_r = out.channel(0);
        const float *out_tile_g = out.channel(1);
        const float *out_tile_b = out.channel(2);
        float *dr = dstR + yi * TILE_SIZE_Y * scale * dstStride;
        float *dg = dstG + yi * TILE_SIZE_Y * scale * dstStride;
        float *db = dstB + yi * TILE_SIZE_Y * scale * dstStride;
        for (int y = 0; y < out.h; y++) {
            for (int x = 0; x < out.w; x++) {
                dr[dstStride * y + x] = std::min(1.F, std::max(0.F, out_tile_r[out.w * y + x] / 255.F));
                dg[dstStride * y + x] = std::min(1.F, std::max(0.F, out_tile_g[out.w * y + x] / 255.F));
                db[dstStride * y + x] = std::min(1.F, std::max(0.F, out_tile_b[out.w * y + x] / 255.F));
            }
        }
    }

    net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
    net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}
