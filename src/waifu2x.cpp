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
#define PAD_TO_ALIGN(a, b) ((((a) + (b) - 1) / (b)) * (b) - (a))

static const uint32_t waifu2x_preproc_fp32_spv_data[] = {
    #include "waifu2x_preproc_fp32.spv.hex.h"
};
static const uint32_t waifu2x_preproc_fp16_spv_data[] = {
    #include "waifu2x_preproc_fp16.spv.hex.h"
};

static const uint32_t waifu2x_postproc_fp32_spv_data[] = {
    #include "waifu2x_postproc_fp32.spv.hex.h"
};
static const uint32_t waifu2x_postproc_fp16_spv_data[] = {
    #include "waifu2x_postproc_fp16.spv.hex.h"
};


Waifu2x::Waifu2x(int width, int height, int scale, int tilesizew, int tilesizeh, int gpuid, int gputhread,
    int precision, int prepadding, const std::string& parampath, const std::string& modelpath) :
    width(width), height(height), scale(scale), tilesizew(tilesizew), tilesizeh(tilesizeh), prepadding(prepadding), semaphore(gputhread)
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

    std::vector<ncnn::vk_specialization_type> specializations;
    waifu2x_preproc = new ncnn::Pipeline(net.vulkan_device());
    waifu2x_preproc->set_optimal_local_size_xyz(8, 8, 3);
    if (net.opt.use_fp16_storage)
        waifu2x_preproc->create(waifu2x_preproc_fp16_spv_data, sizeof(waifu2x_preproc_fp16_spv_data), specializations);
    else
        waifu2x_preproc->create(waifu2x_preproc_fp32_spv_data, sizeof(waifu2x_preproc_fp32_spv_data), specializations);

    waifu2x_postproc = new ncnn::Pipeline(net.vulkan_device());
    waifu2x_postproc->set_optimal_local_size_xyz(8, 8, 3);
    if (net.opt.use_fp16_storage)
        waifu2x_postproc->create(waifu2x_postproc_fp16_spv_data, sizeof(waifu2x_postproc_fp16_spv_data), specializations);
    else
        waifu2x_postproc->create(waifu2x_postproc_fp32_spv_data, sizeof(waifu2x_postproc_fp32_spv_data), specializations);
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

    const int xtiles = DIV_CEIL(width, tilesizew);
    const int ytiles = DIV_CEIL(height, tilesizeh);

    for (int yi = 0; yi < ytiles; yi++) {
        ncnn::VkCompute cmd(net.vulkan_device());

        const int tile_nopad_y0 = yi * tilesizeh;
        const int tile_nopad_y1 = std::min(tile_nopad_y0 + tilesizeh, height);
        const int tile_nopad_h = tile_nopad_y1 - tile_nopad_y0;
        const int prepadding_bottom = prepadding + PAD_TO_ALIGN(tile_nopad_h, 4 / scale);
        const int tile_pad_y0 = std::max(tile_nopad_y0 - prepadding, 0);
        const int tile_pad_y1 = std::min(tile_nopad_y1 + prepadding_bottom, height);
        const int tile_pad_h = tile_pad_y1 - tile_pad_y0;


        // upload
        ncnn::Mat in(width, tile_pad_h, RGB_CHANNELS, sizeof(float));
        for (int y = 0; y < tile_pad_h; y++) {
            memcpy((float*)in.channel(0) + y * width, srcR + (y + tile_pad_y0) * srcStride, sizeof(float) * width);
            memcpy((float*)in.channel(1) + y * width, srcG + (y + tile_pad_y0) * srcStride, sizeof(float) * width);
            memcpy((float*)in.channel(2) + y * width, srcB + (y + tile_pad_y0) * srcStride, sizeof(float) * width);
        }
        
        ncnn::VkMat in_gpu;
        cmd.record_clone(in, in_gpu, opt);
        if (xtiles > 1) {
            if (cmd.submit_and_wait()) {
                return ERROR_UPLOAD;
            }
            cmd.reset();
        }


        ncnn::VkMat out_gpu;
        out_gpu.create(width * scale, tile_nopad_h * scale, RGB_CHANNELS, sizeof(float), blob_vkallocator);

        for (int xi = 0; xi < xtiles; xi++) {
            const int tile_nopad_x0 = xi * tilesizew;
            const int tile_nopad_x1 = std::min(tile_nopad_x0 + tilesizew, width);
            const int tile_nopad_w = tile_nopad_x1 - tile_nopad_x0;
            const int prepadding_right = prepadding + PAD_TO_ALIGN(tile_nopad_w, 4 / scale);

            ncnn::VkMat in_tile_gpu(tile_nopad_x1 - tile_nopad_x0 + prepadding + prepadding_right,
                                    tile_nopad_y1 - tile_nopad_y0 + prepadding + prepadding_bottom,
                                    RGB_CHANNELS, net.opt.use_fp16_storage ? 2u : 4u, 1, blob_vkallocator);

            // preproc
            {
                std::vector<ncnn::VkMat> bindings(2);
                bindings[0] = in_gpu;
                bindings[1] = in_tile_gpu;

                std::vector<ncnn::vk_constant_type> constants(10);
                constants[0].i = in_gpu.w;
                constants[1].i = in_gpu.h;
                constants[2].i = in_gpu.cstep;
                constants[3].i = in_tile_gpu.w;
                constants[4].i = in_tile_gpu.h;
                constants[5].i = in_tile_gpu.cstep;
                constants[6].i = prepadding;
                constants[7].i = prepadding;
                constants[8].i = tile_nopad_x0;
                constants[9].i = std::min(tile_nopad_y0, prepadding);

                ncnn::VkMat dispatcher;
                dispatcher.w = in_tile_gpu.w;
                dispatcher.h = in_tile_gpu.h;
                dispatcher.c = RGB_CHANNELS;

                cmd.record_pipeline(waifu2x_preproc, bindings, constants, dispatcher);
            }


            // waifu2x
            ncnn::VkMat out_tile_gpu;

            ncnn::Extractor ex = net.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("Input1", in_tile_gpu);

            if (ex.extract("Eltwise4", out_tile_gpu, cmd)) {
                return ERROR_EXTRACTOR;
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
                constants[6].i = tile_nopad_x0 * scale;
                constants[7].i = std::min(out_gpu.w - tile_nopad_x0 * scale, tilesizew * scale);

                ncnn::VkMat dispatcher;
                dispatcher.w = std::min(out_gpu.w - tile_nopad_x0 * scale, tilesizew * scale);
                dispatcher.h = out_gpu.h;
                dispatcher.c = RGB_CHANNELS;

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

            for (int y = 0; y < out.h; y++) {
                memcpy(dstR + tile_nopad_y0 * scale * dstStride + y * dstStride, (float *)out.channel(0) + y * out.w, out.w * sizeof(float));
                memcpy(dstG + tile_nopad_y0 * scale * dstStride + y * dstStride, (float *)out.channel(1) + y * out.w, out.w * sizeof(float));
                memcpy(dstB + tile_nopad_y0 * scale * dstStride + y * dstStride, (float *)out.channel(2) + y * out.w, out.w * sizeof(float));
            }
        }
    }

    net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
    net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);
    semaphore.signal(); // release only when successful
    return ERROR_OK;
}
