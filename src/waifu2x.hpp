#ifndef WAIFU2X_HPP
#define WAIFU2X_HPP

#define RGB_CHANNELS 3

#include <string>
#include <mutex>
#include <condition_variable>
#include "net.h"
#include "gpu.h"

class Waifu2x
{
public:
    Waifu2x(int width, int height, int scale, int tilesizew, int tilesizeh, int gpuid, int gputhread,
            int precision, int prepadding, const std::string& parampath, const std::string& modelpath);
    ~Waifu2x();

    int process(const float *srcR, const float *srcG, const float *srcB, float *dstR, float *dstG, float *dstB, ptrdiff_t srcStride, ptrdiff_t dstStride) const;

    enum {
        ERROR_OK = 0,
        ERROR_EXTRACTOR = -1,
        ERROR_SUBMIT = -2,
        ERROR_UPLOAD = -3,
        ERROR_DOWNLOAD = -4
    };

private:
    int width;
    int height;
    int scale;
    int tilesizew;
    int tilesizeh;
    int prepadding;

    ncnn::Net net;
    ncnn::Pipeline* waifu2x_preproc;
    ncnn::Pipeline* waifu2x_postproc;

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

    mutable Semaphore semaphore;
};

#endif
