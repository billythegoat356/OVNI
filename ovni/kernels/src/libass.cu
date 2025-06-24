#include <stdint.h>

extern "C" __global__
void blend_ass_image(
    int w, int h,
    int stride,
    const unsigned char* bitmap,
    uint32_t color,
    int dst_x, int dst_y,
    unsigned char* dst
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

}
