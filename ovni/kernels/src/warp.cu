extern "C" __global__
void warp_affine(
    const unsigned char* src, int srcWidth, int srcHeight,
    unsigned char* dst, int dstWidth, int dstHeight,
    const float* affine)  // 2x3 affine matrix flattened
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstWidth || y >= dstHeight) return;

    // Compute source coords
    float src_x = affine[0] * x + affine[1] * y + affine[2];
    float src_y = affine[3] * x + affine[4] * y + affine[5];

    int x0 = floorf(src_x);
    int y0 = floorf(src_y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if (x0 < 0 || y0 < 0 || x1 >= srcWidth || y1 >= srcHeight) {
        // Out of bounds: set black pixel (all channels)
        int dst_idx = (y * dstWidth + x) * 3;
        dst[dst_idx] = 0;
        dst[dst_idx + 1] = 0;
        dst[dst_idx + 2] = 0;
        return;
    }

    float dx = src_x - x0;
    float dy = src_y - y0;

    for (int c = 0; c < 3; c++) {
        // Index into source pixels for channel c
        int idx00 = (y0 * srcWidth + x0) * 3 + c;
        int idx01 = (y1 * srcWidth + x0) * 3 + c;
        int idx10 = (y0 * srcWidth + x1) * 3 + c;
        int idx11 = (y1 * srcWidth + x1) * 3 + c;

        float p00 = src[idx00];
        float p01 = src[idx01];
        float p10 = src[idx10];
        float p11 = src[idx11];

        float val = (1 - dx) * (1 - dy) * p00 +
                    dx * (1 - dy) * p10 +
                    (1 - dx) * dy * p01 +
                    dx * dy * p11;

        dst[(y * dstWidth + x) * 3 + c] = (unsigned char)(val);
    }
}
